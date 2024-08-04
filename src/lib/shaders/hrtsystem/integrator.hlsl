#pragma once

#include "../utils/math.hlsl"
#include "../utils/random.hlsl"
#include "reflection_frame.hlsl"
#include "material.hlsl"
#include "world.hlsl"
#include "light.hlsl"

// with
//   power == 1 this becomes balance heuristic
//   power == 0 this becomes uniform weighting
float powerHeuristic(const uint fCount, const float fPdf, const uint gCount, const float gPdf, const uint power) {
    return pow(fPdf, power) / (fCount * pow(fPdf, power) + gCount * pow(gPdf, power));
}

float misWeight(const uint fCount, const float fPdf, const uint gCount, const float gPdf) {
    return powerHeuristic(fCount, fPdf, gCount, gPdf, 2);
}

// estimates direct lighting from light + brdf via MIS
// only samples light
template <class Light, class BSDF>
float3 estimateDirectMISLight(RaytracingAccelerationStructure accel, Frame frame, Light light, BSDF material, float3 outgoingDirFs, float3 positionWs, float3 triangleNormalDirWs, float2 rand, uint lightSamplesTaken, uint brdfSamplesTaken) {
    const LightSample lightSample = light.sample(positionWs, triangleNormalDirWs, rand);

    if (lightSample.pdf > 0.0) {
        const float3 lightDirFs = frame.worldToFrame(lightSample.dirWs);
        const float scatteringPdf = material.pdf(lightDirFs, outgoingDirFs);
        if (scatteringPdf > 0.0) {
            const float3 brdf = material.eval(lightDirFs, outgoingDirFs);
            const float weight = misWeight(lightSamplesTaken, lightSample.pdf, brdfSamplesTaken, scatteringPdf);
            const float3 totalRadiance = lightSample.radiance * brdf * abs(Frame::cosTheta(lightDirFs)) * weight / lightSample.pdf;
            if (any(totalRadiance != 0) && !ShadowIntersection::hit(accel, offsetAlongNormal(positionWs, faceForward(triangleNormalDirWs, lightSample.dirWs)), lightSample.dirWs, lightSample.lightDistance)) {
                return totalRadiance;
            }
        }
    }

    return 0;
}

// no MIS, just light
template <class Light, class BSDF>
float3 estimateDirect(RaytracingAccelerationStructure accel, Frame frame, Light light, BSDF material, float3 outgoingDirFs, float3 positionWs, float3 triangleNormalDirWs, float2 rand) {
    const LightSample lightSample = light.sample(positionWs, triangleNormalDirWs, rand);
    const float3 lightDirFs = frame.worldToFrame(lightSample.dirWs);

    if (lightSample.pdf > 0.0) {
        const float3 brdf = material.eval(lightDirFs, outgoingDirFs);
        const float3 totalRadiance = lightSample.radiance * brdf * abs(Frame::cosTheta(lightDirFs)) / lightSample.pdf;
        if (any(totalRadiance != 0) && !ShadowIntersection::hit(accel, offsetAlongNormal(positionWs, faceForward(triangleNormalDirWs, lightSample.dirWs)), lightSample.dirWs, lightSample.lightDistance)) {
            return totalRadiance;
        }
    }

    return 0;
}

// selects a shading normal based on the most preferred normal that is plausible
Frame selectFrame(const MeshAttributes attrs, const Material material, const float3 outgoingDirWs) {
    const Frame textureFrame = material.getTextureFrame(attrs.texcoord, attrs.frame);
    Frame shadingFrame;
    if (sign(dot(attrs.triangleFrame.n, outgoingDirWs)) == sign(dot(outgoingDirWs, textureFrame.n))) {
        // prefer texture normal if we can
        shadingFrame = textureFrame;
    } else if (sign(dot(attrs.triangleFrame.n, outgoingDirWs)) == sign(dot(outgoingDirWs, attrs.frame.n))) {
        // if texture normal not valid, try shading normal
        shadingFrame = attrs.frame;
    } else {
        // otherwise fall back to triangle normal
        shadingFrame = attrs.triangleFrame;
    }

    return shadingFrame;
}

interface Integrator {
    float3 incomingRadiance(Scene scene, RayDesc ray, inout Rng rng);
};

struct PathTracingIntegrator : Integrator {
    uint maxBounces;
    uint envSamplesPerBounce;
    uint meshSamplesPerBounce;

    static PathTracingIntegrator create(uint maxBounces, uint envSamplesPerBounce, uint meshSamplesPerBounce) {
        PathTracingIntegrator integrator;
        integrator.maxBounces = maxBounces;
        integrator.envSamplesPerBounce = envSamplesPerBounce;
        integrator.meshSamplesPerBounce = meshSamplesPerBounce;
        return integrator;
    }

    float3 incomingRadiance(Scene scene, RayDesc initialRay, inout Rng rng) {
        float3 accumulatedColor = float3(0.0, 0.0, 0.0);

        // state updated at each bounce
        RayDesc ray = initialRay;
        float3 throughput = float3(1.0, 1.0, 1.0);
        uint bounceCount = 0;
        float lastMaterialPdf;
        bool isLastMaterialDelta = false;

        // main path tracing loop
        for (Intersection its = Intersection::find(scene.tlas, ray); its.hit(); its = Intersection::find(scene.tlas, ray)) {

            // decode mesh attributes and material from intersection
            const MeshAttributes attrs = MeshAttributes::lookupAndInterpolate(scene.world, its.instanceIndex, its.geometryIndex, its.primitiveIndex, its.barycentrics).inWorld(scene.world, its.instanceIndex);
            const Material material = scene.world.material(its.instanceIndex, its.geometryIndex);
            const PolymorphicBSDF bsdf = PolymorphicBSDF::load(material, attrs.texcoord);

            const float3 outgoingDirWs = -ray.Direction;
            const Frame shadingFrame = selectFrame(attrs, material, outgoingDirWs);
            const float3 outgoingDirSs = shadingFrame.worldToFrame(outgoingDirWs);

            // collect light from emissive meshes
            // lights only emit from front face
            if (dot(outgoingDirWs, attrs.triangleFrame.n) > 0.0) {
                const float3 emissiveLight = material.getEmissive(attrs.texcoord);
                const float areaPdf = scene.meshLights.areaPdf(its.instanceIndex, its.geometryIndex, its.primitiveIndex);
                if (meshSamplesPerBounce == 0 || bounceCount == 0 || isLastMaterialDelta || areaPdf == 0) {
                    // add emissive light at point if light not explicitly sampled or initial bounce
                    accumulatedColor += throughput * emissiveLight;
                } else {
                    // MIS emissive light if it is sampled at later bounces
                    const float lightPdf = areaMeasureToSolidAngleMeasure(attrs.position, ray.Origin, ray.Direction, attrs.triangleFrame.n) * areaPdf;
                    const float weight = misWeight(1, lastMaterialPdf, meshSamplesPerBounce, lightPdf);
                    accumulatedColor += throughput * emissiveLight * weight;
                }
            }

            // possibly terminate if reached max bounce cutoff or lose at russian roulette
            // this needs to be before NEE below otherwise MIS would need to be adjusted
            if (bounceCount >= maxBounces + 1) {
                return accumulatedColor;
            } else if (bounceCount > 3) {
                // russian roulette
                float pSurvive = min(0.95, luminance(throughput));
                if (rng.getFloat() > pSurvive) return accumulatedColor;
                throughput /= pSurvive;
            }

            const bool isCurrentMaterialDelta = bsdf.isDelta();

            if (!isCurrentMaterialDelta) {
                // accumulate direct light samples from env map
                for (uint directCount = 0; directCount < envSamplesPerBounce; directCount++) {
                    float2 rand = float2(rng.getFloat(), rng.getFloat());
                    accumulatedColor += throughput * estimateDirectMISLight(scene.tlas, shadingFrame, scene.envMap, bsdf, outgoingDirSs, attrs.position, attrs.triangleFrame.n, rand, envSamplesPerBounce, 1);
                }

                // accumulate direct light samples from emissive meshes
                for (uint directCount = 0; directCount < meshSamplesPerBounce; directCount++) {
                    float2 rand = float2(rng.getFloat(), rng.getFloat());
                    accumulatedColor += throughput * estimateDirectMISLight(scene.tlas, shadingFrame, scene.meshLights, bsdf, outgoingDirSs, attrs.position, attrs.triangleFrame.n, rand, meshSamplesPerBounce, 1);
                }
            }

            // sample direction for next bounce
            const BSDFSample sample = bsdf.sample(outgoingDirSs, float2(rng.getFloat(), rng.getFloat()));
            if (sample.pdf == 0.0) return accumulatedColor; // in a perfect world this would never happen

            // set up info for next bounce
            ray.Direction = shadingFrame.frameToWorld(sample.dirFs);
            ray.Origin = offsetAlongNormal(attrs.position, faceForward(attrs.triangleFrame.n, ray.Direction));
            throughput *= bsdf.eval(sample.dirFs, outgoingDirSs) * abs(Frame::cosTheta(sample.dirFs)) / sample.pdf;
            bounceCount += 1;
            isLastMaterialDelta = isCurrentMaterialDelta;
            lastMaterialPdf = sample.pdf;
        }

        // we only get here on misses -- terminations for other reasons return from loop

        // handle env map
        if (envSamplesPerBounce == 0 || bounceCount == 0 || isLastMaterialDelta) {
            // add background color if it isn't explicitly sampled or this is a primary ray
            accumulatedColor += throughput * scene.envMap.incomingRadiance(ray.Direction);
        } else {
            // MIS env map if it is sampled at later bounces
            LightEval l = scene.envMap.eval(ray.Direction);

            if (l.pdf > 0.0) {
                float weight = misWeight(1, lastMaterialPdf, envSamplesPerBounce, l.pdf);
                accumulatedColor += throughput * l.radiance * weight;
            }
        }

        return accumulatedColor;
    }
};

// primary ray + light sample
// same as above with max_bounces = 0, but simpler code
struct DirectLightIntegrator : Integrator {
    uint envSamples;
    uint meshSamples;
    uint brdfSamples;

    static DirectLightIntegrator create(uint envSamples, uint meshSamples, uint brdfSamples) {
        DirectLightIntegrator integrator;
        integrator.envSamples = envSamples;
        integrator.meshSamples = meshSamples;
        integrator.brdfSamples = brdfSamples;
        return integrator;
    }

    float3 incomingRadiance(Scene scene, RayDesc initialRay, inout Rng rng) {
        float3 accumulatedColor = float3(0.0, 0.0, 0.0);

        Intersection its = Intersection::find(scene.tlas, initialRay);
        if (its.hit()) {
            // decode mesh attributes and material from intersection
            const MeshAttributes attrs = MeshAttributes::lookupAndInterpolate(scene.world, its.instanceIndex, its.geometryIndex, its.primitiveIndex, its.barycentrics).inWorld(scene.world, its.instanceIndex);
            const Material material = scene.world.material(its.instanceIndex, its.geometryIndex);
            const PolymorphicBSDF bsdf = PolymorphicBSDF::load(material, attrs.texcoord);

            const float3 outgoingDirWs = -initialRay.Direction;
            const Frame shadingFrame = selectFrame(attrs, material, outgoingDirWs);
            const float3 outgoingDirSs = shadingFrame.worldToFrame(outgoingDirWs);

            // collect light from emissive meshes
            accumulatedColor += material.getEmissive(attrs.texcoord);

            const bool isMaterialDelta = bsdf.isDelta();

            if (!isMaterialDelta) {
                // accumulate direct light samples from env map
                for (uint directCount = 0; directCount < envSamples; directCount++) {
                    float2 rand = float2(rng.getFloat(), rng.getFloat());
                    accumulatedColor += estimateDirectMISLight(scene.tlas, shadingFrame, scene.envMap, bsdf, outgoingDirSs, attrs.position, attrs.triangleFrame.n, rand, envSamples, brdfSamples);
                }

                // accumulate direct light samples from emissive meshes
                for (uint directCount = 0; directCount < meshSamples; directCount++) {
                    float2 rand = float2(rng.getFloat(), rng.getFloat());
                    accumulatedColor += estimateDirectMISLight(scene.tlas, shadingFrame, scene.meshLights, bsdf, outgoingDirSs, attrs.position, attrs.triangleFrame.n, rand, meshSamples, brdfSamples);
                }
            }

            for (uint brdfSampleCount = 0; brdfSampleCount < brdfSamples; brdfSampleCount++) {
                const BSDFSample sample = bsdf.sample(outgoingDirSs, float2(rng.getFloat(), rng.getFloat()));
                if (sample.pdf > 0.0) {
                    const float3 throughput = bsdf.eval(outgoingDirSs, sample.dirFs) * abs(Frame::cosTheta(sample.dirFs)) / sample.pdf;
                    if (all(throughput != 0)) {
                        RayDesc ray;
                        ray.TMin = 0;
                        ray.TMax = INFINITY;
                        ray.Direction = shadingFrame.frameToWorld(sample.dirFs);
                        ray.Origin = offsetAlongNormal(attrs.position, faceForward(attrs.triangleFrame.n, ray.Direction));
                        Intersection its = Intersection::find(scene.tlas, ray);
                        if (its.hit()) {
                            // decode mesh attributes and material from intersection
                            const MeshAttributes attrs = MeshAttributes::lookupAndInterpolate(scene.world, its.instanceIndex, its.geometryIndex, its.primitiveIndex, its.barycentrics).inWorld(scene.world, its.instanceIndex);
                            const float3 emissiveLight = scene.world.material(its.instanceIndex, its.geometryIndex).getEmissive(attrs.texcoord);

                            // collect light from emissive meshes
                            // lights only emit from front face
                            if (dot(-ray.Direction, attrs.triangleFrame.n) > 0.0) {
                                const float areaPdf = scene.meshLights.areaPdf(its.instanceIndex, its.geometryIndex, its.primitiveIndex);
                                if (meshSamples == 0 || isMaterialDelta || areaPdf == 0) {
                                    // add emissive light at point if light not explicitly sampled or initial bounce
                                    accumulatedColor += throughput * emissiveLight / brdfSamples;
                                } else {
                                    // MIS emissive light if it is sampled at later bounces
                                    const float lightPdf = areaMeasureToSolidAngleMeasure(attrs.position, ray.Origin, ray.Direction, attrs.triangleFrame.n) * areaPdf;
                                    const float weight = misWeight(brdfSamples, sample.pdf, meshSamples, lightPdf);
                                    accumulatedColor += throughput * emissiveLight * weight;
                                }
                            }
                        } else {
                            if (envSamples == 0 || isMaterialDelta) {
                                accumulatedColor += throughput * scene.envMap.incomingRadiance(ray.Direction) / brdfSamples;
                            } else {
                                LightEval l = scene.envMap.eval(ray.Direction);
                                if (l.pdf > 0) {
                                    const float weight = misWeight(brdfSamples, sample.pdf, envSamples, l.pdf);
                                    accumulatedColor += throughput * scene.envMap.incomingRadiance(ray.Direction) * weight;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // add background color
            accumulatedColor += scene.envMap.incomingRadiance(initialRay.Direction);
        }

        return accumulatedColor;
    }
};
