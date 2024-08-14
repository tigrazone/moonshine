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
    const float3 lightDirWs = normalize(lightSample.connection);

    if (lightSample.pdf > 0.0) {
        const float3 lightDirFs = frame.worldToFrame(lightDirWs);
        const float scatteringPdf = material.pdf(lightDirFs, outgoingDirFs);
        if (scatteringPdf > 0.0) {
            const float3 brdf = material.eval(lightDirFs, outgoingDirFs);
            const float weight = misWeight(lightSamplesTaken, lightSample.pdf, brdfSamplesTaken, scatteringPdf);
            const float3 totalRadiance = lightSample.radiance * brdf * abs(Frame::cosTheta(lightDirFs)) * weight / lightSample.pdf;
            if (any(totalRadiance != 0) && !ShadowIntersection::hit(accel, offsetAlongNormal(positionWs, faceForward(triangleNormalDirWs, lightDirWs)), lightSample.connection)) {
                return totalRadiance;
            }
        }
    }

    return 0;
}

// selects a shading normal based on the most preferred normal that is plausible
Frame selectFrame(const SurfacePoint surface, const Material material, const float3 outgoingDirWs) {
    const Frame textureFrame = material.getTextureFrame(surface.texcoord, surface.frame);
    Frame shadingFrame;
    if (sign(dot(surface.triangleFrame.n, outgoingDirWs)) == sign(dot(outgoingDirWs, textureFrame.n))) {
        // prefer texture normal if we can
        shadingFrame = textureFrame;
    } else if (sign(dot(surface.triangleFrame.n, outgoingDirWs)) == sign(dot(outgoingDirWs, surface.frame.n))) {
        // if texture normal not valid, try shading normal
        shadingFrame = surface.frame;
    } else {
        // otherwise fall back to triangle normal
        shadingFrame = surface.triangleFrame;
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
        float lastMaterialPdf = 1;
        bool isLastMaterialDelta = false;

        // main path tracing loop
        for (Intersection its = Intersection::find(scene.tlas, ray); its.hit(); its = Intersection::find(scene.tlas, ray)) {

            // decode mesh attributes and material from intersection
            const SurfacePoint surface = scene.world.surfacePoint(its.instanceIndex, its.geometryIndex, its.primitiveIndex, its.barycentrics);
            const Material material = scene.world.material(its.instanceIndex, its.geometryIndex);
            const PolymorphicBSDF bsdf = PolymorphicBSDF::load(material, surface.texcoord);

            const float3 outgoingDirWs = -ray.Direction;
            const Frame shadingFrame = selectFrame(surface, material, outgoingDirWs);
            const float3 outgoingDirSs = shadingFrame.worldToFrame(outgoingDirWs);

            // collect light from emissive meshes
            {
                const bool canMeshSample = bounceCount != 0 && !isLastMaterialDelta;
                const float lightPdf = areaMeasureToSolidAngleMeasure(surface.position, ray.Origin, ray.Direction, surface.triangleFrame.n) * scene.meshLights.areaPdf(its.instanceIndex, its.geometryIndex, its.primitiveIndex);
                const float weight = misWeight(1, lastMaterialPdf, canMeshSample ? meshSamplesPerBounce : 0, lightPdf);
                accumulatedColor += throughput * material.getEmissive(surface.texcoord) * weight;
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
                    accumulatedColor += throughput * estimateDirectMISLight(scene.tlas, shadingFrame, scene.envMap, bsdf, outgoingDirSs, surface.position, surface.triangleFrame.n, rand, envSamplesPerBounce, 1);
                }

                // accumulate direct light samples from emissive meshes
                for (uint directCount = 0; directCount < meshSamplesPerBounce; directCount++) {
                    float2 rand = float2(rng.getFloat(), rng.getFloat());
                    accumulatedColor += throughput * estimateDirectMISLight(scene.tlas, shadingFrame, scene.meshLights, bsdf, outgoingDirSs, surface.position, surface.triangleFrame.n, rand, meshSamplesPerBounce, 1);
                }
            }

            // sample direction for next bounce
            const BSDFSample sample = bsdf.sample(outgoingDirSs, float2(rng.getFloat(), rng.getFloat()));
            if (sample.pdf == 0.0) return accumulatedColor; // in a perfect world this would never happen

            // set up info for next bounce
            ray.Direction = shadingFrame.frameToWorld(sample.dirFs);
            ray.Origin = offsetAlongNormal(surface.position, faceForward(surface.triangleFrame.n, ray.Direction));
            throughput *= bsdf.eval(sample.dirFs, outgoingDirSs) * abs(Frame::cosTheta(sample.dirFs)) / sample.pdf;
            bounceCount += 1;
            isLastMaterialDelta = isCurrentMaterialDelta;
            lastMaterialPdf = sample.pdf;
        }

        // we only get here on misses -- terminations for other reasons return from loop

        // handle env map
        {
            const bool canEnvSample = bounceCount != 0 && !isLastMaterialDelta;
            const LightEval l = scene.envMap.eval(ray.Direction);
            const float weight = misWeight(1, lastMaterialPdf, canEnvSample ? envSamplesPerBounce : 0, l.pdf);
            accumulatedColor += throughput * l.radiance * weight;
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
            const SurfacePoint surface = scene.world.surfacePoint(its.instanceIndex, its.geometryIndex, its.primitiveIndex, its.barycentrics);
            const Material material = scene.world.material(its.instanceIndex, its.geometryIndex);
            const PolymorphicBSDF bsdf = PolymorphicBSDF::load(material, surface.texcoord);

            const float3 outgoingDirWs = -initialRay.Direction;
            const Frame shadingFrame = selectFrame(surface, material, outgoingDirWs);
            const float3 outgoingDirSs = shadingFrame.worldToFrame(outgoingDirWs);

            // collect light from emissive meshes
            accumulatedColor += material.getEmissive(surface.texcoord);

            const bool isMaterialDelta = bsdf.isDelta();

            if (!isMaterialDelta) {
                // accumulate direct light samples from env map
                for (uint directCount = 0; directCount < envSamples; directCount++) {
                    float2 rand = float2(rng.getFloat(), rng.getFloat());
                    accumulatedColor += estimateDirectMISLight(scene.tlas, shadingFrame, scene.envMap, bsdf, outgoingDirSs, surface.position, surface.triangleFrame.n, rand, envSamples, brdfSamples);
                }

                // accumulate direct light samples from emissive meshes
                for (uint directCount = 0; directCount < meshSamples; directCount++) {
                    float2 rand = float2(rng.getFloat(), rng.getFloat());
                    accumulatedColor += estimateDirectMISLight(scene.tlas, shadingFrame, scene.meshLights, bsdf, outgoingDirSs, surface.position, surface.triangleFrame.n, rand, meshSamples, brdfSamples);
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
                        ray.Origin = offsetAlongNormal(surface.position, faceForward(surface.triangleFrame.n, ray.Direction));
                        Intersection its = Intersection::find(scene.tlas, ray);
                        if (its.hit()) {
                            // hit -- collect light from emissive meshes
                            const SurfacePoint surface = scene.world.surfacePoint(its.instanceIndex, its.geometryIndex, its.primitiveIndex, its.barycentrics);
                            const float lightPdf = areaMeasureToSolidAngleMeasure(surface.position, ray.Origin, ray.Direction, surface.triangleFrame.n) * scene.meshLights.areaPdf(its.instanceIndex, its.geometryIndex, its.primitiveIndex);
                            const float weight = misWeight(brdfSamples, sample.pdf, !isMaterialDelta ? meshSamples : 0, lightPdf);
                            accumulatedColor += throughput * scene.world.material(its.instanceIndex, its.geometryIndex).getEmissive(surface.texcoord) * weight;
                        } else {
                            // miss -- collect light from env map
                            const LightEval l = scene.envMap.eval(ray.Direction);
                            const float weight = misWeight(brdfSamples, sample.pdf, !isMaterialDelta ? envSamples : 0, l.pdf);
                            accumulatedColor += throughput * l.radiance * weight;
                        }
                    }
                }
            }
        } else {
            // add background color
            accumulatedColor += scene.envMap.eval(initialRay.Direction).radiance;
        }

        return accumulatedColor;
    }
};
