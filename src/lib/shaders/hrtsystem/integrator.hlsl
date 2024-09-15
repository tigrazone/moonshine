#pragma once

#include "../utils/math.hlsl"
#include "../utils/random.hlsl"
#include "reflection_frame.hlsl"
#include "material.hlsl"
#include "world.hlsl"
#include "light.hlsl"
#include "ray.hlsl"

// with
//   power == 1 this becomes balance heuristic
//   power == 0 this becomes uniform weighting
float powerHeuristic(const uint fCount, const float fPdf, const uint gCount, const float gPdf, const uint power) {
    return pow(fPdf, power) / (fCount * pow(fPdf, power) + gCount * pow(gPdf, power));
}

float misWeight(const uint fCount, const float fPdf, const uint gCount, const float gPdf) {
    if (fPdf == 1.#INF) return 1.0 / fCount; // delta distribution for f, g not relevant
    return powerHeuristic(fCount, fPdf, gCount, gPdf, 2);
}

// estimates direct lighting from light + brdf via MIS
// only samples light
template <class Light, class BSDF>
float3 estimateDirectMISLight(RaytracingAccelerationStructure accel, Frame frame, Light light, BSDF material, float3 outgoingDirFs, float3 positionWs, float3 triangleNormalDirWs, float spawnOffset, float2 rand, uint lightSamplesTaken, uint brdfSamplesTaken) {
    const LightSample lightSample = light.sample(positionWs, triangleNormalDirWs, rand);
    const float3 lightDirWs = normalize(lightSample.connection);

    if (lightSample.pdf > 0.0) {
        const float3 lightDirFs = frame.worldToFrame(lightDirWs);
        const BSDFEvaluation bsdfEval = material.evaluate(lightDirFs, outgoingDirFs);
        if (bsdfEval.pdf > 0.0) {
            const float weight = misWeight(lightSamplesTaken, lightSample.pdf, brdfSamplesTaken, bsdfEval.pdf);
            const float3 totalRadiance = lightSample.radiance * bsdfEval.reflectance * abs(Frame::cosTheta(lightDirFs)) * weight;
            if (any(totalRadiance != 0) && !ShadowIntersection::hit(accel, positionWs + faceForward(triangleNormalDirWs, lightDirWs) * spawnOffset, lightSample.connection - faceForward(triangleNormalDirWs, lightDirWs) * spawnOffset)) {
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

struct Path {
    Ray ray;
    float3 throughput;
    float3 radiance;
    uint bounceCount;

    static Path create(const Ray ray) {
        Path p;
        p.ray = ray;
        p.throughput = 1.0;
        p.radiance = 0.0;
        p.bounceCount = 0;
        return p;
    }
};

interface Integrator {
    float3 incomingRadiance(const Scene scene, const Ray initialRay, inout Rng rng);
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

    float3 incomingRadiance(const Scene scene, const Ray initialRay, inout Rng rng) {
        Path path = Path::create(initialRay);

        for (Intersection its = Intersection::find(scene.tlas, path.ray.desc()); its.hit(); its = Intersection::find(scene.tlas, path.ray.desc())) {

            // decode mesh attributes and material from intersection
            const SurfacePoint surface = scene.world.surfacePoint(its.instanceIndex, its.geometryIndex, its.primitiveIndex, its.barycentrics);
            const Material material = scene.world.material(its.instanceIndex, its.geometryIndex);
            const PolymorphicBSDF bsdf = PolymorphicBSDF::load(material, surface.texcoord);

            const float3 outgoingDirWs = -path.ray.direction;
            const Frame shadingFrame = selectFrame(surface, material, outgoingDirWs);
            const float3 outgoingDirSs = shadingFrame.worldToFrame(outgoingDirWs);

            // collect light from emissive meshes
            {
                const float lightPdf = areaMeasureToSolidAngleMeasure(surface.position, path.ray.origin, path.ray.direction, surface.triangleFrame.n) * scene.meshLights.areaPdf(its.instanceIndex, its.geometryIndex, its.primitiveIndex);
                const float weight = misWeight(1, path.ray.pdf, meshSamplesPerBounce, lightPdf);
                path.radiance += path.throughput * material.getEmissive(surface.texcoord) * weight;
            }

            // possibly terminate if reached max bounce cutoff or lose at russian roulette
            // this needs to be before NEE below otherwise MIS would need to be adjusted
            if (path.bounceCount >= maxBounces + 1) {
                return path.radiance;
            } else if (path.bounceCount > 3) {
                // russian roulette
                float pSurvive = min(0.95, luminance(path.throughput));
                if (rng.getFloat() > pSurvive) return path.radiance;
                path.throughput /= pSurvive;
            }

            if (!bsdf.isDelta()) {
                // accumulate direct light samples from env map
                for (uint directCount = 0; directCount < envSamplesPerBounce; directCount++) {
                    float2 rand = float2(rng.getFloat(), rng.getFloat());
                    path.radiance += path.throughput * estimateDirectMISLight(scene.tlas, shadingFrame, scene.envMap, bsdf, outgoingDirSs, surface.position, surface.triangleFrame.n, surface.spawnOffset, rand, envSamplesPerBounce, 1);
                }

                // accumulate direct light samples from emissive meshes
                for (uint directCount = 0; directCount < meshSamplesPerBounce; directCount++) {
                    float2 rand = float2(rng.getFloat(), rng.getFloat());
                    path.radiance += path.throughput * estimateDirectMISLight(scene.tlas, shadingFrame, scene.meshLights, bsdf, outgoingDirSs, surface.position, surface.triangleFrame.n, surface.spawnOffset, rand, meshSamplesPerBounce, 1);
                }
            }

            // sample direction for next bounce
            const BSDFSample sample = bsdf.sample(outgoingDirSs, float2(rng.getFloat(), rng.getFloat()));
            if (sample.eval.pdf == 0.0) return path.radiance; // in a perfect world this would never happen

            // set up info for next bounce
            path.ray.direction = shadingFrame.frameToWorld(sample.dirFs);
            path.ray.origin = surface.position + faceForward(surface.triangleFrame.n, path.ray.direction) * surface.spawnOffset;
            path.ray.pdf = sample.eval.pdf;
            path.throughput *= sample.eval.reflectance * abs(Frame::cosTheta(sample.dirFs));
            path.bounceCount += 1;
        }

        // we only get here on misses -- terminations for other reasons return from loop

        // handle env map
        {
            const LightEvaluation l = scene.envMap.evaluate(path.ray.direction);
            const float weight = misWeight(1, path.ray.pdf, envSamplesPerBounce, l.pdf);
            path.radiance += path.throughput * l.radiance * weight;
        }

        return path.radiance;
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

    float3 incomingRadiance(const Scene scene, const Ray initialRay, inout Rng rng) {
        float3 pathRadiance = float3(0.0, 0.0, 0.0);

        Intersection its = Intersection::find(scene.tlas, initialRay.desc());
        if (its.hit()) {
            // decode mesh attributes and material from intersection
            const SurfacePoint surface = scene.world.surfacePoint(its.instanceIndex, its.geometryIndex, its.primitiveIndex, its.barycentrics);
            const Material material = scene.world.material(its.instanceIndex, its.geometryIndex);
            const PolymorphicBSDF bsdf = PolymorphicBSDF::load(material, surface.texcoord);

            const float3 outgoingDirWs = -initialRay.direction;
            const Frame shadingFrame = selectFrame(surface, material, outgoingDirWs);
            const float3 outgoingDirSs = shadingFrame.worldToFrame(outgoingDirWs);

            // collect light from emissive meshes
            pathRadiance += material.getEmissive(surface.texcoord);

            if (!bsdf.isDelta()) {
                // accumulate direct light samples from env map
                for (uint directCount = 0; directCount < envSamples; directCount++) {
                    float2 rand = float2(rng.getFloat(), rng.getFloat());
                    pathRadiance += estimateDirectMISLight(scene.tlas, shadingFrame, scene.envMap, bsdf, outgoingDirSs, surface.position, surface.triangleFrame.n, surface.spawnOffset, rand, envSamples, brdfSamples);
                }

                // accumulate direct light samples from emissive meshes
                for (uint directCount = 0; directCount < meshSamples; directCount++) {
                    float2 rand = float2(rng.getFloat(), rng.getFloat());
                    pathRadiance += estimateDirectMISLight(scene.tlas, shadingFrame, scene.meshLights, bsdf, outgoingDirSs, surface.position, surface.triangleFrame.n, surface.spawnOffset, rand, meshSamples, brdfSamples);
                }
            }

            for (uint brdfSampleCount = 0; brdfSampleCount < brdfSamples; brdfSampleCount++) {
                const BSDFSample sample = bsdf.sample(outgoingDirSs, float2(rng.getFloat(), rng.getFloat()));
                if (sample.eval.pdf > 0.0) {
                    const float3 throughput = sample.eval.reflectance * abs(Frame::cosTheta(sample.dirFs));
                    if (all(throughput != 0)) {
                        Ray ray = initialRay;
                        ray.direction = shadingFrame.frameToWorld(sample.dirFs);
                        ray.origin = surface.position + faceForward(surface.triangleFrame.n, ray.direction) * surface.spawnOffset;
                        Intersection its = Intersection::find(scene.tlas, ray.desc());
                        if (its.hit()) {
                            // hit -- collect light from emissive meshes
                            const SurfacePoint surface = scene.world.surfacePoint(its.instanceIndex, its.geometryIndex, its.primitiveIndex, its.barycentrics);
                            const float lightPdf = areaMeasureToSolidAngleMeasure(surface.position, ray.origin, ray.direction, surface.triangleFrame.n) * scene.meshLights.areaPdf(its.instanceIndex, its.geometryIndex, its.primitiveIndex);
                            const float weight = misWeight(brdfSamples, sample.eval.pdf, meshSamples, lightPdf);
                            pathRadiance += throughput * scene.world.material(its.instanceIndex, its.geometryIndex).getEmissive(surface.texcoord) * weight;
                        } else {
                            // miss -- collect light from env map
                            const LightEvaluation l = scene.envMap.evaluate(ray.direction);
                            const float weight = misWeight(brdfSamples, sample.eval.pdf, envSamples, l.pdf);
                            pathRadiance += throughput * l.radiance * weight;
                        }
                    }
                }
            }
        } else {
            // add background color
            pathRadiance += scene.envMap.evaluate(initialRay.direction).radiance;
        }

        return pathRadiance;
    }
};
