#pragma once

#include "../utils/math.hlsl"
#include "../utils/random.hlsl"
#include "reflection_frame.hlsl"
#include "material.hlsl"
#include "world.hlsl"
#include "light.hlsl"
#include "ray.hlsl"
#include "spectrum.hlsl"

// with
//   power == 1 this becomes balance heuristic
//   power == 0 this becomes uniform weighting
float powerHeuristic2(const uint fCount, const float fPdf2, const uint gCount, const float gPdf) {
    return fPdf2 / (fCount * fPdf2 + gCount * pow2(gPdf));
}

float misWeight(const uint fCount, const float fPdf, const uint gCount, const float gPdf) {
    if (fPdf == 1.#INF) return 1.0 / fCount; // delta distribution for f, g not relevant
    return powerHeuristic2(fCount, pow2(fPdf), gCount, gPdf);
}

// estimates direct lighting from light + brdf via MIS
// only samples light
template <class Light, class BSDF>
float estimateDirectMISLight(RaytracingAccelerationStructure accel, Frame frame, Light light, BSDF material, float3 outgoingDirFs, float λ, float3 positionWs, float3 triangleNormalDirWs, float spawnOffset, float2 rand, uint lightSamplesTaken, uint brdfSamplesTaken) {
    const LightSample lightSample = light.sample(λ, positionWs, rand);

    if (lightSample.eval.radiance > 0) {
    const float3 lightDirWs = normalize(lightSample.connection);
        const BSDFEvaluation bsdfEval = material.evaluate(frame.worldToFrame(lightDirWs), outgoingDirFs);
        if (bsdfEval.reflectance > 0) {
            float3 dir = faceForward(triangleNormalDirWs, lightDirWs) * spawnOffset;
            if (!ShadowIntersection::hit(accel, positionWs + dir, lightSample.connection - dir)) {                
                return lightSample.eval.radiance * bsdfEval.reflectance * misWeight(lightSamplesTaken, lightSample.eval.pdf, brdfSamplesTaken, bsdfEval.pdf);
            }
        }
    }

    return 0;
}

// selects a shading normal based on the most preferred normal that is plausible
Frame selectFrame(const SurfacePoint surface, const Material material, const float3 outgoingDirWs) {
    const Frame textureFrame = material.getTextureFrame(surface.texcoord, surface.frame);
    Frame shadingFrame;
    int sign0 = sign(dot(surface.triangleFrame.n, outgoingDirWs));
    if (sign0 == sign(dot(outgoingDirWs, textureFrame.n))) {
        // prefer texture normal if we can
        shadingFrame = textureFrame;
    } else if (sign0 == sign(dot(outgoingDirWs, surface.frame.n))) {
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
    float throughput;
    float radiance;
    uint bounceCount;

    static Path create(const Ray ray) {
        Path p;
        p.ray = ray;
        p.throughput = 1;
        p.radiance = 0;
        p.bounceCount = 0;
        return p;
    }
};

interface Integrator {
    float incomingRadiance(const Scene scene, const Ray initialRay, const float λ, inout Rng rng);
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

    float incomingRadiance(const Scene scene, const Ray initialRay, const float λ, inout Rng rng) {
        Path path = Path::create(initialRay);

        for (Intersection its = Intersection::find(scene.tlas, path.ray.desc()); its.hit(); its = Intersection::find(scene.tlas, path.ray.desc())) {

            // decode mesh attributes and material from intersection
            SurfacePoint surface = scene.world.surfacePoint(its.instanceIndex, its.geometryIndex, its.primitiveIndex, its.barycentrics);
            const Material material = scene.world.material(its.instanceIndex, its.geometryIndex);

            // collect light from emissive meshes
            if(meshSamplesPerBounce > 0)
            {
                const float lightPdf = areaMeasureToSolidAngleMeasure(surface.position - path.ray.origin, path.ray.direction, surface.triangleFrame.n) * scene.meshLights.areaPdf(its.instanceIndex, its.geometryIndex, its.primitiveIndex);
                const float weight = misWeight(1, path.ray.pdf, meshSamplesPerBounce, lightPdf);
                path.radiance += path.throughput * material.getEmissive(λ, surface.texcoord) * weight;
            } else path.radiance += path.throughput * material.getEmissive(λ, surface.texcoord); 

            if(path.bounceCount > maxBounces) return path.radiance;

            // possibly terminate if reached max bounce cutoff or lose at russian roulette
            // max bounce cutoff needs to be before NEE below, and after light contribution above, otherwise MIS would need to be adjusted
            if(path.bounceCount > 3)
            {
                const float pSurvive = min(0.95, path.throughput);
                if (rng.getFloat() > pSurvive) return path.radiance;
                path.throughput /= pSurvive;
            }

            const PolymorphicBSDF bsdf = PolymorphicBSDF::load(material, surface.texcoord, λ);

            const float3 outgoingDirWs = -path.ray.direction;
            const Frame shadingFrame = selectFrame(surface, material, outgoingDirWs);
            const float3 outgoingDirSs = shadingFrame.worldToFrame(outgoingDirWs);

            //spawnOffset needed from here
            surface = scene.world.calcSpawnOffset(its.instanceIndex, surface);

            if (!bsdf.isDelta()) {
                // accumulate direct light samples from env map
                for (uint directCount = 0; directCount < envSamplesPerBounce; directCount++) {
                    float2 rand = rng.getFloat2();
                    path.radiance += path.throughput * estimateDirectMISLight(scene.tlas, shadingFrame, scene.envMap, bsdf, outgoingDirSs, λ, surface.position, surface.triangleFrame.n, surface.spawnOffset, rand, envSamplesPerBounce, 1);
                }

                // accumulate direct light samples from emissive meshes
                for (uint directCount = 0; directCount < meshSamplesPerBounce; directCount++) {
                    float2 rand = rng.getFloat2();
                    path.radiance += path.throughput * estimateDirectMISLight(scene.tlas, shadingFrame, scene.meshLights, bsdf, outgoingDirSs, λ, surface.position, surface.triangleFrame.n, surface.spawnOffset, rand, meshSamplesPerBounce, 1);
                }
            }

            // sample direction for next bounce
            const BSDFSample sample = bsdf.sample(outgoingDirSs, rng.getFloat2());
            if (sample.eval.reflectance < NEARzero) return path.radiance;

            // set up info for next bounce
            path.ray.direction = shadingFrame.frameToWorld(sample.dirFs);
            path.ray.origin = surface.position + faceForward(surface.triangleFrame.n, path.ray.direction) * surface.spawnOffset;
            path.ray.pdf = sample.eval.pdf;
            path.throughput *= sample.eval.reflectance;
            path.bounceCount += 1;
        }

        // we only get here on misses -- terminations for other reasons return from loop

        // handle env map
        if(envSamplesPerBounce > 0)
        {
            const LightEvaluation l = scene.envMap.evaluate(λ, path.ray.direction);
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

    float incomingRadiance(const Scene scene, const Ray initialRay, const float λ, inout Rng rng) {
        float pathRadiance = 0;

        Intersection its = Intersection::find(scene.tlas, initialRay.desc());
        if (its.hit()) {
            // decode mesh attributes and material from intersection
            const SurfacePoint surface = scene.world.surfacePoint(its.instanceIndex, its.geometryIndex, its.primitiveIndex, its.barycentrics);
            const Material material = scene.world.material(its.instanceIndex, its.geometryIndex);
            const PolymorphicBSDF bsdf = PolymorphicBSDF::load(material, surface.texcoord, λ);

            const float3 outgoingDirWs = -initialRay.direction;
            const Frame shadingFrame = selectFrame(surface, material, outgoingDirWs);
            const float3 outgoingDirSs = shadingFrame.worldToFrame(outgoingDirWs);

            // collect light from emissive meshes
            pathRadiance += material.getEmissive(λ, surface.texcoord);

            if (!bsdf.isDelta()) {
                // accumulate direct light samples from env map
                for (uint directCount = 0; directCount < envSamples; directCount++) {
                    float2 rand = rng.getFloat2();
                    pathRadiance += estimateDirectMISLight(scene.tlas, shadingFrame, scene.envMap, bsdf, outgoingDirSs, λ, surface.position, surface.triangleFrame.n, surface.spawnOffset, rand, envSamples, brdfSamples);
                }

                // accumulate direct light samples from emissive meshes
                for (uint directCount = 0; directCount < meshSamples; directCount++) {
                    float2 rand = rng.getFloat2();
                    pathRadiance += estimateDirectMISLight(scene.tlas, shadingFrame, scene.meshLights, bsdf, outgoingDirSs, λ, surface.position, surface.triangleFrame.n, surface.spawnOffset, rand, meshSamples, brdfSamples);
                }
            }

            for (uint brdfSampleCount = 0; brdfSampleCount < brdfSamples; brdfSampleCount++) {
                const BSDFSample sample = bsdf.sample(outgoingDirSs, rng.getFloat2());
                if (sample.eval.reflectance > 0) {
                    Ray ray = initialRay;
                    ray.direction = shadingFrame.frameToWorld(sample.dirFs);
                    ray.origin = surface.position + faceForward(surface.triangleFrame.n, ray.direction) * surface.spawnOffset;
                    Intersection its = Intersection::find(scene.tlas, ray.desc());
                    if (its.hit()) {
                        // hit -- collect light from emissive meshes
                        const SurfacePoint surface = scene.world.surfacePoint(its.instanceIndex, its.geometryIndex, its.primitiveIndex, its.barycentrics);
                        const float lightPdf = areaMeasureToSolidAngleMeasure(surface.position - ray.origin, ray.direction, surface.triangleFrame.n) * scene.meshLights.areaPdf(its.instanceIndex, its.geometryIndex, its.primitiveIndex);
                        const float weight = misWeight(brdfSamples, sample.eval.pdf, meshSamples, lightPdf);
                        pathRadiance += sample.eval.reflectance * scene.world.material(its.instanceIndex, its.geometryIndex).getEmissive(λ, surface.texcoord) * weight;
                    } else {
                        // miss -- collect light from env map
                        const LightEvaluation l = scene.envMap.evaluate(λ, ray.direction);
                        const float weight = misWeight(brdfSamples, sample.eval.pdf, envSamples, l.pdf);
                        pathRadiance += sample.eval.reflectance * l.radiance * weight;
                    }
                }
            }
        } else {
            // add background color
            pathRadiance += scene.envMap.evaluate(λ, initialRay.direction).radiance;
        }

        return pathRadiance;
    }
};
