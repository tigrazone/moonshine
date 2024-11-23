#pragma once

#include "world.hlsl"
#include "material.hlsl"
#include "spectrum.hlsl"
#include "../utils/reservoir.hlsl"

struct LightEvaluation {
    float radiance;
    float pdf;

    static LightEvaluation empty() {
        LightEvaluation eval;
        eval.radiance = 0;
        eval.pdf = 0;
        return eval;
    }
};

struct LightSample {
    float3 connection; // connection vector in world space from initial position to sampled position
    LightEvaluation eval;
};

struct TriangleMetadata {
	uint instanceIndex;
	uint geometryIndex;
	float area_;
};

interface Light {
    // samples a light direction based on given position, returns
    // radiance at that point from light and pdf of this direction + radiance, ignoring visibility
    LightSample sample(float λ, float3 positionWs, float2 square);
    LightSample sample(float λ, float3 positionWs, float2 square, float area_);
};

struct EnvMap : Light {
    Texture2D<float3> rgbTexture;
    SamplerState sampler;
    Texture2D<float> luminanceTexture;

    static EnvMap create(Texture2D<float3> rgbTexture, SamplerState sampler, Texture2D<float> luminanceTexture) {
        EnvMap map;
        map.rgbTexture = rgbTexture;
        map.sampler = sampler;
        map.luminanceTexture = luminanceTexture;
        return map;
    }

    LightSample sample(float λ, float3 positionWs, float2 rand) {
        const uint size = textureDimensions(luminanceTexture).x;
        const uint mipCount = log2(size) + 1;
        float reservour_rand = rand.x;

        uint2 idx = uint2(0, 0);
        uint2 idx2 = idx;
        for (uint level = mipCount; level-- > 0;) {
            Reservoir<uint2> r = Reservoir<uint2>::empty();
            for (uint i = 0; i < 2; i++) {
                for (uint j = 0; j < 2; j++) {
                    const uint2 coords = idx2 + uint2(i, j);
                    r.update(coords, luminanceTexture.Load(uint3(coords, level)), reservour_rand);
                }
            }
            idx = r.selected;
            idx2 = idx + idx;
        }
        const float integral = luminanceTexture.Load(uint3(0, 0, mipCount - 1));

        const float discretePdf = luminanceTexture[idx] * float(size * size) / integral;
        const float2 uv = (float2(idx) + rand) / float2(size, size);

        const float envMapDistance = 10000000000;
        LightSample lightSample;
        lightSample.connection = squareToEqualAreaSphere(uv) * envMapDistance;
        lightSample.eval.pdf = discretePdf * M_4PI;
        lightSample.eval.radiance = Spectrum::sampleEmission(λ, rgbTexture[idx]) / lightSample.eval.pdf;

        return lightSample;
    }

    LightSample sample(float λ, float3 positionWs, float2 rand, float area_) {
        return sample(λ, positionWs, rand);
    }

    // pdf is with respect to solid angle (no trace)
    LightEvaluation evaluate(float λ, float3 dirWs) {
        const uint size = textureDimensions(luminanceTexture).x;
        const uint mipCount = log2(size) + 1;
        const float integral = luminanceTexture.Load(uint3(0, 0, mipCount - 1));

        if (integral < NEARzero) return LightEvaluation::empty();

        const float2 uv = squareToEqualAreaSphereInverse(dirWs);
        const uint2 idx = clamp(uint2(uv * size), uint2(0, 0), uint2(size, size));
        const float discretePdf = luminanceTexture[idx] * float(size * size) / integral;

        LightEvaluation eval;
        eval.pdf = discretePdf * M_4PI;
        eval.radiance = Spectrum::sampleEmission(λ, rgbTexture[idx]);
        return eval;
    }
};

float areaMeasureToSolidAngleMeasure(float3 pos1, float3 pos2, float3 dir1, float3 dir2) {
    return dot(pos1 - pos2, pos1 - pos2) / abs(dot(dir1, dir2));
}

struct TriangleLight: Light {
    TriangleLocalSpace t;
    float3x4 toWorld;
    float3x4 toMesh;
    Material material;

    static TriangleLight create(uint instanceIndex, uint geometryIndex, uint primitiveIndex, World world) {
        TriangleLight light;
        light.t = world.triangleLocalSpace(instanceIndex, geometryIndex, primitiveIndex);
        light.toWorld = world.toWorld(instanceIndex);
        light.toMesh = world.toMesh(instanceIndex);
        light.material = world.material(instanceIndex, geometryIndex);
        return light;
    }

    LightSample sample(float λ, float3 positionWs, float2 rand, float area_) {
        const float2 barycentrics = squareToTriangle(rand);
        const SurfacePoint surface = t.surfacePoint(barycentrics, toWorld, toMesh);

        LightSample lightSample;
        lightSample.connection = surface.position - positionWs;
        lightSample.connection += faceForward(surface.triangleFrame.n, -lightSample.connection) * surface.spawnOffset;
        lightSample.eval.pdf = areaMeasureToSolidAngleMeasure(surface.position, positionWs, normalize(lightSample.connection), surface.triangleFrame.n) * area_;
        lightSample.eval.radiance = material.getEmissive(λ, surface.texcoord) / lightSample.eval.pdf;

        return lightSample;
    }

    LightSample sample(float λ, float3 positionWs, float2 rand) {
        float area_;
        return sample(λ, positionWs, rand, area_);
    }
};

// all mesh lights in scene
struct MeshLights : Light {
    Texture1D<float> power;
    StructuredBuffer<TriangleMetadata> metadata;
    StructuredBuffer<uint> geometryToTrianglePowerOffset;
    uint emissiveTriangleCount;
    World world;

    static MeshLights create(Texture1D<float> power, StructuredBuffer<TriangleMetadata> metadata, StructuredBuffer<uint> geometryToTrianglePowerOffset, uint emissiveTriangleCount, World world) {
        MeshLights lights;
        lights.power = power;
        lights.metadata = metadata;
        lights.geometryToTrianglePowerOffset = geometryToTrianglePowerOffset;
        lights.emissiveTriangleCount = emissiveTriangleCount;
        lights.world = world;
        return lights;
    }

    LightSample sample(float λ, float3 positionWs, float2 rand) {
        LightSample lightSample;
        lightSample.eval = LightEvaluation::empty();

        if (integral() < NEARzero) return lightSample;

        const uint mipCount = uint(ceil(log2(emissiveTriangleCount))) + 1;
        float reservour_rand = rand.x;

        uint idx = 0;
        for (uint level = mipCount; level-- > 0;) {
            Reservoir<uint> r = Reservoir<uint>::empty();
            for (uint i = 0; i < 2; i++) {
                const uint coord = idx + idx + i;
                r.update(coord, power.Load(uint2(coord, level)), reservour_rand);
            }
            idx = r.selected;
        }
        const TriangleMetadata meta = metadata[idx];

        const uint instanceID = world.instances[meta.instanceIndex].instanceCustomIndex;
        const uint primitiveIndex = idx - geometryToTrianglePowerOffset[instanceID + meta.geometryIndex];

        const TriangleLight inner = TriangleLight::create(meta.instanceIndex, meta.geometryIndex, primitiveIndex, world);
        lightSample = inner.sample(λ, positionWs, rand, meta.area_);
        float selPdf = selectionPdf(meta.instanceIndex, meta.geometryIndex, primitiveIndex);
        lightSample.eval.pdf *= selPdf;
        lightSample.eval.radiance /= selPdf;
        return lightSample;
    }

    LightSample sample(float λ, float3 positionWs, float2 rand, float area_) {
        return sample(λ, positionWs, rand);
    }

    float selectionPdf(uint instanceIndex, uint geometryIndex, uint primitiveIndex) {
        float integral_ = integral();
        if (integral_ < NEARzero) return 0.0; // no lights
        const uint instanceID = world.instances[instanceIndex].instanceCustomIndex;
        const uint offset = geometryToTrianglePowerOffset[instanceID + geometryIndex];
        if (offset == MAX_UINT) return 0.0; // no light at this triangle
        const uint idx = offset + primitiveIndex;
        return power.Load(uint2(idx, 0)) / integral_;
    }

    float areaPdf(uint instanceIndex, uint geometryIndex, uint primitiveIndex) {
        float integral_ = integral();
        if (integral_ < NEARzero) return 0.0; // no lights
        const uint instanceID = world.instances[instanceIndex].instanceCustomIndex;
        const uint offset = geometryToTrianglePowerOffset[instanceID + geometryIndex];
        if (offset == MAX_UINT) return 0.0; // no light at this triangle
        const uint idx = offset + primitiveIndex;
        return power.Load(uint2(idx, 0)) * metadata[idx].area_ / integral_;
    }

    float integral() {
        if (emissiveTriangleCount == 0) return 0;

        const uint mipCount = uint(ceil(log2(emissiveTriangleCount))) + 1;
        return power.Load(uint2(0, mipCount - 1));
    }
};
