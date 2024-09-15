#pragma once

#include "world.hlsl"
#include "material.hlsl"
#include "../utils/reservoir.hlsl"

struct LightSample {
    float3 connection; // connection vector in world space from initial position to sampled position
    float3 radiance;
    float pdf;
};

struct LightEvaluation {
    float3 radiance;
    float pdf;
};

struct TriangleMetadata {
	uint instanceIndex;
	uint geometryIndex;
};

interface Light {
    // samples a light direction based on given position and geometric normal, returning
    // radiance at that point from light and pdf of this direction + radiance, ignoring visibility
    LightSample sample(float3 positionWs, float3 triangleNormalDirWs, float2 square);
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

    LightSample sample(float3 positionWs, float3 normalWs, float2 rand) {
        const uint size = textureDimensions(luminanceTexture).x;
        const uint mipCount = log2(size) + 1;

        uint2 idx = uint2(0, 0);
        for (uint level = mipCount; level-- > 0;) {
            Reservoir<uint2> r = Reservoir<uint2>::empty();
            for (uint i = 0; i < 2; i++) {
                for (uint j = 0; j < 2; j++) {
                    const uint2 coords = 2 * idx + uint2(i, j);
                    r.update(coords, luminanceTexture.Load(uint3(coords, level)), rand.x);
                }
            }
            idx = r.selected;
        }
        const float integral = luminanceTexture.Load(uint3(0, 0, mipCount - 1));

        const float discretePdf = luminanceTexture[idx] * float(size * size) / integral;
        const float2 uv = (float2(idx) + rand) / float2(size, size);

        const float envMapDistance = 10000000000;
        LightSample lightSample;
        lightSample.pdf = discretePdf / (4.0 * PI);
        lightSample.connection = squareToEqualAreaSphere(uv) * envMapDistance;
        lightSample.radiance = rgbTexture[idx] / lightSample.pdf;

        return lightSample;
    }

    // pdf is with respect to solid angle (no trace)
    LightEvaluation evaluate(float3 dirWs) {
        const uint size = textureDimensions(luminanceTexture).x;
        const uint mipCount = log2(size) + 1;
        const float integral = luminanceTexture.Load(uint3(0, 0, mipCount - 1));

        if (integral == 0) {
            LightEvaluation l;
            l.pdf = 0;
            l.radiance = 0;
            return l;
        }

        const float2 uv = squareToEqualAreaSphereInverse(dirWs);
        const uint2 idx = clamp(uint2(uv * size), uint2(0, 0), uint2(size, size));
        const float discretePdf = luminanceTexture[idx] * float(size * size) / integral;

        LightEvaluation l;
        l.pdf = discretePdf / (4.0 * PI);
        l.radiance = rgbTexture[idx];
        return l;
    }
};

float areaMeasureToSolidAngleMeasure(float3 pos1, float3 pos2, float3 dir1, float3 dir2) {
    const float r2 = dot(pos1 - pos2, pos1 - pos2);
    const float lightCos = abs(dot(-dir1, dir2));

    return r2 / lightCos;
}

struct TriangleLight: Light {
	uint instanceIndex;
	uint geometryIndex;
	uint primitiveIndex;
    World world;

    static TriangleLight create(uint instanceIndex, uint geometryIndex, uint primitiveIndex, World world) {
        TriangleLight light;
        light.instanceIndex = instanceIndex;
        light.geometryIndex = geometryIndex;
        light.primitiveIndex = primitiveIndex;
        light.world = world;
        return light;
    }

    LightSample sample(float3 positionWs, float3 triangleNormalDirWs, float2 rand) {
        const float2 barycentrics = squareToTriangle(rand);
        const SurfacePoint surface = world.surfacePoint(instanceIndex, geometryIndex, primitiveIndex, barycentrics);

        LightSample lightSample;
        lightSample.connection = surface.position - positionWs;
        lightSample.connection += faceForward(surface.triangleFrame.n, -lightSample.connection) * surface.spawnOffset;
        lightSample.pdf = areaMeasureToSolidAngleMeasure(surface.position, positionWs, normalize(lightSample.connection), surface.triangleFrame.n) / world.triangleArea(instanceIndex, geometryIndex, primitiveIndex);
        lightSample.radiance = world.material(instanceIndex, geometryIndex).getEmissive(surface.texcoord) / lightSample.pdf;

        return lightSample;
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

    LightSample sample(float3 positionWs, float3 triangleNormalDirWs, float2 rand) {
        LightSample lightSample;
        lightSample.pdf = 0.0;

        if (integral() == 0.0) return lightSample;

        const uint mipCount = uint(ceil(log2(emissiveTriangleCount))) + 1;

        uint idx = 0;
        for (uint level = mipCount; level-- > 0;) {
            Reservoir<uint> r = Reservoir<uint>::empty();
            for (uint i = 0; i < 2; i++) {
                const uint coord = 2 * idx + i;
                r.update(coord, power.Load(uint2(coord, level)), rand.x);
            }
            idx = r.selected;
        }
        const TriangleMetadata meta = metadata[idx];

        const uint instanceID = world.instances[meta.instanceIndex].instanceCustomIndex;
        const uint primitiveIndex = idx - geometryToTrianglePowerOffset[instanceID + meta.geometryIndex];

        const TriangleLight inner = TriangleLight::create(meta.instanceIndex, meta.geometryIndex, primitiveIndex, world);
        lightSample = inner.sample(positionWs, triangleNormalDirWs, rand);
        lightSample.pdf *= selectionPdf(meta.instanceIndex, meta.geometryIndex, primitiveIndex);
        lightSample.radiance /= selectionPdf(meta.instanceIndex, meta.geometryIndex, primitiveIndex);
        return lightSample;
    }

    float selectionPdf(uint instanceIndex, uint geometryIndex, uint primitiveIndex) {
        if (integral() == 0.0) return 0.0; // no lights
        const uint instanceID = world.instances[instanceIndex].instanceCustomIndex;
        const uint offset = geometryToTrianglePowerOffset[instanceID + geometryIndex];
        const uint invalidOffset = 0xFFFFFFFF;
        if (offset == invalidOffset) return 0.0; // no light at this triangle
        const uint idx = offset + primitiveIndex;
        return power.Load(uint2(idx, 0)) / integral();
    }

    float areaPdf(uint instanceIndex, uint geometryIndex, uint primitiveIndex) {
        const float triangleSelectionPdf = selectionPdf(instanceIndex, geometryIndex, primitiveIndex);
        const float triangleAreaPdf = 1.0 / world.triangleArea(instanceIndex, geometryIndex, primitiveIndex);
        return triangleSelectionPdf * triangleAreaPdf;
    }

    float integral() {
        const uint size = emissiveTriangleCount;

        if (size == 0) return 0;

        const uint mipCount = uint(ceil(log2(emissiveTriangleCount))) + 1;
        return power.Load(uint2(0, mipCount - 1));
    }
};
