#pragma once

#include "world.hlsl"
#include "material.hlsl"

struct LightSample {
    float3 dirWs;
    float lightDistance;
    float3 radiance;
    float pdf;
};

struct LightEval {
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
            idx *= 2;
            const float2 probs_x = float2(
                luminanceTexture.Load(uint3(idx + uint2(0, 0), level)) + luminanceTexture.Load(uint3(idx + uint2(0, 1), level)),
                luminanceTexture.Load(uint3(idx + uint2(1, 0), level)) + luminanceTexture.Load(uint3(idx + uint2(1, 1), level))
            );
            idx.x += coinFlipRemap(probs_x.y / (probs_x.x + probs_x.y), rand.x);
            const float2 probs_y = float2(
                luminanceTexture.Load(uint3(idx + uint2(0, 0), level)),
                luminanceTexture.Load(uint3(idx + uint2(0, 1), level))
            );
            idx.y += coinFlipRemap(probs_y.y / (probs_y.x + probs_y.y), rand.y);
        }
        const float integral = luminanceTexture.Load(uint3(0, 0, mipCount - 1));

        const float discretePdf = luminanceTexture[idx] * float(size * size) / integral;
        const float2 uv = (float2(idx) + rand) / float2(size, size);

        LightSample lightSample;
        lightSample.pdf = discretePdf / (4.0 * PI);
        lightSample.dirWs = squareToEqualAreaSphere(uv);
        lightSample.lightDistance = INFINITY;
        lightSample.radiance = rgbTexture[idx];

        return lightSample;
    }

    // pdf is with respect to solid angle (no trace)
    LightEval eval(float3 dirWs) {
        const uint size = textureDimensions(luminanceTexture).x;
        const uint mipCount = log2(size) + 1;
        const float integral = luminanceTexture.Load(uint3(0, 0, mipCount - 1));

        if (integral == 0) {
            LightEval l;
            l.pdf = 0;
            l.radiance = 0;
            return l;
        }

        const float2 uv = squareToEqualAreaSphereInverse(dirWs);
        const uint2 idx = clamp(uint2(uv * size), uint2(0, 0), uint2(size, size));
        const float discretePdf = luminanceTexture[idx] * float(size * size) / integral;

        LightEval l;
        l.pdf = discretePdf / (4.0 * PI);
        l.radiance = rgbTexture[idx];
        return l;
    }

    float3 incomingRadiance(float3 dirWs) {
        float2 uv = squareToEqualAreaSphereInverse(dirWs);
        return rgbTexture.SampleLevel(sampler, uv, 0);
    }
};

float areaMeasureToSolidAngleMeasure(float3 pos1, float3 pos2, float3 dir1, float3 dir2) {
    float r2 = dot(pos1 - pos2, pos1 - pos2);
    float lightCos = dot(-dir1, dir2);

    return lightCos > 0.0f ? r2 / lightCos : 0.0f;
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
        const MeshAttributes attrs = MeshAttributes::lookupAndInterpolate(world, instanceIndex, geometryIndex, primitiveIndex, barycentrics).inWorld(world, instanceIndex);

        LightSample lightSample;
        lightSample.radiance = world.material(instanceIndex, geometryIndex).getEmissive(attrs.texcoord);
        lightSample.dirWs = normalize(attrs.position - positionWs);
        lightSample.pdf = areaMeasureToSolidAngleMeasure(attrs.position, positionWs, lightSample.dirWs, attrs.triangleFrame.n) / MeshAttributes::triangleArea(world, instanceIndex, geometryIndex, primitiveIndex);

        // compute precise ray distance
        const float3 offsetLightPositionWs = offsetAlongNormal(attrs.position, attrs.triangleFrame.n);
        const float3 offsetShadingPositionWs = offsetAlongNormal(positionWs, faceForward(triangleNormalDirWs, lightSample.dirWs));
        const float tmax = distance(offsetLightPositionWs, offsetShadingPositionWs);
        lightSample.lightDistance = tmax;

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
            idx *= 2;
            const float2 probs = float2(
                power.Load(uint2(idx + 0, level)),
                power.Load(uint2(idx + 1, level))
            );
            idx += coinFlipRemap(probs.y / (probs.x + probs.y), rand.x);
        }
        const TriangleMetadata meta = metadata[idx];

        const uint instanceID = world.instances[meta.instanceIndex].instanceID();
        const uint primitiveIndex = idx - geometryToTrianglePowerOffset[instanceID + meta.geometryIndex];

        const TriangleLight inner = TriangleLight::create(meta.instanceIndex, meta.geometryIndex, primitiveIndex, world);
        lightSample = inner.sample(positionWs, triangleNormalDirWs, rand);
        lightSample.pdf *= selectionPdf(meta.instanceIndex, meta.geometryIndex, primitiveIndex);
        return lightSample;
    }

    float selectionPdf(uint instanceIndex, uint geometryIndex, uint primitiveIndex) {
        if (integral() == 0.0) return 0.0; // no lights
        const uint instanceID = world.instances[instanceIndex].instanceID();
        const uint offset = geometryToTrianglePowerOffset[instanceID + geometryIndex];
        const uint invalidOffset = 0xFFFFFFFF;
        if (offset == invalidOffset) return 0.0; // no light at this triangle
        const uint idx = offset + primitiveIndex;
        return power.Load(uint2(idx, 0)) / integral();
    }

    float areaPdf(uint instanceIndex, uint geometryIndex, uint primitiveIndex) {
        const float triangleSelectionPdf = selectionPdf(instanceIndex, geometryIndex, primitiveIndex);
        const float triangleAreaPdf = 1.0 / MeshAttributes::triangleArea(world, instanceIndex, geometryIndex, primitiveIndex);
        return triangleSelectionPdf * triangleAreaPdf;
    }

    float integral() {
        const uint size = emissiveTriangleCount;

        if (size == 0) return 0;

        const uint mipCount = uint(ceil(log2(emissiveTriangleCount))) + 1;
        return power.Load(uint2(0, mipCount - 1));
    }
};
