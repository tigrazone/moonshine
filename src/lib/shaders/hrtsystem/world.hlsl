#pragma once

#include "reflection_frame.hlsl"
#include "material.hlsl"

struct Instance { // same required by vulkan on host side
    row_major float3x4 transform;
    uint instanceCustomIndexAndMask;
    uint instanceShaderBindingTableRecordOffsetAndFlags;
    uint64_t accelerationStructureReference;

    uint instanceID() {
        return instanceCustomIndexAndMask & 0x00FFFFFF;
    }
};

struct Geometry {
    uint meshIndex;
    uint materialIndex;
};

struct Mesh {
    uint64_t positionAddress;
    uint64_t texcoordAddress; // may be zero, for no texcoords
    uint64_t normalAddress; // may be zero, for no vertex normals

    uint64_t indexAddress; // may be zero, for unindexed geometry
};

struct SurfacePoint {
    float3 position;
    float2 texcoord;

    Frame triangleFrame; // from triangle positions
    Frame frame; // from vertex attributes

    float spawnOffset; // minimum offset along normal that a ray will not intersect
};

float3 loadPosition(uint64_t addr, uint index) {
    return vk::RawBufferLoad<float3>(addr + sizeof(float3) * index);
}

float2 loadTexcoord(uint64_t addr, uint index) {
    return vk::RawBufferLoad<float2>(addr + sizeof(float2) * index);
}

float3 loadNormal(uint64_t addr, uint index) {
    return vk::RawBufferLoad<float3>(addr + sizeof(float3) * index);
}

void getTangentBitangent(float3 p0, float3 p1, float3 p2, float2 t0, float2 t1, float2 t2, out float3 tangent, out float3 bitangent) {
    float2 deltaT10 = t1 - t0;
    float2 deltaT20 = t2 - t0;

    float3 deltaP10 = p1 - p0;
    float3 deltaP20 = p2 - p0;

    float det = deltaT10.x * deltaT20.y - deltaT10.y * deltaT20.x;
    if (det == 0.0) {
        coordinateSystem(normalize(cross(deltaP10, deltaP20)), tangent, bitangent);
    } else {
        tangent = normalize((deltaT20.y * deltaP10 - deltaT10.y * deltaP20) / det);
        bitangent = normalize((-deltaT20.x * deltaP10 + deltaT10.x * deltaP20) / det);
    }
}

template <typename T>
T interpolate(float3 barycentrics, T v1, T v2, T v3) {
    return barycentrics.x * v1 + barycentrics.y * v2 + barycentrics.z * v3;
}

struct World {
    StructuredBuffer<Instance> instances;
    StructuredBuffer<row_major float3x4> worldToInstance;

    StructuredBuffer<Mesh> meshes;
    StructuredBuffer<Geometry> geometries;

    StructuredBuffer<Material> materials;

    // TODO: there's a lot of indirection in these two functions just to load some data
    // probably can reorganize this for there to be some more direct path 
    Mesh mesh(uint instanceIndex, uint geometryIndex) {
        const uint instanceID = instances[instanceIndex].instanceID();
        const Geometry geometry = geometries[NonUniformResourceIndex(instanceID + geometryIndex)];
        return meshes[NonUniformResourceIndex(geometry.meshIndex)];
    }

    Material material(uint instanceIndex, uint geometryIndex) {
        const uint instanceID = instances[instanceIndex].instanceID();
        const Geometry geometry = geometries[NonUniformResourceIndex(instanceID + geometryIndex)];
        return materials[NonUniformResourceIndex(geometry.materialIndex)];
    }

    float triangleArea(uint instanceIndex, uint geometryIndex, uint primitiveIndex) {
        Mesh mesh = this.mesh(instanceIndex, geometryIndex);

        const uint3 ind = mesh.indexAddress != 0 ? vk::RawBufferLoad<uint3>(mesh.indexAddress + sizeof(uint3) * primitiveIndex) : float3(primitiveIndex * 3 + 0, primitiveIndex * 3 + 1, primitiveIndex * 3 + 2);

        float3x4 toWorld = instances[NonUniformResourceIndex(instanceIndex)].transform;
        float3 p0 = mul(toWorld, float4(loadPosition(mesh.positionAddress, ind.x), 1.0));
        float3 p1 = mul(toWorld, float4(loadPosition(mesh.positionAddress, ind.y), 1.0));
        float3 p2 = mul(toWorld, float4(loadPosition(mesh.positionAddress, ind.z), 1.0));

        return length(cross(p1 - p0, p2 - p0)) / 2.0;
    }

    SurfacePoint surfacePoint(uint instanceIndex, uint geometryIndex, uint primitiveIndex, float2 attribs) {
        SurfacePoint surface;


        Mesh mesh = this.mesh(instanceIndex, geometryIndex);

        const uint3 ind = mesh.indexAddress != 0 ? vk::RawBufferLoad<uint3>(mesh.indexAddress + sizeof(uint3) * primitiveIndex) : float3(primitiveIndex * 3 + 0, primitiveIndex * 3 + 1, primitiveIndex * 3 + 2);
        const float3 barycentrics = float3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

        // positions always available
        const float3 p0 = loadPosition(mesh.positionAddress, ind.x);
        const float3 p1 = loadPosition(mesh.positionAddress, ind.y);
        const float3 p2 = loadPosition(mesh.positionAddress, ind.z);
        const float3 edge1 = p1 - p0;
        const float3 edge2 = p2 - p0;
        surface.position = p0 + ((attribs.x * edge1) + (attribs.y * edge2));

        {

            // texcoords optional
            float2 t0, t1, t2;
            if (mesh.texcoordAddress != 0) {
                t0 = loadTexcoord(mesh.texcoordAddress, ind.x);
                t1 = loadTexcoord(mesh.texcoordAddress, ind.y);
                t2 = loadTexcoord(mesh.texcoordAddress, ind.z);
            } else {
                // textures should be constant in this case
                t0 = float2(0, 0);
                t1 = float2(1, 0);
                t2 = float2(1, 1);
            }
            surface.texcoord = interpolate(barycentrics, t0, t1, t2);

            getTangentBitangent(p0, p1, p2, t0, t1, t2, surface.triangleFrame.s, surface.triangleFrame.t);
            surface.triangleFrame.n = normalize(cross(edge1, edge2));
            surface.triangleFrame.reorthogonalize();

            // normals optional
            if (mesh.normalAddress != 0) {
                float3 n0 = loadNormal(mesh.normalAddress, ind.x);
                float3 n1 = loadNormal(mesh.normalAddress, ind.y);
                float3 n2 = loadNormal(mesh.normalAddress, ind.z);
                surface.frame = surface.triangleFrame;
                surface.frame.n = normalize(interpolate(barycentrics, n0, n1, n2));
                surface.frame.reorthogonalize();
            } else {
                // just use one from triangle
                surface.frame = surface.triangleFrame;
            }
        }

        const float3x4 toWorld = instances[NonUniformResourceIndex(instanceIndex)].transform;
        const float3x4 toMesh = worldToInstance[NonUniformResourceIndex(instanceIndex)];
        const float3 worldPosition = mul(toWorld, float4(surface.position, 1.0));

        // https://developer.nvidia.com/blog/solving-self-intersection-artifacts-in-directx-raytracing/
        {
            float3 wldNormal = mul(transpose((float3x3)toMesh), surface.triangleFrame.n);

            const float wldScale = rsqrt(dot(wldNormal, wldNormal));
            wldNormal = mul(wldScale, wldNormal);

            // nvidia magic constants
            const float c0 = 5.9604644775390625E-8f;
            const float c1 = 1.788139769587360206060111522674560546875E-7f;
            const float c2 = 1.19209317972490680404007434844970703125E-7f;

            const float3 extent3 = abs(edge1) + abs(edge2) + abs(edge1 - edge2);
            const float extent = max(max(extent3.x, extent3.y), extent3.z);

            float3 objErr = c0 * abs(p0) + mul(c1, extent);

            const float3 wldErr = c1 * mul(abs((float3x3)toWorld), abs(surface.position)) + mul(c2, abs(transpose(toWorld)[3]));

            objErr += c2 * mul(abs(toMesh), float4(abs(worldPosition), 1));

            const float wldOffset = dot(wldErr, abs(wldNormal));
            const float objOffset = dot(objErr, abs(surface.triangleFrame.n));

            surface.spawnOffset = wldScale * objOffset + wldOffset;
        }

        // convert to world space
        {
            surface.position = worldPosition;

            surface.triangleFrame = surface.triangleFrame.inSpace(transpose(toMesh));
            surface.frame = surface.frame.inSpace(transpose(toMesh));
        }

        return surface;
    }
};
