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

struct MeshAttributes {
    float3 position;
    float2 texcoord;

    Frame triangleFrame; // from triangle positions
    Frame frame; // from vertex attributes

    static MeshAttributes lookupAndInterpolate(World world, uint instanceIndex, uint geometryIndex, uint primitiveIndex, float2 attribs) {
        MeshAttributes attrs;

        // construct attributes in object space
        {
            Mesh mesh = world.mesh(instanceIndex, geometryIndex);

            const uint3 ind = mesh.indexAddress != 0 ? vk::RawBufferLoad<uint3>(mesh.indexAddress + sizeof(uint3) * primitiveIndex) : float3(primitiveIndex * 3 + 0, primitiveIndex * 3 + 1, primitiveIndex * 3 + 2);
            const float3 barycentrics = float3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

            // positions always available
            float3 p0 = loadPosition(mesh.positionAddress, ind.x);
            float3 p1 = loadPosition(mesh.positionAddress, ind.y);
            float3 p2 = loadPosition(mesh.positionAddress, ind.z);
            attrs.position = interpolate(barycentrics, p0, p1, p2);

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
            attrs.texcoord = interpolate(barycentrics, t0, t1, t2);

            getTangentBitangent(p0, p1, p2, t0, t1, t2, attrs.triangleFrame.s, attrs.triangleFrame.t);
            attrs.triangleFrame.n = normalize(cross(p1 - p0, p2 - p0));
            attrs.triangleFrame.reorthogonalize();

            // normals optional
            if (mesh.normalAddress != 0) {
                float3 n0 = loadNormal(mesh.normalAddress, ind.x);
                float3 n1 = loadNormal(mesh.normalAddress, ind.y);
                float3 n2 = loadNormal(mesh.normalAddress, ind.z);
                attrs.frame = attrs.triangleFrame;
                attrs.frame.n = normalize(interpolate(barycentrics, n0, n1, n2));
                attrs.frame.reorthogonalize();
            } else {
                // just use one from triangle
                attrs.frame = attrs.triangleFrame;
            }
        }

        // convert to world space
        {
            const float3x4 toWorld = world.instances[NonUniformResourceIndex(instanceIndex)].transform;
            const float3x4 toMesh = world.worldToInstance[NonUniformResourceIndex(instanceIndex)];

            attrs.position = mul(toWorld, float4(attrs.position, 1.0));

            attrs.triangleFrame = attrs.triangleFrame.inSpace(transpose(toMesh));
            attrs.frame = attrs.frame.inSpace(transpose(toMesh));
        }

        return attrs;
    }

    static float triangleArea(World world, uint instanceIndex, uint geometryIndex, uint primitiveIndex) {
        Mesh mesh = world.mesh(instanceIndex, geometryIndex);

        const uint3 ind = mesh.indexAddress != 0 ? vk::RawBufferLoad<uint3>(mesh.indexAddress + sizeof(uint3) * primitiveIndex) : float3(primitiveIndex * 3 + 0, primitiveIndex * 3 + 1, primitiveIndex * 3 + 2);

        float3x4 toWorld = world.instances[NonUniformResourceIndex(instanceIndex)].transform;
        float3 p0 = mul(toWorld, float4(loadPosition(mesh.positionAddress, ind.x), 1.0));
        float3 p1 = mul(toWorld, float4(loadPosition(mesh.positionAddress, ind.y), 1.0));
        float3 p2 = mul(toWorld, float4(loadPosition(mesh.positionAddress, ind.z), 1.0));

        return length(cross(p1 - p0, p2 - p0)) / 2.0;
    }
};
