#pragma once

#include "reflection_frame.hlsl"
#include "material.hlsl"

struct Instance { // same required by vulkan on host side
    row_major float3x4 transform;
    uint instanceCustomIndex : 24;
    uint mask : 8;
    uint instanceShaderBindingTableRecordOffset : 24;
    uint flags : 8;
    uint64_t accelerationStructureReference;
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
    float3 position0;   //position before transform
    float2 texcoord;

    Frame triangleFrame; // from triangle positions
    Frame frame; // from vertex attributes

    float spawnOffset; // minimum offset along normal that a ray will not intersect

    float3 positions[3];
    float3 normals[3];
    float2 texcoords[3];
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

void getTangentBitangent(float3 positions[3], float2 texcoords[3], out float3 tangent, out float3 bitangent) {
    float2 deltaT10 = texcoords[1] - texcoords[0];
    float2 deltaT20 = texcoords[2] - texcoords[0];

    float3 deltaP10 = positions[1] - positions[0];
    float3 deltaP20 = positions[2] - positions[0];

    float det = deltaT10.x * deltaT20.y - deltaT10.y * deltaT20.x;
    if (isZERO(det)) {
        coordinateSystem(normalize(cross(deltaP10, deltaP20)), tangent, bitangent);
    } else {
        det = 1.0 / det;
        tangent = normalize((deltaT20.y * deltaP10 - deltaT10.y * deltaP20) * det);
        bitangent = normalize((-deltaT20.x * deltaP10 + deltaT10.x * deltaP20) * det);
    }
}

#define mixBary(a, b, c, bary) ( (a) + ((b) - (a)) * (bary).y + ((c) - (a)) * (bary).z )

#define interpolate(bary, v) ( mixBary(((v)[0]), ((v)[1]), ((v)[2]), (bary)) )

struct TriangleLocalSpace {
    float3 positions[3];
    float3 normals[3];
    float2 texcoords[3];

    SurfacePoint surfacePoint(const float2 attribs, const float3x4 toWorld, const float3x4 toMesh) {
        SurfacePoint surface;

        const float3 barycentrics = float3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

        const float3 edge1 = positions[1] - positions[0];
        const float3 edge2 = positions[2] - positions[0];
        surface.position0 = positions[0] + ((attribs.x * edge1) + (attribs.y * edge2));

        surface.texcoord = interpolate(barycentrics, texcoords);

        getTangentBitangent(positions, texcoords, surface.triangleFrame.s, surface.triangleFrame.t);
        surface.triangleFrame.n = normalize(cross(edge1, edge2));
        surface.triangleFrame.reorthogonalize();

        surface.frame = surface.triangleFrame;
        surface.frame.n = normalize(interpolate(barycentrics, normals));
        surface.frame.reorthogonalize();

        const float3x3 m_toMesh = (float3x3) transpose(toMesh);

        // convert to world space
        {
            surface.position = mul(toWorld, float4(surface.position0, 1.0));

            surface.triangleFrame = surface.triangleFrame.inSpace(m_toMesh, normalize( mul(m_toMesh, surface.triangleFrame.n) ) );
            surface.frame = surface.frame.inSpace(m_toMesh);
        }

        return surface;
    }

    SurfacePoint calcSpawnOffset(const float3x4 toWorld, const float3x4 toMesh, SurfacePoint surface)
    {
        const float3 wldNormal = mul((float3x3) transpose(toMesh), surface.triangleFrame.n);

        const float3 edge1 = positions[1] - positions[0];
        const float3 edge2 = positions[2] - positions[0];

        // https://developer.nvidia.com/blog/solving-self-intersection-artifacts-in-directx-raytracing/
        {
            // nvidia magic constants
            const float c0 = 5.9604644775390625E-8f;
            const float c1 = 1.788139769587360206060111522674560546875E-7f;
            const float c2 = 1.19209317972490680404007434844970703125E-7f;

            const float3 extent3 = abs(edge1) + abs(edge2) + abs(edge1 - edge2);

            float3 objErr = c0 * abs(positions[0]) + c1 * max(max(extent3.x, extent3.y), extent3.z);
            //4*

            const float3 wldErr = c1 * mul(abs((float3x3)toWorld), abs(surface.position0)) + mul(c2, abs(transpose(toWorld)[3]));
            //9* 1* 3* -> 13*

            objErr += c2 * mul(abs(toMesh), float4(abs(surface.position), 1));
            //13*

            const float objOffset = dot(objErr, abs(surface.triangleFrame.n));
            const float wldOffset = dot(wldErr, abs(wldNormal));
            //6*

            surface.spawnOffset = (objOffset + wldOffset) / length(wldNormal);
            //1/

            //total 4 + 13 + 13 + 6 = 36* 1/
        }

        return surface;
    }

    float area(const float3x3 toWorld3x3) {
        float3 p0 = mul(toWorld3x3, positions[0]); //was added m[0][3] + m[1][3] + m[2][3] but with p1 - p0 and p2 - p0 - this addition is collapsed to 0
        float3 p1 = mul(toWorld3x3, positions[1]);
        float3 p2 = mul(toWorld3x3, positions[2]);

        return length(cross(p1 - p0, p2 - p0)) * 0.5f;
    }
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
        const uint instanceID = instances[instanceIndex].instanceCustomIndex;
        const Geometry geometry = geometries[NonUniformResourceIndex(instanceID + geometryIndex)];
        return meshes[NonUniformResourceIndex(geometry.meshIndex)];
    }

    Material material(uint instanceIndex, uint geometryIndex) {
        const uint instanceID = instances[instanceIndex].instanceCustomIndex;
        const Geometry geometry = geometries[NonUniformResourceIndex(instanceID + geometryIndex)];
        return materials[NonUniformResourceIndex(geometry.materialIndex)];
    }

    float triangleArea(uint instanceIndex, uint geometryIndex, uint primitiveIndex) {
        return triangleLocalSpacePos(instanceIndex, geometryIndex, primitiveIndex).area((float3x3) toWorld(instanceIndex));
    }

    TriangleLocalSpace triangleLocalSpace(uint instanceIndex, uint geometryIndex, uint primitiveIndex) {
        TriangleLocalSpace t;

        Mesh mesh = this.mesh(instanceIndex, geometryIndex);

        const uint primitiveIndex3 = primitiveIndex * 3;
        const uint3 ind = mesh.indexAddress != 0 ? vk::RawBufferLoad<uint3>(mesh.indexAddress + sizeof(uint3) * primitiveIndex) : uint3(primitiveIndex3, primitiveIndex3 + 1, primitiveIndex3 + 2);

        // positions always available
        t.positions[0] = loadPosition(mesh.positionAddress, ind.x);
        t.positions[1] = loadPosition(mesh.positionAddress, ind.y);
        t.positions[2] = loadPosition(mesh.positionAddress, ind.z);

        // texcoords optional
        if (mesh.texcoordAddress != 0) {
            t.texcoords[0] = loadTexcoord(mesh.texcoordAddress, ind.x);
            t.texcoords[1] = loadTexcoord(mesh.texcoordAddress, ind.y);
            t.texcoords[2] = loadTexcoord(mesh.texcoordAddress, ind.z);
        } else {
            // sane defaults for constant textures
            t.texcoords[0] = float2(0, 0);
            t.texcoords[1] = float2(1, 0);
            t.texcoords[2] = float2(1, 1);
        }

        // normals optional
        if (mesh.normalAddress != 0) {
            t.normals[0] = loadNormal(mesh.normalAddress, ind.x);
            t.normals[1] = loadNormal(mesh.normalAddress, ind.y);
            t.normals[2] = loadNormal(mesh.normalAddress, ind.z);
        } else {
            // use triangle normal
            const float3 normal = normalize(cross(t.positions[1] - t.positions[0], t.positions[2] - t.positions[0]));
            t.normals[0] = normal;
            t.normals[1] = normal;
            t.normals[2] = normal;
        }

        return t;
    }

    TriangleLocalSpace triangleLocalSpacePos(uint instanceIndex, uint geometryIndex, uint primitiveIndex) {
        TriangleLocalSpace t;

        Mesh mesh = this.mesh(instanceIndex, geometryIndex);
        const uint primitiveInd3 = primitiveIndex * 3;

        const uint3 ind = mesh.indexAddress != 0 ? vk::RawBufferLoad<uint3>(mesh.indexAddress + sizeof(uint3) * primitiveIndex) : uint3(primitiveInd3, primitiveInd3 + 1, primitiveInd3 + 2);

        // positions always available
        t.positions[0] = loadPosition(mesh.positionAddress, ind.x);
        t.positions[1] = loadPosition(mesh.positionAddress, ind.y);
        t.positions[2] = loadPosition(mesh.positionAddress, ind.z);

        return t;
    }

    float3x4 toWorld(uint instanceIndex) {
        return instances[NonUniformResourceIndex(instanceIndex)].transform;
    }

    float3x4 toMesh(uint instanceIndex) {
        return worldToInstance[NonUniformResourceIndex(instanceIndex)];
    }

    SurfacePoint surfacePoint(uint instanceIndex, uint geometryIndex, uint primitiveIndex, float2 attribs) {
        return triangleLocalSpace(instanceIndex, geometryIndex, primitiveIndex).surfacePoint(attribs, toWorld(instanceIndex), toMesh(instanceIndex));
    }

    SurfacePoint calcSpawnOffset(uint instanceIndex, SurfacePoint surface) {
        TriangleLocalSpace tls;
        tls.positions = surface.positions;
        tls.normals = surface.normals;
        tls.texcoords = surface.texcoords;
        return tls.calcSpawnOffset(toWorld(instanceIndex), toMesh(instanceIndex), surface);
    }
};
