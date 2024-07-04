#include "intersection.hlsl"
#include "camera.hlsl"
#include "scene.hlsl"
#include "integrator.hlsl"

// I use the `d` prefix to indicate a descriptor variable
// because as a functional programmer impure functions scare me

// GEOMETRY
[[vk::binding(0, 0)]] RaytracingAccelerationStructure dTLAS;
[[vk::binding(1, 0)]] StructuredBuffer<Instance> dInstances;
[[vk::binding(2, 0)]] StructuredBuffer<row_major float3x4> dWorldToInstance;
[[vk::binding(3, 0)]] StructuredBuffer<Mesh> dMeshes;
[[vk::binding(4, 0)]] StructuredBuffer<Geometry> dGeometries;
[[vk::binding(5, 0)]] StructuredBuffer<MaterialVariantData> dMaterials;

// EMISSIVE TRIANGLES
[[vk::binding(6, 0)]] Texture1D<float> dTrianglePower;
[[vk::binding(7, 0)]] StructuredBuffer<TriangleMetadata> dTriangleMetadata;
[[vk::binding(8, 0)]] StructuredBuffer<uint> dGeometryToTrianglePowerOffset;
[[vk::binding(9, 0)]] StructuredBuffer<uint> dEmissiveTriangleCount;

// BACKGROUND
[[vk::combinedImageSampler]] [[vk::binding(10, 0)]] Texture2D<float3> dBackgroundRgbTexture;
[[vk::combinedImageSampler]] [[vk::binding(10, 0)]] SamplerState dBackgroundSampler;
[[vk::binding(11, 0)]] Texture2D<float> dBackgroundLuminanceTexture;

// OUTPUT
[[vk::binding(12, 0)]] RWTexture2D<float4> dOutputImage;

// SPECIALIZATION CONSTANTS
[[vk::constant_id(0)]] const bool dIndexedAttributes = true;    // whether non-position vertex attributes are indexed
[[vk::constant_id(1)]] const bool dTwoComponentNormalTexture = true;  // whether normal textures are two or three component vectors
[[vk::constant_id(2)]] const uint dSamplesPerRun = 1;
[[vk::constant_id(3)]] const uint dMaxBounces = 4;
[[vk::constant_id(4)]] const uint dEnvSamplesPerBounce = 1;  // how many times the environment map should be sampled per bounce for light
[[vk::constant_id(5)]] const uint dMeshSamplesPerBounce = 1; // how many times emissive meshes should be sampled per bounce for light

// PUSH CONSTANTS
struct PushConsts {
	Camera camera;
	uint sampleCount;
};
[[vk::push_constant]] PushConsts pushConsts;

[shader("raygeneration")]
void raygen() {
    const uint2 imageCoords = DispatchRaysIndex().xy;
    const uint2 imageSize = DispatchRaysDimensions().xy;

    const PathTracingIntegrator integrator = PathTracingIntegrator::create(dMaxBounces, dEnvSamplesPerBounce, dMeshSamplesPerBounce);

    World world;
    world.instances = dInstances;
    world.worldToInstance = dWorldToInstance;
    world.meshes = dMeshes;
    world.geometries = dGeometries;
    world.materials = dMaterials;
    world.indexedAttributes = dIndexedAttributes;
    world.twoComponentNormalTexture = dTwoComponentNormalTexture;

    Scene scene;
    scene.tlas = dTLAS;
    scene.world = world;
    scene.envMap = EnvMap::create(dBackgroundRgbTexture, dBackgroundSampler, dBackgroundLuminanceTexture);
    scene.meshLights = MeshLights::create(dTrianglePower, dTriangleMetadata, dGeometryToTrianglePowerOffset, dEmissiveTriangleCount[0], world);

    for (uint sampleCount = pushConsts.sampleCount; sampleCount < pushConsts.sampleCount + dSamplesPerRun; sampleCount++) {
        // create rng for this sample
        Rng rng = Rng::fromSeed(uint3(sampleCount, imageCoords.x, imageCoords.y));

        // set up initial ray
        const float2 jitter = float2(rng.getFloat(), rng.getFloat());
        const float2 imageUV = (imageCoords + jitter) / imageSize;
        const RayDesc initialRay = pushConsts.camera.generateRay(dOutputImage, imageUV, float2(rng.getFloat(), rng.getFloat()));

        // trace the ray
        const float3 newSample = integrator.incomingRadiance(scene, initialRay, rng);

        // accumulate
        const float3 priorSampleAverage = sampleCount == 0 ? 0 : dOutputImage[imageCoords].xyz;
        dOutputImage[imageCoords] = float4(accumulate(priorSampleAverage, newSample, sampleCount), 1);
    }
}

struct Attributes
{
    float2 barycentrics;
};

[shader("closesthit")]
void closesthit(inout Intersection its, in Attributes attribs) {
    its.instanceIndex = InstanceIndex();
    its.geometryIndex = GeometryIndex();
    its.primitiveIndex = PrimitiveIndex();
    its.barycentrics = attribs.barycentrics;
}

[shader("miss")]
void miss(inout Intersection its) {
    its = Intersection::createMiss();
}

[shader("miss")]
void shadowmiss(inout ShadowIntersection its) {
    its.inShadow = false;
}

