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
[[vk::binding(9, 0)]] StructuredBuffer<uint> emissiveTriangleCount;

// BACKGROUND
[[vk::combinedImageSampler]] [[vk::binding(10, 0)]] Texture2D<float3> dBackgroundRgbTexture;
[[vk::combinedImageSampler]] [[vk::binding(10, 0)]] SamplerState dBackgroundSampler;
[[vk::binding(11, 0)]] Texture2D<float> dBackgroundLuminanceTexture;

// OUTPUT
[[vk::binding(12, 0)]] RWTexture2D<float4> dOutputImage;

// PUSH CONSTANTS
struct PushConsts {
	Camera camera;
	uint sampleCount;
};
[[vk::push_constant]] PushConsts pushConsts;

// SPECIALIZATION CONSTANTS
[[vk::constant_id(0)]] const uint samples_per_run = 1;
[[vk::constant_id(1)]] const uint max_bounces = 4;
[[vk::constant_id(2)]] const uint env_samples_per_bounce = 1;   // how many times the environment map should be sampled per bounce for light
[[vk::constant_id(3)]] const uint mesh_samples_per_bounce = 1;  // how many times emissive meshes should be sampled per bounce for light
[[vk::constant_id(4)]] const bool flip_image = true;
[[vk::constant_id(5)]] const bool indexed_attributes = true;    // whether non-position vertex attributes are indexed
[[vk::constant_id(6)]] const bool two_component_normal_texture = true;  // whether normal textures are two or three component vectors

// https://www.nu42.com/2015/03/how-you-average-numbers.html
void accumulateColor(float3 sampledColor, uint sampleCount) {
    uint2 imageCoords = DispatchRaysIndex().xy;
    if (sampleCount == 0) {
        dOutputImage[imageCoords] = float4(sampledColor, 1.0);
    } else {
        float3 priorSampleAverage = dOutputImage[imageCoords].rgb;
        dOutputImage[imageCoords] += float4((sampledColor - priorSampleAverage) / (sampleCount + 1), 1.0);
    }
}

// returns uv of dispatch in [0..1]x[0..1], with slight variation based on rand
float2 dispatchUV(float2 rand) {
    float2 randomCenter = float2(0.5, 0.5) + 0.5 * squareToGaussian(rand);
    float2 uv = (float2(DispatchRaysIndex().xy) + randomCenter) / float2(DispatchRaysDimensions().xy);
    if (flip_image) uv.y = 1.0f - uv.y;
    return uv;
}

[shader("raygeneration")]
void raygen() {
    PathTracingIntegrator integrator = PathTracingIntegrator::create(max_bounces, env_samples_per_bounce, mesh_samples_per_bounce);

    World world;
    world.instances = dInstances;
    world.worldToInstance = dWorldToInstance;
    world.meshes = dMeshes;
    world.geometries = dGeometries;
    world.materials = dMaterials;
    world.indexed_attributes = indexed_attributes;
    world.two_component_normal_texture = two_component_normal_texture;

    Scene scene;
    scene.tlas = dTLAS;
    scene.world = world;
    scene.envMap = EnvMap::create(dBackgroundRgbTexture, dBackgroundSampler, dBackgroundLuminanceTexture);
    scene.meshLights = MeshLights::create(dTrianglePower, dTriangleMetadata, dGeometryToTrianglePowerOffset, emissiveTriangleCount[0], world);

    for (uint sampleCount = 0; sampleCount < samples_per_run; sampleCount++) {
        // create rng for this sample
        Rng rng = Rng::fromSeed(uint3(pushConsts.sampleCount + sampleCount, DispatchRaysIndex().x, DispatchRaysIndex().y));

        // set up initial directions for first bounce
        RayDesc initialRay = pushConsts.camera.generateRay(dOutputImage, dispatchUV(float2(rng.getFloat(), rng.getFloat())), float2(rng.getFloat(), rng.getFloat()));

        // trace the ray
        float3 color = integrator.incomingRadiance(scene, initialRay, rng);
        accumulateColor(color, pushConsts.sampleCount + sampleCount);
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

