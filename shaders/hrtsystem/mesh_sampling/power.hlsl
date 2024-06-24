#include "../../utils/helpers.hlsl"
#include "../scene.hlsl"

// world info
[[vk::constant_id(0)]] const bool indexed_attributes = true;
[[vk::constant_id(1)]] const bool two_component_normal_texture = true;

[[vk::binding(0, 0)]] StructuredBuffer<Instance> dInstances;
[[vk::binding(1, 0)]] StructuredBuffer<row_major float3x4> dWorldToInstance;
[[vk::binding(2, 0)]] StructuredBuffer<Mesh> dMeshes;
[[vk::binding(3, 0)]] StructuredBuffer<Geometry> dGeometries;
[[vk::binding(4, 0)]] StructuredBuffer<MaterialVariantData> dMaterials;
[[vk::binding(5, 0)]] StructuredBuffer<uint> emissiveTriangleCount;

// dst
[[vk::binding(6, 0)]] RWTexture1D<float> dstPower;
[[vk::binding(7, 0)]] RWStructuredBuffer<TriangleMetadata> dstTriangleMetadata;

// mesh info
struct PushConsts {
	uint instanceIndex;
	uint geometryIndex;
	uint triangleCount;
};
[[vk::push_constant]] PushConsts pushConsts;

[numthreads(32, 1, 1)]
void main(uint3 dispatchXYZ: SV_DispatchThreadID) {
	const uint srcIndex = dispatchXYZ.x;

	if (any(srcIndex >= pushConsts.triangleCount)) return;

	World world;
    world.instances = dInstances;
    world.worldToInstance = dWorldToInstance;
    world.meshes = dMeshes;
    world.geometries = dGeometries;
    world.materials = dMaterials;
    world.indexed_attributes = indexed_attributes;
    world.two_component_normal_texture = two_component_normal_texture;
	
	float total_emissive = 0;

	const uint samples_per_dim = 8;
	for (uint i = 0; i < samples_per_dim; i++) {
		for (uint j = 0; j < samples_per_dim; j++) {
			const float2 barycentrics = squareToTriangle(float2(i, j) / float(samples_per_dim));
			const MeshAttributes attrs = MeshAttributes::lookupAndInterpolate(world, pushConsts.instanceIndex, pushConsts.geometryIndex, srcIndex, barycentrics);
			const uint instanceID = world.instances[pushConsts.instanceIndex].instanceID();
			total_emissive += luminance(getEmissive(world, world.materialIdx(instanceID, pushConsts.geometryIndex), attrs.texcoord));
		}
	}

	const float average_emissive = total_emissive / float(samples_per_dim * samples_per_dim);

	const float power = PI * MeshAttributes::triangleArea(world, pushConsts.instanceIndex, pushConsts.geometryIndex, srcIndex) * average_emissive;

	const uint dstOffset = emissiveTriangleCount[0];
	dstPower[dstOffset + srcIndex] = power;
	dstTriangleMetadata[dstOffset + srcIndex].instanceIndex = pushConsts.instanceIndex;
	dstTriangleMetadata[dstOffset + srcIndex].geometryIndex = pushConsts.geometryIndex;
}