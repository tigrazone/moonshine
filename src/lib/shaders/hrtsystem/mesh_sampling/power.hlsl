#include "../../utils/helpers.hlsl"
#include "../world.hlsl"
#include "../light.hlsl"

// world info
[[vk::binding(0, 0)]] StructuredBuffer<Instance> dInstances;
[[vk::binding(1, 0)]] StructuredBuffer<row_major float3x4> dWorldToInstance;
[[vk::binding(2, 0)]] StructuredBuffer<Mesh> dMeshes;
[[vk::binding(3, 0)]] StructuredBuffer<Geometry> dGeometries;
[[vk::binding(4, 0)]] StructuredBuffer<Material> dMaterials;
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
	const uint srcPrimitive = dispatchXYZ.x;

	if (any(srcPrimitive >= pushConsts.triangleCount)) return;

	World world;
    world.instances = dInstances;
    world.worldToInstance = dWorldToInstance;
    world.meshes = dMeshes;
    world.geometries = dGeometries;
    world.materials = dMaterials;

	float total_emissive = 0;

	const uint samples_per_dim = 8;
	for (uint i = 0; i < samples_per_dim; i++) {
		for (uint j = 0; j < samples_per_dim; j++) {
			const float2 barycentrics = squareToTriangle(float2(i, j) / float(samples_per_dim));
			const MeshAttributes attrs = world.meshAttributes(pushConsts.instanceIndex, pushConsts.geometryIndex, srcPrimitive, barycentrics);
        	total_emissive += luminance(world.material(pushConsts.instanceIndex, pushConsts.geometryIndex).getEmissive(attrs.texcoord));
		}
	}

	const float average_emissive = total_emissive / float(samples_per_dim * samples_per_dim);

	const float power = PI * world.triangleArea(pushConsts.instanceIndex, pushConsts.geometryIndex, srcPrimitive) * average_emissive;

	const uint dstOffset = emissiveTriangleCount[0];
	dstPower[dstOffset + srcPrimitive] = power;
	dstTriangleMetadata[dstOffset + srcPrimitive].instanceIndex = pushConsts.instanceIndex;
	dstTriangleMetadata[dstOffset + srcPrimitive].geometryIndex = pushConsts.geometryIndex;
}