#include "../../utils/helpers.hlsl"
#include "../../utils/mappings.hlsl"

[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] Texture2D<float3> srcTexture;
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] SamplerState srcTextureSampler;

[[vk::binding(1, 0)]] RWTexture2D<float4> dstImage;

[numthreads(8, 8, 1)]
void main(uint3 dispatchXYZ: SV_DispatchThreadID) {
	#define pixelIndex		(dispatchXYZ.xy)
	#define dstImageSize	(textureDimensions(dstImage))

	if (any(pixelIndex >= dstImageSize)) return;

	float3 color = float3(0.0, 0.0, 0.0);
	#define samples_per_dim	3
	const float2 cartesianToSpherical_angles_div = 1.0 / float2(M_TWO_PI, PI);
	const float2 subpixel_div = 1.0 / float2(samples_per_dim + 1, samples_per_dim + 1);
	const float2 dstImageSize_div = 1.0 / float2(dstImageSize);
	for (uint i = 0; i < samples_per_dim; i++) {
		for (uint j = 0; j < samples_per_dim; j++) {
			const float2 subpixel = float2(1 + i, 1 + j) * subpixel_div;
			const float2 dstCoords = (float2(pixelIndex) + subpixel) * dstImageSize_div;
			const float3 dir = squareToEqualAreaSphere(dstCoords);
			const float2 srcCoords = cartesianToSpherical(dir) * cartesianToSpherical_angles_div;
			// not sure if there's a standard canonical environment map orientation,
			// but rotate this half a turn so that our default matches blender
			const float2 srcCoordsRotated = frac(float2(srcCoords.x + 0.5, srcCoords.y));
			color += srcTexture.SampleLevel(srcTextureSampler, srcCoordsRotated, 0); // could also technically compute some sort of gradient and get area subtended by this pixel
		}
	}

	const uint total_samples = samples_per_dim * samples_per_dim;
	dstImage[pixelIndex] = float4(color / float(total_samples), 1);

	#undef pixelIndex
	#undef dstImageSize
	#undef samples_per_dim
}