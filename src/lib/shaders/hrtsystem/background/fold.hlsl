#include "../../utils/helpers.hlsl"

[[vk::binding(0, 0)]] Texture2D<float> srcMip;
[[vk::binding(1, 0)]] RWTexture2D<float> dstMip;

[numthreads(8, 8, 1)]
void main(uint3 dispatchXYZ: SV_DispatchThreadID) {
	#define pixelIndex		(dispatchXYZ.xy)
	#define dstImageSize	(textureDimensions(dstMip))

	if (any(pixelIndex >= dstImageSize)) return;
	const uint2 pixelIndex2 = pixelIndex + pixelIndex;

	dstMip[pixelIndex] = srcMip[pixelIndex2]
	                   + srcMip[uint2(pixelIndex2.x + 1, pixelIndex2.y)]
	                   + srcMip[uint2(pixelIndex2.x, pixelIndex2.y + 1)]
	                   + srcMip[uint2(pixelIndex2.x + 1, pixelIndex2.y + 1)];

	#undef pixelIndex
	#undef dstImageSize
}