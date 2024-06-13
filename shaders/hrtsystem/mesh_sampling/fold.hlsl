#include "../../utils/helpers.hlsl"

[[vk::binding(0, 0)]] Texture1D<float> srcMip;
[[vk::binding(1, 0)]] RWTexture1D<float> dstMip;

[numthreads(32, 1, 1)]
void main(uint3 dispatchXYZ: SV_DispatchThreadID) {
	const uint dstIndex = dispatchXYZ.x;
	const uint dstImageSize = textureDimensions(dstMip);

	if (any(dstIndex >= dstImageSize)) return;

	dstMip[dstIndex] = srcMip[2 * dstIndex + 0]
	                 + srcMip[2 * dstIndex + 1];
}