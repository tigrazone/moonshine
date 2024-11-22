#pragma once

#include "math.hlsl"

float2 squareToTriangle(float2 square) {
    float a = 1 - sqrt(1 - square.x);
    return float2(a, square.y * (1 - a));
}

float2 squareToUniformDiskConcentric(float2 uOffset) {
    uOffset += uOffset - 1;

    if (isZERO(uOffset.x) && isZERO(uOffset.y)) {
        return float2(0.0, 0.0);
    }

    float theta, r;

    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = M_PI_4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = M_PI_2 - M_PI_4 * (uOffset.x / uOffset.y);
    }
    sincos(theta, uOffset.y, uOffset.x);
    return r * uOffset;
}

float3 squareToCosineHemisphere(float2 square) {
    float2 d = squareToUniformDiskConcentric(square);
    return float3(d, sqrt(max(0.0, 1.0 - dot(d, d))));
}

float3 sphericalToCartesian(float sinTheta, float cosTheta, float phi) {
    return float3(sinTheta * float2(cos(phi), sin(phi)), cosTheta);
}

// (phi, theta) -- ([0, 2pi], [0, pi])
// assumes vector normalized
float2 cartesianToSpherical(float3 v) {
    float p = atan2(v.y, v.x);
    return float2((p < 0) ? (p + M_TWO_PI) : p, acos(v.z));
}

// from PBRTv4 3.8.3 "Equal-Area Mapping"
float3 squareToEqualAreaSphere(float2 uv) {
	uv += uv - 1;
	const float2 uvp = abs(uv);

	const float signedDistance = 1.0 - (uvp.x + uvp.y);
	const float r = 1.0 - abs(signedDistance);

	if(isZERO(r)) return float3(0, 0, sign(signedDistance));

	const float phi = ((uvp.y - uvp.x) / r + 1.0) * M_PI_4;

	float2 sin_cos;
	sincos(phi, sin_cos.y, sin_cos.x);
	return sign(float3(uv, signedDistance)) * float3(
		sin_cos * r * sqrt(2.0 - r * r),
		1.0 - r * r
	);
}

float2 squareToEqualAreaSphereInverse(float3 dir) {
	const float3 xyz = abs(dir);

	float phi = (isZERO(xyz.x) && isZERO(xyz.y)) ? 0.0 : atan2(min(xyz.x, xyz.y), max(xyz.x, xyz.y)) * M_2INV_PI;
	if (xyz.x < xyz.y) phi = 1.0 - phi;

	float2 uv = float2(1.0 - phi, phi) * sqrt(1.0 - xyz.z);

	if (dir.z < 0) uv = float2(1.0 - uv.y, 1.0 - uv.x);

	uv *= sign(dir.xy);

	return float2(1.0 + uv.x, 1.0 + uv.y) * 0.5;
}

// selects true with probability p (false otherwise),
// remapping rand back into (0..1)
bool coinFlipRemap(float p, inout float rand) {
    if (rand < p) {
        rand /= p;
        return true;
    } else {
        rand = (rand - p) / (1.0 - p);
        return false;
    }
}
