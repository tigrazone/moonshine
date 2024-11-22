#pragma once

#include "../utils/mappings.hlsl"
#include "../utils/helpers.hlsl"
#include "ray.hlsl"

// in camera space, cameras are oriented:
// * forward along +X
// * right along +Y
// * up along +Z

struct ThinLens {
    float vfov_tan;
    float aspect;
    float aperture;
    float focusDistance;

    Ray generateRay(const float2 rand, const float2 uv) {
        const float2 uvNDC = uv + uv - 1;

        const float halfViewportHeight = vfov_tan;
        const float halfViewportWidth = aspect * halfViewportHeight;
        const float2 halfViewport = float2(halfViewportWidth, halfViewportHeight);

        const float3 directionCameraSpaceUnorm = float3(1.0, uvNDC * halfViewport);

        const float2 lens = aperture > 0.0 ? aperture * squareToUniformDiskConcentric(rand) * 0.5f : float2(0,0);
        const float3 focus = focusDistance * directionCameraSpaceUnorm;

        Ray ray;
        ray.origin = float3(0.0, lens);
        ray.direction = normalize(focus - ray.origin);
        ray.pdf = 1.#INF;

        return ray;
    }
};

struct Camera {
    float3 origin;
    float3 forward;
    float3 up;
    float vfov;
    float vfov_tan;
    float aperture;
    float focusDistance;
    float3 u;
    float3 v;

    Ray generateRay(const float aspect, const float2 uv, const float2 rand) {
        const ThinLens thinLens = {vfov_tan, aspect, aperture, focusDistance};
        const Ray rayCameraSpace = thinLens.generateRay(rand, uv);

        const float4x3 toWorld = {forward, u, v, origin};

        return rayCameraSpace.transformed(toWorld);
    }
};

