#pragma once

struct Ray {
    float3 origin;
    float3 direction;
    float pdf; // probability this ray was sampled

    RayDesc desc() {
        RayDesc desc;
        desc.Origin = origin;
        desc.Direction = direction;
        desc.TMin = 0;
        desc.TMax = 1.#INF;
        return desc;
    }

    Ray transformed(float4x3 mat) {
        Ray ray;
        ray.origin = mul(float4(origin, 1.0), mat);
        ray.direction = normalize(mul(float4(direction, 0.0), mat));
        ray.pdf = pdf;
        return ray;
    }
};