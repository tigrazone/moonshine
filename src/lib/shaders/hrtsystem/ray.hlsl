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
};