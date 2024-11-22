#pragma once

#include "../utils/math.hlsl"

struct Frame {
    float3 n; // normal
    float3 s; // tangent
    float3 t; // bitangent

    // takes in a normalized vector, returns a frame where x,y,z are bitangent, normal, and tangent respectively
    static Frame create(float3 n) {
        float3 t, s;
        coordinateSystem(n, t, s);
        return Frame::create(n, s, t);
    }

    static Frame create(float3 n, float3 s, float3 t) {
        Frame frame;
        frame.n = n;
        frame.s = s;
        frame.t = t;
        return frame;
    }

    Frame inSpace(float3x3 m) {
        float3 n2 = normalize(mul(m, n));
        float3 s2 = normalize(mul(m, s));
        float3 t2 = normalize(mul(m, t));

        return Frame::create(n2, s2, t2);
    }

    Frame inSpace(float3x3 m, float3 n2) {
        float3 s2 = normalize(mul(m, s));
        float3 t2 = normalize(mul(m, t));

        return Frame::create(n2, s2, t2);
    }

    void reorthogonalize() {
        // Gram-Schmidt
        s = normalize(s - n * dot(n, s));
        t = normalize(cross(n, s));
    }

    float3 worldToFrame(float3 v) {
        float3x3 toFrame = { s, t, n };
        return mul(toFrame, v);
    }

    float3 frameToWorld(float3 v) {
        float3x3 toFrame = { s, t, n };
        return mul(transpose(toFrame), v);
    }

    static float cosTheta(float3 v) {
        return v.z;
    }

    static float tan2Theta(float3 v) {
        float cos2Theta0 = v.z * v.z;
        return max(0.0, 1.0 - cos2Theta0) / cos2Theta0;
    }

    static bool sameHemisphere(float3 v1, float3 v2) {
        return v1.z * v2.z > 0.0;
    }
};

