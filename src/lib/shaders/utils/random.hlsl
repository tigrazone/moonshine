#pragma once

#include "math.hlsl"

// Returns a float between 0 and 1
#define uint_to_float(x) ( asfloat(0x3f800000 | ((x) >> 9)) - 1.0f )

namespace Hash {
    /*
    > High-quality hash that takes 96 bits of data and outputs 32, roughly twice
    > as slow as `pcg`.

    You can use this to generate a seed for subsequent random number generators;
    for instance, provide `uvec3(pixel.x, pixel.y, frame_number).

    From https://github.com/Cyan4973/xxHash and https://www.shadertoy.com/view/XlGcRh.
    */
    uint xxhash32(uint3 p)
    {
      const uint4 primes = uint4(2246822519U, 3266489917U, 668265263U, 374761393U);
      uint        h32;
      h32 = p.z + primes.w + p.x * primes.y;
      h32 = primes.z * ((h32 << 17) | (h32 >> 15));
      h32 += p.y * primes.y;
      h32 = primes.z * ((h32 << 17) | (h32 >> 15));
      h32 = primes.x * (h32 ^ (h32 >> 15));
      h32 = primes.y * (h32 ^ (h32 >> 13));
      return h32 ^ (h32 >> 16);
    }

    uint4 pcg4d(uint4 v)
    {
        v = v * 1664525u + 1013904223u;
        v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
        v = v ^ (v >> 16u);
        v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
        return v;
    }
}

struct Rng {
    uint4 state;

    static Rng fromSeed(uint3 seed) {
        Rng rng;
        rng.state = uint4(seed.x, seed.y, seed.z, Hash::xxhash32(uint3(seed.x, seed.y, seed.z)));
        return rng;
    }

    void stepState() {
        state = Hash::pcg4d(state);
    }

    float getFloat() {
        stepState();
        return uint_to_float(state.x);
    }

    float2 getFloat2() {
        stepState();
        return float2(uint_to_float(state.x), uint_to_float(state.y));
    }
};
