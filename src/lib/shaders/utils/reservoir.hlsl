#pragma once

#include "mappings.hlsl"

template <class Candidate>
struct Reservoir {
    Candidate selected;
    float weightSum;

    static Reservoir empty() {
        Reservoir r;
        r.weightSum = 0.0;
        return r;
    }

    void update(const Candidate newCandidate, const float newWeight, inout float rand) {
        weightSum += newWeight;
        if (coinFlipRemap(newWeight / weightSum, rand)) {
            selected = newCandidate;
        }
    }

    // not valid to look at selected unless this returns true
    bool valid() {
        return weightSum != 0;
    }
};
