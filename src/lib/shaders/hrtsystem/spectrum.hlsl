#pragma once

[[vk::binding(0, 2)]] SamplerState dSpectrumSampler;
[[vk::binding(1, 2)]] Texture1D<float> dSpectrumCIEX;
[[vk::binding(2, 2)]] Texture1D<float> dSpectrumCIEY;
[[vk::binding(3, 2)]] Texture1D<float> dSpectrumCIEZ;
[[vk::binding(4, 2)]] Texture1D<float> dSpectrumR;
[[vk::binding(5, 2)]] Texture1D<float> dSpectrumG;
[[vk::binding(6, 2)]] Texture1D<float> dSpectrumB;
[[vk::binding(7, 2)]] Texture1D<float> dSpectrumD65;

#include "../utils/random.hlsl"
#include "../utils/math.hlsl"
#include "material.hlsl"

//static const float CIE1931YIntegral = 106.85691710117189;
#define CIE1931YIntegral_   0.0093583085490884124791293370648
#define XYZdelta_           0.00212314225053078556263269639066
#define XYZdelta_2          0.00188323917137476459510357815443

namespace Spectrum {
    // exclusive range
    // precalculated
    float sampleTabulated(const float λdelta, Texture1D<float> t) {
        return t.SampleLevel(dTextureSampler, λdelta, 0);
    }

    float sampleReflectance(const float λ, const float3 reflectance) {
        const float λdelta = (λ - 360) * XYZdelta_;
        const float3 rgb = float3(
            sampleTabulated(λdelta, dSpectrumR),
            sampleTabulated(λdelta, dSpectrumG),
            sampleTabulated(λdelta, dSpectrumB)
        );
        return dot(rgb, reflectance);
    }

    // a somewhat roundabout way of doing this but I believe it's correct
    float sampleEmission(const float λ, const float3 emission) {
        const float sampledReflectance = sampleReflectance(λ, emission);
        const float sampledD65 = sampleTabulated((λ - 300) * XYZdelta_2, dSpectrumD65);
        return sampledReflectance * sampledD65;
    }

    float3 toXYZ(const float λ, const float s) {
        const float λdelta = (λ - 360) * XYZdelta_;
        const float3 rgb = float3(
            sampleTabulated(λdelta, dSpectrumCIEX),
            sampleTabulated(λdelta, dSpectrumCIEY),
            sampleTabulated(λdelta, dSpectrumCIEZ)
        );
        return rgb * (s * CIE1931YIntegral_);
    }

    float3 toLinearSRGB(const float λ, const float s) {
        const float3 xyz = toXYZ(λ, s);
        const float3x3 XYZtoLinearSRGB = { 0.03276749869518854, -0.015543557073358668, -0.00504115364541362, -0.009800789737065342, 0.018969392573362078, 0.00042019608374440075, 0.0005626856849785213, -0.0020631808449212427, 0.010691028014591895 };
        return mul(XYZtoLinearSRGB, xyz);
    }
};

struct WavelengthSample {
    float λ;
    float pdf;

    static WavelengthSample sampleUniform(const float start, const float end, const float rand) {
        WavelengthSample s;
        s.λ = lerp(start, end, rand);
        s.pdf = 1 / (end - start);
        return s;
    }

    static WavelengthSample sampleVisible(const float rand) {
        WavelengthSample s;
        s.λ = 538 - 138.888889f * atanh(0.85691062f - 1.82750197f * rand);
        s.pdf = 0.0039398042f / pow2(cosh(0.0072f * (s.λ - 538)));
        return s;
    }
};
