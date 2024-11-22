#pragma once

[[vk::binding(0, 1)]] Texture2D dTextures[];
[[vk::binding(1, 1)]] SamplerState dTextureSampler;

#include "../utils/math.hlsl"
#include "../utils/mappings.hlsl"
#include "spectrum.hlsl"

Frame createTextureFrame(float3 normalWorldSpace, Frame tangentFrame) {
    Frame textureFrame = tangentFrame;
    textureFrame.n = normalWorldSpace;
    textureFrame.reorthogonalize();

    return textureFrame;
}

enum class BSDFType : uint {
    Glass,
    Lambert,
    PerfectMirror,
    StandardPBR,
};

struct Material {
    uint normal;
    uint emissive;

    // find appropriate thing to decode from address using `type`
    BSDFType type;
    uint64_t addr;

    Frame getTextureFrame(float2 texcoords, Frame tangentFrame) {
        float3 normalTangentSpace = dTextures[NonUniformResourceIndex(normal)].SampleLevel(dTextureSampler, texcoords, 0).rgb;
        const float3 normalWorldSpace = normalize(tangentFrame.frameToWorld(normalTangentSpace + normalTangentSpace - 1)).xyz;
        return createTextureFrame(normalWorldSpace, tangentFrame);
    }

    float getEmissive(float λ, float2 texcoords) {
        return Spectrum::sampleEmission(λ, dTextures[NonUniformResourceIndex(emissive)].SampleLevel(dTextureSampler, texcoords, 0).rgb);
    }
};

// all code below expects stuff to be in the reflection frame

interface MicrofacetDistribution {
    float D(float3 m);
    float G(float3 w_i, float3 w_o); // smith, backfacing facets should be ignored elsewhere
    float3 sample(float3 w_o, float2 square);
    float pdf(float3 w_o, float3 m);
};

// AKA Trowbridge-Reitz
struct GGX : MicrofacetDistribution {
    float α;
    float α2;
    float2 alpha;

    static GGX create(float α) {
        GGX ggx;
        ggx.α = α;
        ggx.α2 = α * α;
        ggx.alpha = float2(α, α);
        return ggx;
    }

    // GGX NDF
    // m must be in frame space
    float D(float3 m) {
        return α2 / (PI * pow2F(pow2(Frame::cosTheta(m)) * (α2 - 1) + 1));
    }

    float Λ(float3 v) {
        float tan_theta_v_squared = Frame::tan2Theta(v);
        if (isinf(tan_theta_v_squared)) return 0.0f;
        return (sqrt(1.0f + α2 * tan_theta_v_squared) - 1.0f) * 0.5f;
    }

    // w_i, w_o must be in frame space
    float G(float3 w_i, float3 w_o) {
        return 1.0f / (1.0f + Λ(w_i) + Λ(w_o));
    }

    // Kenta Eto and Yusuke Tokuyoshi. 2023. Bounded VNDF Sampling for Smith-GGX Reflections.
    // In SIGGRAPH Asia 2023 Technical Communications (SA Technical Communications '23), December 12-15, 2023, Sydney, NSW, Australia. ACM, New York, NY, USA, 4 pages.
    // https://gpuopen.com/download/publications/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf
    float3 sample(float3 i, float2 rand) {
        float3 i_std = normalize ( float3 ( i.xy * alpha , i.z ) ) ;
        // Sample a spherical cap
        float phi = M_TWO_PI * rand.x ;
        float b = i_std. z ;
        if(i.z > 0) {
            float a = saturate(min(alpha.x, alpha.y)); // Eq. 6
            float awiz_s = a * i.z / (1.0f + length(i.xy));
            b *= ((1.0f - a * a) / (1.0f + awiz_s * awiz_s));
        }

        float z = mad(1.0f - rand .y , 1.0f + b , -b ) ;

        float2 sin_cos;
        sincos(phi, sin_cos.y, sin_cos.x);
        float3 o_std = float3( sqrt ( saturate (1.0f - z * z ) ) * sin_cos , z);

        // Compute the microfacet normal m
        float3 m_std = i_std + o_std ;
        // Return the reflection vector o
        return normalize(float3( m_std.xy * alpha , m_std.z ));
    }

    float pdf ( float3 i , float3 m ) {
        float2 ai = alpha * i.xy;
        float len2 = dot( ai, ai );
        float t = sqrt ( len2 + i.z * i.z );
        if(i.z >= 0.0f) {
            float a = saturate(min(alpha.x, alpha.y)); // Eq. 6
            float awiz_s = a * i.z / (1.0f + length(i.xy));

            float awiz_s21 = 1.0f + awiz_s * awiz_s;
            return (D(m) * 0.5f * awiz_s21) / ((1.0f - a * a) * i.z + t * awiz_s21) ; // Eq. 8 * || dm/do ||
            //7* 2/
        }
        // Numerically stable form of the previous PDF for i.z < 0
        return D(m) * ( t - i.z ) / (len2 + len2 ) ; // = Eq. 7 * || dm/do ||
    }
};

// ηi is index of refraction for medium on current side of boundary
// ηt is index of refraction for medium on other side of boundary
namespace Fresnel {

    float schlickWeight(float cosTheta) {
        return pow5(1 - cosTheta);
    }

    float schlick(float cosTheta, float R0) {
        return lerp(schlickWeight(cosTheta), 1, R0);
    }

    // boundary of two dielectric surfaces
    // PBRT version
    float dielectric(float cosThetaI, float ηi, float ηt) {
        cosThetaI = clamp(cosThetaI, -1, 1);
        float divided;

        // potentially swap indices of refraction
        // TODO: should this be here?
        bool entering = cosThetaI > 0;
        if (!entering) {
            divided = ηt / ηi;
            cosThetaI = -cosThetaI;
        } else divided = ηi / ηt;

        // compute cosThetaT using Snell's Law
        float sinThetaT2 = pow2(divided) * max(0, 1 - cosThetaI * cosThetaI);

        // handle total internal reflection
        if (sinThetaT2 >= 1) return 1;

        float cosThetaT = sqrt(max(0, 1 - sinThetaT2));

        /*
        float r_parl = ((ηt * cosThetaI) - (ηi * cosThetaT)) / ((ηt * cosThetaI) + (ηi * cosThetaT));
        float r_perp = ((ηi * cosThetaI) - (ηt * cosThetaT)) / ((ηi * cosThetaI) + (ηt * cosThetaT));
        return (r_parl * r_parl + r_perp * r_perp) * 0.5f;
        */

        float A0 = ηt * cosThetaI;
        float A1 = ηi * cosThetaT;
        float B0 = ηi * cosThetaI;
        float B1 = ηt * cosThetaT;

        float r_parl = (A0 - A1) / (A0 + A1);
        float r_perp = (B0 - B1) / (B0 + B1);
        return (r_parl * r_parl + r_perp * r_perp) * 0.5f; //11* 2/ -> 6* 2/
    }
};

struct BSDFEvaluation {
    float reflectance;
    float pdf;

    static BSDFEvaluation empty() {
        BSDFEvaluation eval;
        eval.reflectance = 0;
        eval.pdf = 0;
        return eval;
    }
};

struct BSDFSample {
    float3 dirFs;
    BSDFEvaluation eval;
};

interface BSDF {
    BSDFEvaluation evaluate(float3 w_o, float3 w_i);
    BSDFSample sample(float3 w_o, float2 square);
};

// evenly diffuse lambertian material
struct Lambert : BSDF {
    float reflectance; // fraction of light that is reflected

    static Lambert create(float reflectance) {
        Lambert lambert;
        lambert.reflectance = reflectance;
        return lambert;
    }

    static Lambert load(const uint64_t addr, const float2 texcoords, float λ) {
        uint colorTextureIndex = vk::RawBufferLoad<uint>(addr);

        Lambert material;
        material.reflectance = Spectrum::sampleReflectance(λ, dTextures[NonUniformResourceIndex(colorTextureIndex)].SampleLevel(dTextureSampler, texcoords, 0).rgb);
        return material;
    }

    BSDFEvaluation evaluate(float3 w_i, float3 w_o) {
        BSDFEvaluation eval;
        eval.pdf = Frame::sameHemisphere(w_i, w_o) ? abs(Frame::cosTheta(w_i)) * M_INV_PI : 0.0;
        eval.reflectance = reflectance * eval.pdf;
        return eval;
    }

    BSDFEvaluation evaluateShort(float3 w_i, float3 w_o) {
        BSDFEvaluation eval;
        if(Frame::sameHemisphere(w_i, w_o)) {
            eval.pdf = abs(Frame::cosTheta(w_i)) * M_INV_PI;
            eval.reflectance = reflectance;
        } else {
            eval.pdf = 0.0;
            eval.reflectance = 0.0;
        }
        return eval;
    }

    BSDFSample sample(float3 w_o, float2 square) {
        float3 w_i = squareToCosineHemisphere(square);
        if (w_o.z < 0.0) w_i.z = -w_i.z;

        BSDFSample sample;
        sample.dirFs = w_i;
        sample.eval = evaluateShort(w_i, w_o);
        return sample;
    }

    static bool isDelta() {
        return false;
    }
};

// blends between provided microfacet distribution
// and lambertian diffuse based on metalness factor
struct StandardPBR : BSDF {
    GGX distr;      // microfacet distribution used by this material

    float reflectance; // reflectance - everywhere within [0, 1]
    float metalness; // metalness - k_s - part it is specular. diffuse is (1 - specular); [0, 1]
    float ior; // ior - internal index of refraction; [0, inf)pSpecularSample
    float pSpecularSample;

    static StandardPBR load(const uint64_t addr, const float2 texcoords, const float λ) {
        uint colorTextureIndex = vk::RawBufferLoad<uint>(addr);
        uint metalnessTextureIndex = vk::RawBufferLoad<uint>(addr + sizeof(uint) * 1);
        uint roughnessTextureIndex = vk::RawBufferLoad<uint>(addr + sizeof(uint) * 2);
        float ior = vk::RawBufferLoad<float>(addr + sizeof(uint) * 3);

        StandardPBR material;
        material.reflectance = Spectrum::sampleReflectance(λ, dTextures[NonUniformResourceIndex(colorTextureIndex)].SampleLevel(dTextureSampler, texcoords, 0).rgb);
        material.metalness = dTextures[NonUniformResourceIndex(metalnessTextureIndex)].SampleLevel(dTextureSampler, texcoords, 0).r;
        float roughness = dTextures[NonUniformResourceIndex(roughnessTextureIndex)].SampleLevel(dTextureSampler, texcoords, 0).r;
        material.distr = GGX::create(max(pow2(roughness), 0.001));
        material.ior = ior;
        material.pSpecularSample = 1.0 / (2.0 - material.metalness);
        return material;
    }

    float microfacetPdf(float3 w_i, float3 w_o) {
        if (!Frame::sameHemisphere(w_o, w_i)) return 0.0;
        float3 h = normalize(w_i + w_o);
        return distr.pdf(w_o, h);
    }

    BSDFSample sample(float3 w_o, float2 square) {
        /*
        float specularWeight = 1;
        float diffuseWeight = 1 - metalness;
        float pSpecularSample = specularWeight / (specularWeight + diffuseWeight);
        */

        BSDFSample sample;
        if (coinFlipRemap(pSpecularSample, square.x)) {
            float3 h = distr.sample(w_o, square);
            sample.dirFs = -reflect(w_o, h);
        } else {
            sample.dirFs = Lambert::create(reflectance).sample(w_o, square).dirFs;
        }
        sample.eval = evaluate(sample.dirFs, w_o);
        // ideally we would never sample something with a zero pdf...
        // not sure if there's a bug here currently or if this is to be expected
        sample.eval.reflectance = sample.eval.pdf > 0 ? sample.eval.reflectance / sample.eval.pdf : 0;
        return sample;
    }

    BSDFEvaluation evaluate(float3 w_i, float3 w_o) {
        BSDFEvaluation diffuseEvaluation = Lambert::create(reflectance).evaluate(w_i, w_o);
        float diffuse = diffuseEvaluation.reflectance;

        BSDFEvaluation eval;
        eval.reflectance = (1.0 - metalness) * diffuse;

        float lambert_pdf = diffuseEvaluation.pdf;
        float micro_pdf = microfacetPdf(w_i, w_o);

        eval.pdf = lerp(lambert_pdf, micro_pdf, pSpecularSample);

        if(Frame::sameHemisphere(w_o, w_i)) {
            float3 h = normalize(w_i + w_o);
            float dot_w_i_h = dot(w_i, h);
            float fDielectric = Fresnel::dielectric(dot_w_i_h, AIR_IOR, ior);

            float F = metalness > NEARzero ? lerp(fDielectric, Fresnel::schlick(dot_w_i_h, reflectance), metalness) : fDielectric;
            float G = distr.G(w_i, w_o);
            float D = distr.D(h);
            eval.reflectance += (F * G * D) / (4 * abs(Frame::cosTheta(w_o)));
        }
        return eval;
    }

    static bool isDelta() {
        return false;
    }
};

struct PerfectMirror : BSDF {
    BSDFSample sample(float3 w_o, float2 square) {
        BSDFSample sample;
        sample.dirFs = float3(-w_o.xy, w_o.z);
        sample.eval.reflectance = 1;
        sample.eval.pdf = 1.#INF;
        return sample;
    }

    BSDFEvaluation evaluate(float3 w_i, float3 w_o) {
        return BSDFEvaluation::empty();
    }

    static bool isDelta() {
        return true;
    }
};

float3 refractDir(float3 wi, bool side, float eta) {
    float sin2ThetaT = eta * eta * max(0, 1 - wi.z * wi.z);
    if (sin2ThetaT >= 1) return 0.0;

    return float3((-eta) * wi.xy, side ? -sqrt(1 - sin2ThetaT) : sqrt(1 - sin2ThetaT));
}

float cauchyIOR(const float a, const float b, const float λ) {
    return a + b / (λ * λ);
}

struct Glass : BSDF {
    float intIOR;

    static Glass load(const uint64_t addr, const float λ) {
        Glass material;
        const float a = vk::RawBufferLoad<float>(addr);
        const float b = vk::RawBufferLoad<float>(addr + sizeof(float));
        material.intIOR = cauchyIOR(a, b, λ);
        return material;
    }

    BSDFSample sample(float3 w_o, float2 square) {
        float fresnel = Fresnel::dielectric(Frame::cosTheta(w_o), AIR_IOR, intIOR);
        BSDFSample sample;

        if (coinFlipRemap(fresnel, square.x)) {
            sample.dirFs = float3(-w_o.xy, w_o.z);
        } else {
            float etaI;
            float etaT;
            if (Frame::cosTheta(w_o) > 0) {
                etaI = AIR_IOR;
                etaT = intIOR;
            } else {
                etaT = AIR_IOR;
                etaI = intIOR;
            }
            sample.dirFs = refractDir(w_o, w_o.z > 0, etaI / etaT);
        }

        if (isNotZERO(sample.dirFs.x) && isNotZERO(sample.dirFs.y) && isNotZERO(sample.dirFs.z)) {
            sample.eval.reflectance = 1;
            sample.eval.pdf = 1.#INF;
        } else {
            sample.eval = BSDFEvaluation::empty();
        }
        return sample;
    }

    BSDFEvaluation evaluate(float3 w_i, float3 w_o) {
        return BSDFEvaluation::empty();
    }

    static bool isDelta() {
        return true;
    }
};

struct PolymorphicBSDF : BSDF {
    BSDFType type;
    uint64_t addr;
    float2 texcoords;
    float λ;

    static PolymorphicBSDF load(Material material, float2 texcoords, float λ) {
        PolymorphicBSDF bsdf;
        bsdf.type = material.type;
        bsdf.addr = material.addr;
        bsdf.texcoords = texcoords;
        bsdf.λ = λ;
        return bsdf;
    }

    BSDFEvaluation evaluate(float3 w_i, float3 w_o) {
        switch (type) {
            case BSDFType::StandardPBR: {
                StandardPBR m = StandardPBR::load(addr, texcoords, λ);
                return m.evaluate(w_i, w_o);
            }
            case BSDFType::Lambert: {
                Lambert m = Lambert::load(addr, texcoords, λ);
                return m.evaluate(w_i, w_o);
            }
            case BSDFType::PerfectMirror: {
                PerfectMirror m;
                return m.evaluate(w_i, w_o);
            }
            case BSDFType::Glass: {
                Glass m = Glass::load(addr, λ);
                return m.evaluate(w_i, w_o);
            }
        }
    }

    BSDFSample sample(float3 w_o, float2 square) {
        switch (type) {
            case BSDFType::StandardPBR: {
                StandardPBR m = StandardPBR::load(addr, texcoords, λ);
                return m.sample(w_o, square);
            }
            case BSDFType::Lambert: {
                Lambert m = Lambert::load(addr, texcoords, λ);
                return m.sample(w_o, square);
            }
            case BSDFType::PerfectMirror: {
                PerfectMirror m;
                return m.sample(w_o, square);
            }
            case BSDFType::Glass: {
                Glass m = Glass::load(addr, λ);
                return m.sample(w_o, square);
            }
        }
    }

    bool isDelta() {
        switch (type) {
            case BSDFType::StandardPBR: {
                return StandardPBR::isDelta();
            }
            case BSDFType::Lambert: {
                return Lambert::isDelta();
            }
            case BSDFType::PerfectMirror: {
                return PerfectMirror::isDelta();
            }
            case BSDFType::Glass: {
                return Glass::isDelta();
            }
        }
    }
};

