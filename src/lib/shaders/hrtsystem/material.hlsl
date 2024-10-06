#pragma once

[[vk::binding(0, 1)]] Texture2D dTextures[];
[[vk::binding(1, 1)]] SamplerState dTextureSampler;

#include "../utils/math.hlsl"
#include "../utils/mappings.hlsl"
#include "spectrum.hlsl"

float3 decodeNormal(float2 rg) {
    rg = rg * 2 - 1;
    return float3(rg, sqrt(1.0 - saturate(dot(rg, rg)))); // saturate due to float/compression annoyingness
}

float3 tangentNormalToWorld(float3 normalTangentSpace, Frame tangentFrame) {
    return normalize(tangentFrame.frameToWorld(normalTangentSpace)).xyz;
}

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
        const float2 rg = dTextures[NonUniformResourceIndex(normal)].SampleLevel(dTextureSampler, texcoords, 0).rg;
        const float3 normalTangentSpace = decodeNormal(rg);
        const float3 normalWorldSpace = tangentNormalToWorld(normalTangentSpace, tangentFrame);
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

    static GGX create(float α) {
        GGX ggx;
        ggx.α = α;
        return ggx;
    }

    // GGX NDF
    // m must be in frame space
    float D(float3 m) {
        float α2 = pow(α, 2);
        float denom = PI * pow(pow(Frame::cosTheta(m), 2) * (α2 - 1) + 1, 2);
        return α2 / denom;
    }

    float Λ(float3 v) {
        float tan_theta_v_squared = Frame::tan2Theta(v);
        if (isinf(tan_theta_v_squared)) return 0.0f;
        return (sqrt(1.0f + pow(α, 2) * tan_theta_v_squared) - 1.0f) / 2.0f;
    }

    // w_i, w_o must be in frame space
    float G(float3 w_i, float3 w_o) {
        return 1.0f / (1.0f + Λ(w_i) + Λ(w_o));
    }

    // samples a half vector from the distribution
    // TODO: sample visible normals
    float3 sample(float3 w_o, float2 square) {
        // figure out spherical coords of half vector
        float tanThetaSquared = α * α * square.x / (1 - square.x);
        float cosThetaSquared = 1 / (1 + tanThetaSquared);
        float sinTheta = sqrt(max(0, 1 - cosThetaSquared));
        float cosTheta = sqrt(cosThetaSquared);
        float phi = 2 * PI * square.y;

        // convert them to cartesian
        float3 h = sphericalToCartesian(sinTheta, cosTheta, phi);
        if (!Frame::sameHemisphere(w_o, h)) h = -h;
        return h;
    }

    float pdf(float3 w_o, float3 m) {
        return D(m) * abs(Frame::cosTheta(m));
    }
};

// ηi is index of refraction for medium on current side of boundary
// ηt is index of refraction for medium on other side of boundary
namespace Fresnel {
    float schlickR0(float ηi, float ηt) {
        return pow((ηt - ηi) / (ηt + ηi), 2);
    }

    float schlickWeight(float cosTheta) {
        return pow(1 - cosTheta, 5);
    }

    float schlick(float cosTheta, float R0) {
        return lerp(schlickWeight(cosTheta), 1, R0);
    }

    // lerp between layer1 and layer2 based on schlick fresnel
    float fresnelLerp(float cosTheta, float ηi, float ηt, float layer1, float layer2) {
        const float f = schlick(cosTheta, schlickR0(ηi, ηt));
        return lerp(layer1, layer2, f);
    }

    // boundary of two dielectric surfaces
    // PBRT version
    float dielectric(float cosThetaI, float ηi, float ηt) {
        cosThetaI = clamp(cosThetaI, -1, 1);

        // potentially swap indices of refraction
        // TODO: should this be here?
        bool entering = cosThetaI > 0;
        if (!entering) {
            float tmp = ηi;
            ηi = ηt;
            ηt = tmp;
            cosThetaI = abs(cosThetaI);
        }

        // compute cosThetaT using Snell's Law
        float sinThetaI = sqrt(max(0, 1 - cosThetaI * cosThetaI));
        float sinThetaT = ηi / ηt * sinThetaI;

        // handle total internal reflection
        if (sinThetaT >= 1) return 1;

        float cosThetaT = sqrt(max(0, 1 - sinThetaT * sinThetaT));

        float r_parl = ((ηt * cosThetaI) - (ηi * cosThetaT)) / ((ηt * cosThetaI) + (ηi * cosThetaT));
        float r_perp = ((ηi * cosThetaI) - (ηt * cosThetaT)) / ((ηi * cosThetaI) + (ηt * cosThetaT));

        return (r_parl * r_parl + r_perp * r_perp) / 2;
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
        eval.reflectance = Frame::sameHemisphere(w_i, w_o) ? abs(Frame::cosTheta(w_i)) * reflectance / PI : 0.0;
        eval.pdf = Frame::sameHemisphere(w_i, w_o) ? abs(Frame::cosTheta(w_i)) / PI : 0.0;
        return eval;
    }

    BSDFSample sample(float3 w_o, float2 square) {
        float3 w_i = squareToCosineHemisphere(square);
        if (w_o.z < 0.0) w_i.z *= -1;

        BSDFSample sample;
        sample.dirFs = w_i;
        sample.eval = evaluate(w_i, w_o);
        sample.eval.reflectance /= sample.eval.pdf;
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
    float ior; // ior - internal index of refraction; [0, inf)

    static StandardPBR load(const uint64_t addr, const float2 texcoords, const float λ) {
        uint colorTextureIndex = vk::RawBufferLoad<uint>(addr + sizeof(uint) * 0);
        uint metalnessTextureIndex = vk::RawBufferLoad<uint>(addr + sizeof(uint) * 1);
        uint roughnessTextureIndex = vk::RawBufferLoad<uint>(addr + sizeof(uint) * 2);
        float ior = vk::RawBufferLoad<float>(addr + sizeof(uint) * 3);

        StandardPBR material;
        material.reflectance = Spectrum::sampleReflectance(λ, dTextures[NonUniformResourceIndex(colorTextureIndex)].SampleLevel(dTextureSampler, texcoords, 0).rgb);
        material.metalness = dTextures[NonUniformResourceIndex(metalnessTextureIndex)].SampleLevel(dTextureSampler, texcoords, 0).r;
        float roughness = dTextures[NonUniformResourceIndex(roughnessTextureIndex)].SampleLevel(dTextureSampler, texcoords, 0).r;
        material.distr = GGX::create(max(pow(roughness, 2), 0.001));
        material.ior = ior;
        return material;
    }

    float microfacetPdf(float3 w_i, float3 w_o) {
        if (!Frame::sameHemisphere(w_o, w_i)) return 0.0;
        float3 h = normalize(w_i + w_o);
        return distr.pdf(w_o, h) / (4.0 * dot(w_o, h));
    }

    BSDFSample sample(float3 w_o, float2 square) {
        float specularWeight = 1;
        float diffuseWeight = 1 - metalness;
        float pSpecularSample = specularWeight / (specularWeight + diffuseWeight);

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

    float pdf(float3 w_i, float3 w_o) {
        float specularWeight = 1;
        float diffuseWeight = 1 - metalness;
        float pSpecularSample = specularWeight / (specularWeight + diffuseWeight);

        float lambert_pdf = Lambert::create(reflectance).evaluate(w_i, w_o).pdf;
        float micro_pdf = microfacetPdf(w_i, w_o);

        return lerp(lambert_pdf, micro_pdf, pSpecularSample);
    }

    BSDFEvaluation evaluate(float3 w_i, float3 w_o) {
        float3 h = normalize(w_i + w_o);

        float fDielectric = Fresnel::dielectric(dot(w_i, h), AIR_IOR, ior);
        float fMetallic = Fresnel::schlick(dot(w_i, h), reflectance);

        float F = lerp(fDielectric, fMetallic, metalness);
        float G = distr.G(w_i, w_o);
        float D = distr.D(h);
        float specular = Frame::sameHemisphere(w_o, w_i) ? (F * G * D) / (4 * abs(Frame::cosTheta(w_i)) * abs(Frame::cosTheta(w_o))) : 0;

        float diffuse = Lambert::create(reflectance).evaluate(w_i, w_o).reflectance;

        BSDFEvaluation eval;
        eval.reflectance = abs(Frame::cosTheta(w_i)) * specular + (1.0 - metalness) * diffuse;
        eval.pdf = pdf(w_i, w_o);
        return eval;
    }

    static bool isDelta() {
        return false;
    }
};

struct DisneyDiffuse : BSDF {
    float reflectance;
    float roughness;

    static DisneyDiffuse create(float reflectance, float roughness) {
        DisneyDiffuse material;
        material.reflectance = reflectance;
        material.roughness = roughness;
        return material;
    }

    BSDFSample sample(float3 w_o, float2 square) {
        return Lambert::create(reflectance).sample(w_o, square);
    }

    BSDFEvaluation evaluate(float3 w_i, float3 w_o) {
        BSDFEvaluation eval = Lambert::create(reflectance).evaluate(w_i, w_o);

        float3 h = normalize(w_i + w_o);
        float cosThetaHI = dot(w_i, h);

        float cosThetaNI = abs(Frame::cosTheta(w_i));
        float cosThetaNO = abs(Frame::cosTheta(w_o));
        float F_I = pow(1 - cosThetaNI, 5);
        float F_O = pow(1 - cosThetaNO, 5);

        float R_R = 2 * roughness * cosThetaHI * cosThetaHI;
        float retro = R_R * (F_I + F_O + F_I * F_O * (R_R - 1));

        eval.reflectance *= ((1 - F_I / 2) * (1 - F_O / 2) + retro);
        return eval;
    }

    static bool isDelta() {
        return false;
    }
};

struct PerfectMirror : BSDF {
    BSDFSample sample(float3 w_o, float2 square) {
        BSDFSample sample;
        sample.dirFs = float3(-w_o.x, -w_o.y, w_o.z);
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

float3 refractDir(float3 wi, float3 n, float eta) {
    float cosThetaI = dot(n, wi);
    float sin2ThetaI = max(0, 1 - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;
    if (sin2ThetaT >= 1) return 0.0;

    float cosThetaT = sqrt(1 - sin2ThetaT);

    return eta * -wi + (eta * cosThetaI - cosThetaT) * n;
}

struct Glass : BSDF {
    float intIOR;

    static Glass load(uint64_t addr) {
        Glass material;
        material.intIOR = vk::RawBufferLoad<float>(addr);
        return material;
    }

    BSDFSample sample(float3 w_o, float2 square) {
        float fresnel = Fresnel::dielectric(Frame::cosTheta(w_o), AIR_IOR, intIOR);
        BSDFSample sample;

        if (coinFlipRemap(fresnel, square.x)) {
            sample.dirFs = float3(-w_o.x, -w_o.y, w_o.z);
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
            sample.dirFs = refractDir(w_o, faceForward(float3(0.0, 0.0, 1.0), w_o), etaI / etaT);
        }
        if (all(sample.dirFs != 0.0)) {
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
                Glass m = Glass::load(addr);
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
                Glass m = Glass::load(addr);
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

