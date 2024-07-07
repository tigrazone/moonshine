#include "main_shared.hlsl"

[[vk::constant_id(0)]] const uint dEnvSamplesPerBounce = 1;  // how many times the environment map should be sampled per bounce for light
[[vk::constant_id(1)]] const uint dMeshSamplesPerBounce = 1; // how many times emissive meshes should be sampled per bounce for light

[shader("raygeneration")]
void raygen() {
    const DirectLightIntegrator integrator = DirectLightIntegrator::create(dEnvSamplesPerBounce, dMeshSamplesPerBounce);
    integrate(integrator);
}

