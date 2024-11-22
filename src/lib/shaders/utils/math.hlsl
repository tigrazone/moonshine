#pragma once

static const float PI = 3.14159265;
static const float EPSILON = 0.000000119;
#define MAX_UINT    0xFFFFFFFF
static const float AIR_IOR = 1.000277;

#define NEARzero 1e-25f
#define isZERO(x) ((x)>-NEARzero && (x)<NEARzero)
#define isNotZERO(x) ((x)>NEARzero || (x)<-NEARzero)

#define	pow2(x) ( (x) * (x) )

float	pow5(float x) { return x * x * x * x * x; }

#define atanh(x) ( log((1 + (x)) / (1 - (x))) * 0.5f )

#define luminance(color) ( dot(float3(0.212671f, 0.715160f, 0.072169f), (color)) )

#define faceForward(n, d) ( dot((n), (d)) > 0 ? (n) : -(n) )

// https://www.nu42.com/2015/03/how-you-average-numbers.html
#define accumulate(priorAverage, newSample, sampleCount) ( (priorAverage) + ((newSample) - (priorAverage)) / float((sampleCount) + 1) )

void coordinateSystem(float3 normal, out float3 tangent, out float3 bitangent)
{
  if(normal.z < -0.99998796f)  // Handle the singularity
  {
    tangent   = float3(0.0f, -1.0f, 0.0f);
    bitangent = float3(-1.0f, 0.0f, 0.0f);
    return;
  }

  float inv_1plus_nz = 1.0f / (1.0f + normal.z);
  float nxa = -normal.x * inv_1plus_nz;
  tangent   = float3(1.0f + normal.x * nxa, nxa * normal.y, -normal.x);
  bitangent = float3(tangent.y, 1.0f - normal.y * normal.y * inv_1plus_nz, -normal.y);
}
