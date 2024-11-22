#pragma once

#define	MAX_UINT  0xFFFFFFFF
#define	AIR_IOR   1.000277f

#define	PI          3.1415926535897932384626433832795f
#define	M_TWO_PI	6.283185307179586476925286766559f  // 2*PI
#define	M_4PI       0.07957747154594766788444188168626f  // 1/(4*PI)
#define	M_INV_PI	0.31830988618379067153776752674503f  // 1/PI
#define	M_2INV_PI	0.63661977236758134307553505349006f  // 2/PI
#define	M_PI_4		0.78539816339744830961566084581988f  // PI/4
#define	M_PI_2		1.5707963267948966192313216916398f   // PI/2

#define NEARzero 1e-25f
#define isZERO(x) ((x)>-NEARzero && (x)<NEARzero)
#define isNotZERO(x) ((x)>NEARzero || (x)<-NEARzero)

#define	pow2(x) ( (x) * (x) )

float	pow2F(float x) { return x * x; }
float	pow5(float x) { return x * x * x * x * x; }

#define atanh(x) ( log((1 + (x)) / (1 - (x))) * 0.5f )

#define luminance(color) ( dot(float3(0.212671, 0.715160, 0.072169), (color)) )

#define faceForward(n, d) ( dot((n), (d)) > 0 ? (n) : -(n) )

// https://www.nu42.com/2015/03/how-you-average-numbers.html
#define accumulate(priorAverage, newSample, sampleCount) ( (priorAverage) + ((newSample) - (priorAverage)) / float((sampleCount) + 1) )

void coordinateSystem(float3 normal, out float3 tangent, out float3 bitangent)
{
  if(normal.z < -0.99998796)  // Handle the singularity
  {
    tangent   = float3(0.0, -1.0, 0.0);
    bitangent = float3(-1.0, 0.0, 0.0);
    return;
  }

  float inv_1plus_nz = 1.0 / (1.0 + normal.z);
  float nxa = -normal.x * inv_1plus_nz;
  tangent   = float3(1.0 + normal.x * nxa, nxa * normal.y, -normal.x);
  bitangent = float3(tangent.y, 1.0 - normal.y * normal.y * inv_1plus_nz, -normal.y);
}
