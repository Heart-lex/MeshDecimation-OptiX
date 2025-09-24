/*

 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <optix.h>

#include "optixPathTracer.h"
#include "random.h"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>
#include <texture_indirect_functions.h>

extern "C" {
    __constant__ Params params;
}

//------------------------------------------------------------------------------
//
// Orthonormal basis helper
//
//------------------------------------------------------------------------------


struct Onb
{
    __forceinline__ __device__ Onb(const float3& normal)
    {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};


//------------------------------------------------------------------------------
//
// Utility functions 
//
//------------------------------------------------------------------------------


static __forceinline__ __device__ RadiancePRD loadClosesthitRadiancePRD()
{
    RadiancePRD prd = {};

    prd.attenuation.x = __uint_as_float(optixGetPayload_0());
    prd.attenuation.y = __uint_as_float(optixGetPayload_1());
    prd.attenuation.z = __uint_as_float(optixGetPayload_2());
    prd.seed = optixGetPayload_3();
    prd.depth = optixGetPayload_4();
    return prd;
}

static __forceinline__ __device__ RadiancePRD loadMissRadiancePRD()
{
    RadiancePRD prd = {};
    return prd;
}

static __forceinline__ __device__ void storeClosesthitRadiancePRD(RadiancePRD prd)
{
    optixSetPayload_0(__float_as_uint(prd.attenuation.x));
    optixSetPayload_1(__float_as_uint(prd.attenuation.y));
    optixSetPayload_2(__float_as_uint(prd.attenuation.z));

    optixSetPayload_3(prd.seed);
    optixSetPayload_4(prd.depth);

    optixSetPayload_5(__float_as_uint(prd.emitted.x));
    optixSetPayload_6(__float_as_uint(prd.emitted.y));
    optixSetPayload_7(__float_as_uint(prd.emitted.z));

    optixSetPayload_8(__float_as_uint(prd.radiance.x));
    optixSetPayload_9(__float_as_uint(prd.radiance.y));
    optixSetPayload_10(__float_as_uint(prd.radiance.z));

    optixSetPayload_11(__float_as_uint(prd.origin.x));
    optixSetPayload_12(__float_as_uint(prd.origin.y));
    optixSetPayload_13(__float_as_uint(prd.origin.z));

    optixSetPayload_14(__float_as_uint(prd.direction.x));
    optixSetPayload_15(__float_as_uint(prd.direction.y));
    optixSetPayload_16(__float_as_uint(prd.direction.z));

    optixSetPayload_17(prd.done);
}


static __forceinline__ __device__ void storeMissRadiancePRD(RadiancePRD prd)
{
    optixSetPayload_5(__float_as_uint(prd.emitted.x));
    optixSetPayload_6(__float_as_uint(prd.emitted.y));
    optixSetPayload_7(__float_as_uint(prd.emitted.z));

    optixSetPayload_8(__float_as_uint(prd.radiance.x));
    optixSetPayload_9(__float_as_uint(prd.radiance.y));
    optixSetPayload_10(__float_as_uint(prd.radiance.z));

    optixSetPayload_17(prd.done);
}

static __forceinline__ __device__ AgentPRD loadClosesthitAgentPRD()
{
    AgentPRD prd = {};

	prd.rotation_count = optixGetPayload_0();
    prd.depth = optixGetPayload_1();
    return prd;
}

static __forceinline__ __device__ void storeClosesthitAgentPRD(AgentPRD prd)
{
    optixSetPayload_0(prd.rotation_count);
    optixSetPayload_1(prd.depth);

    optixSetPayload_2(__float_as_uint(prd.origin.x));
    optixSetPayload_3(__float_as_uint(prd.origin.y));
    optixSetPayload_4(__float_as_uint(prd.origin.z));

    optixSetPayload_5(__float_as_uint(prd.direction.x));
    optixSetPayload_6(__float_as_uint(prd.direction.y));
    optixSetPayload_7(__float_as_uint(prd.direction.z));

    optixSetPayload_8(prd.done);
}

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}


static __forceinline__ __device__ void traceRadiance(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    RadiancePRD& prd
)
{
    unsigned int u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17;

    u0 = __float_as_uint(prd.attenuation.x);
    u1 = __float_as_uint(prd.attenuation.y);
    u2 = __float_as_uint(prd.attenuation.z);
    u3 = prd.seed;
    u4 = prd.depth;

    // Note:
    // This demonstrates the usage of the OptiX shader execution reordering 
    // (SER) API.  In the case of this computationally simple shading code, 
    // there is no real performance benefit.  However, with more complex shaders
    // the potential performance gains offered by reordering are significant.
    optixTraverse(
        PAYLOAD_TYPE_RADIANCE,
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                     // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0,                        // SBT offset
        RAY_TYPE_COUNT,           // SBT stride
        0,                        // missSBTIndex
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17);
    optixReorder(
        // Application specific coherence hints could be passed in here
    );

    optixInvoke(PAYLOAD_TYPE_RADIANCE,
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17);

    prd.attenuation = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
    prd.seed = u3;
    prd.depth = u4;

    prd.emitted = make_float3(__uint_as_float(u5), __uint_as_float(u6), __uint_as_float(u7));
    prd.radiance = make_float3(__uint_as_float(u8), __uint_as_float(u9), __uint_as_float(u10));
    prd.origin = make_float3(__uint_as_float(u11), __uint_as_float(u12), __uint_as_float(u13));
    prd.direction = make_float3(__uint_as_float(u14), __uint_as_float(u15), __uint_as_float(u16));
    prd.done = u17;
}


// Returns true if ray is occluded, else false
static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax
)
{
    // We are only casting probe rays so no shader invocation is needed
    optixTraverse(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax, 0.0f,                // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,                         // SBT offset
        RAY_TYPE_COUNT,            // SBT stride
        0                          // missSBTIndex
    );
    return optixHitObjectIsHit();
}

static __forceinline__ __device__ void launchAgentRay(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    AgentPRD&              prd
)
{
    unsigned int u0, u1, u2, u3, u4, u5, u6, u7, u8;

	u0 = prd.rotation_count;
	u1 = prd.depth;

    optixTrace(
		PAYLOAD_TYPE_AGENT,
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax, 0.0f,                // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,                         // SBT offset
        RAY_TYPE_COUNT,            // SBT stride
        0,                          // missSBTIndex
        u0, u1, u2, u3, u4, u5, u6, u7, u8        // payloads
        );

        prd.origin = make_float3(__uint_as_float(u2), __uint_as_float(u3), __uint_as_float(u4));
        prd.direction = make_float3(__uint_as_float(u5), __uint_as_float(u6), __uint_as_float(u7));
        prd.done = u8;
}

// Returns distance to object if ray is occluded, else -1
static __forceinline__ __device__ float traceDistance(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax
)
{
    // We are only casting probe rays so no shader invocation is needed
    optixTraverse(
        PAYLOAD_TYPE_AGENT,
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax, 0.0f,                // rayTime
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,                         // SBT offset
        RAY_TYPE_COUNT,            // SBT stride
        0                          // missSBTIndex
    );

	return optixGetRayTmax(); // Return distance to hit point
}


//------------------------------------------------------------------------------
//
// Programs
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    const int    w = params.width;
    const int    h = params.height;
    const float3 eye = params.eye;
    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = params.W;
    const uint3  idx = optixGetLaunchIndex();
    const int    subframe_index = params.subframe_index;

    unsigned int seed = tea<4>(idx.y * w + idx.x, subframe_index);

    float3 result = make_float3(0.0f);
    int i = params.samples_per_launch;
    do
    {
        // The center of each pixel is at fraction (0.5,0.5)
        const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

        const float2 d = 2.0f * make_float2(
            (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(w),
            (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(h)
        ) - 1.0f;
        float3 ray_direction = normalize(d.x * U + d.y * V + W);
        float3 ray_origin = eye;

        RadiancePRD prd;
        prd.attenuation = make_float3(1.f);
        prd.seed = seed;
        prd.depth = 0;

        for (;; )
        {
            traceRadiance(
                params.handle,
                ray_origin,
                ray_direction,
                0.01f,  // tmin       // TODO: smarter offset
                1e16f,  // tmax
                prd);

            result += prd.emitted;
            result += prd.radiance * prd.attenuation;

            const float p = dot(prd.attenuation, make_float3(0.30f, 0.59f, 0.11f));
            const bool done = prd.done || rnd(prd.seed) > p;
            if (done)
                break;
            prd.attenuation /= p;

            ray_origin = prd.origin;
            ray_direction = prd.direction;

            ++prd.depth;
        }
    } while (--i);

    const uint3    launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.width + launch_index.x;
    float3         accum_color = result / static_cast<float>(params.samples_per_launch);

    if (subframe_index > 0)
    {
        const float                 a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    params.frame_buffer[image_index] = make_color(accum_color);

    //if (idx.x == 0 && idx.y == 0 && params.accumulate_weights)
    //{
    //    for (size_t i = 0; i < params.scene_triangle_count; i++)
    //    {
    //        printf("Triangle [%llu] : %.2f\n\n", i, params.d_triangle_weights[i]);
    //    }
    //}

}


extern "C" __global__ void __miss__radiance()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);

    MissData* rt_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    RadiancePRD prd = loadMissRadiancePRD();

    prd.radiance = make_float3(rt_data->bg_color);
    prd.emitted = make_float3(0.f);
    prd.done = true;

    storeMissRadiancePRD(prd);
}

// Compute a weight based on distance and radiance for a given hit.
__forceinline__ __device__ float computeWeight(float distance, float3 radiance)
{
    // Avoid division by zero or very small distances
    const float minDistance = 1e-3f;
    float safeDistance = fmaxf(distance, minDistance);

    // Compute luminance (perceptual brightness) from RGB radiance
    float luminance = 0.2126f * radiance.x + 0.7152f * radiance.y + 0.0722f * radiance.z;

    // Weight: higher for closer and brighter hits.
    float weight = abs(luminance / (safeDistance * safeDistance));

    // Clamp to avoid extreme values
    return fminf(weight, 1);
}

template<typename T>
__forceinline__ __device__ T sampleTexture(const MaterialData::Texture& texture, const float2 uv)
{
    // Assumes the caller checked that texture.object is valid.
    const float2 UV = uv * texture.texcoord_scale;
    const float2 rotation = texture.texcoord_rotation; // .x = sin, .y = cos
    const float2 UV_trans = make_float2(dot(make_float2(rotation.y, rotation.x), UV),
        dot(make_float2(-rotation.x, rotation.y), UV))
        + texture.texcoord_offset;

    return tex2D<T>(texture.tex, UV_trans.x, UV_trans.y);
}

__device__ float3 getBaseColor(const GeometryData::TriangleMesh& mesh, const MaterialData& material);
 
// Get TRIANGLES' base color.
// For the KHR_materials_unlit case, only the baseColor is required.
__forceinline__ __device__ float3 getBaseColor(const GeometryData::TriangleMesh& mesh,
    const MaterialData& material)
{
    const unsigned int prim_idx = optixGetPrimitiveIndex();

    // INDICES 
    uint3 tri;
    if (mesh.indices)
    {
        tri = make_uint3(mesh.positions[prim_idx]);
    }
    else
    {
        const unsigned int base_idx = prim_idx * 3;

        tri = make_uint3(base_idx, base_idx + 1, base_idx + 2);
    }

    const float2 barys = optixGetTriangleBarycentrics(); // .x = beta, .y = gamma
    const float  barys_a = 1.0f - barys.x - barys.y;       // alpha

    float3 baseColor = make_float3(material.pbr.base_color);

    // COLOR_0
    if (mesh.colors)
    {
        // If there are color attributes the alpha modulates the opacity.
        baseColor *= barys_a * make_float3(mesh.colors[tri.x]) +
            barys.x * make_float3(mesh.colors[tri.y]) +
            barys.y * make_float3(mesh.colors[tri.z]);
    }

    // If the baseColor has a texture the alpha in there modulates the opacity.
    if (material.pbr.base_color_tex)
    {
        // No need to calculate texture coordinates when there is no baseColorTexture 
        // and only one of the two supported texture coordinates is required here.
        // TEXCOORD_0 or TEXCOORD_1
        const int j = material.pbr.base_color_tex.texcoord;

        float2 texcoord = make_float2(0.0f);

        if (mesh.texcoords[j])
        {
            // Texture coordinate of the hit point.
            texcoord = barys_a * mesh.texcoords[j][tri.x] +
                barys.x * mesh.texcoords[j][tri.y] +
                barys.y * mesh.texcoords[j][tri.z];
        }

        baseColor *= make_float3(sampleTexture<float4>(material.pbr.base_color_tex, texcoord)); // sRGB
    }

    return baseColor;
}

extern "C" __global__ void __closesthit__radiance()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);

    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

    // Geometry and material data
    const GeometryData::TriangleMesh& mesh = rt_data->geometry_data.getTriangleMesh();
    const MaterialData& material = rt_data->material_data;

    const int    prim_idx = optixGetPrimitiveIndex();
    const float3 ray_dir = optixGetWorldRayDirection();
    const int    vert_idx_offset = prim_idx * 3;

    const float3 v0 = make_float3(mesh.indices[vert_idx_offset + 0]);
    const float3 v1 = make_float3(mesh.indices[vert_idx_offset + 1]);
    const float3 v2 = make_float3(mesh.indices[vert_idx_offset + 2]);
    const float3 N_0 = normalize(cross(v1 - v0, v2 - v0));


    const float3 N = faceforward(N_0, -ray_dir, N_0);
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    RadiancePRD prd = loadClosesthitRadiancePRD();

    if (prd.depth == 0)
        prd.emitted = getBaseColor(mesh, material);
    else
        prd.emitted = make_float3(0.0f);

    unsigned int seed = prd.seed;
    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in;
        cosine_sample_hemisphere(z1, z2, w_in);
        Onb onb(N);
        onb.inverse_transform(w_in);
        prd.direction = w_in;
        prd.origin = P;

        prd.attenuation *= material.emissive_factor;
    }

    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd.seed = seed;

    ParallelogramLight light = params.light;
    const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float  Ldist = length(light_pos - P);
    const float3 L = normalize(light_pos - P);
    const float  nDl = dot(N, L);
    const float  LnDl = -dot(light.normal, L);

    float weight = 0.0f;
    if (nDl > 0.0f && LnDl > 0.0f)
    {
        const bool occluded =
            traceOcclusion(
                params.handle,
                P,
                L,
                0.01f,           // tmin
                Ldist - 0.01f);  // tmax

        if (!occluded)
        {
            const float A = length(cross(light.v1, light.v2));
            weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
        }
    }

    prd.radiance = light.emission * weight;
    prd.done = false;

    if (params.accumulate_weights && params.agent.step_count < params.agent.max_steps)
    {
		// Accumulate weight for the triangle that was hit
        params.d_triangle_weights[prim_idx] += computeWeight(length(prd.origin - P), make_float3(1.f));

        // Increment hit count for the triangle
		params.d_triangle_hits_count[prim_idx] += 1u;
    }

    storeClosesthitRadiancePRD(prd);
}

//-----------------------------------------------------------------------------------
// Ray generation kernel for the agent simulation
//-----------------------------------------------------------------------------------
extern "C" __global__ void __raygen__agent()
{
    if (!params.agent_mode) return;      // Only run if agent mode is enabled

	AgentPRD prd;
	prd.rotation_count = 0u;  // Start with the first rotation (forward)
	prd.depth = 0;

    float3 dir = params.agent.direction;        // Current facing direction of the agent

    const int num_rays = 4;             // We want 4 rays to cover 360 (90 increments)
    float angles[num_rays] = { 0.f, 90.f, 180.f, 270.f }; // Relative rotation angles

    // Rotate the agent's forward direction around the Y-axis by angles[i]
    float rad = angles[0] * M_PIf / 180.0f;
    float3 rotated_dir = make_float3(
        cos(rad) * dir.x - sin(rad) * dir.z,
        dir.y,
        sin(rad) * dir.x + cos(rad) * dir.z
    );

    // Store the distance for this ray (Front)

    params.agent.d_rotations[0] = rotated_dir;  // Store the distance for this ray (Front)

    launchAgentRay(
        params.handle,
        params.agent.eye_position + make_float3(0.f, -1.f, 0.f),
        rotated_dir,
        0.01f,  // tmin
        1e16f,   // tmax
        prd
    );

    ++prd.rotation_count;

    rad = angles[1] * M_PIf / 180.0f;
    rotated_dir = make_float3(
        cos(rad) * dir.x - sin(rad) * dir.z,
        dir.y,
        sin(rad) * dir.x + cos(rad) * dir.z
    );

    params.agent.d_rotations[1] = rotated_dir;  // Store the distance for this ray (Right)

    launchAgentRay(
        params.handle,
        params.agent.eye_position + make_float3(0.f, -1.f, 0.f),
        rotated_dir,
        0.01f,  // tmin
        1e16f,   // tmax
        prd
    );

    ++prd.rotation_count;

    rad = angles[2] * M_PIf / 180.0f;
    rotated_dir = make_float3(
        cos(rad) * dir.x - sin(rad) * dir.z,
        dir.y,
        sin(rad) * dir.x + cos(rad) * dir.z
    );

    params.agent.d_rotations[2] = rotated_dir;  // Store the distance for this ray (Back)

    launchAgentRay(
        params.handle,
        params.agent.eye_position + make_float3(0.f, -1.f, 0.f),
        rotated_dir,
        0.01f,  // tmin
        1e16f,   // tmax
		prd
    );

    ++prd.rotation_count;

    rad = angles[3] * M_PIf / 180.0f;
    rotated_dir = make_float3(
        cos(rad) * dir.x - sin(rad) * dir.z,
        dir.y,
        sin(rad) * dir.x + cos(rad) * dir.z
    );

    params.agent.d_rotations[3] = rotated_dir;  // Store the distance for this ray (Left)

    launchAgentRay(
        params.handle,
        params.agent.eye_position + make_float3(0.f, -1.f, 0.f),
        rotated_dir,
        0.01f,  // tmin
        1e16f,   // tmax
        prd
    );
}

// Dummy miss program for the agent rays (does nothing)
extern "C" __global__ void __miss__agent()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_AGENT);
}

extern "C" __global__ void __closesthit__agent()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_AGENT);

	AgentPRD prd = loadClosesthitAgentPRD();

    switch (prd.rotation_count)
    {
    case 0u: // Forward rotation
        float distance_front = traceDistance(params.handle, params.agent.eye_position, params.agent.d_rotations[0], 0.01f, 1e16f);
        params.agent.d_distances[0] = distance_front;
        break;
    case 1u: // Right rotation
        float distance_right = traceDistance(params.handle, params.agent.eye_position, params.agent.d_rotations[1], 0.01f, 1e16f);
        params.agent.d_distances[1] = distance_right;
        break;
    case 2u: // Backward rotation
        float distance_back = traceDistance(params.handle, params.agent.eye_position, params.agent.d_rotations[2], 0.01f, 1e16f);
        params.agent.d_distances[2] = distance_back;
        break;
    case 3u: // Left rotation
        float distance_left = traceDistance(params.handle, params.agent.eye_position, params.agent.d_rotations[3], 0.01f, 1e16f);
        params.agent.d_distances[3] = distance_left;
        break;
    default:
        printf("Invalid rotation step!\n");
        break;
    }

	storeClosesthitAgentPRD(prd);
}
