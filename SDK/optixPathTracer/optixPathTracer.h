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
#include <vector_types.h>
#include <cuda/GeometryData.h>
#include <cuda/MaterialData.h>
#include <vector>

constexpr unsigned int RAY_TYPE_COUNT = 1;

constexpr OptixPayloadTypeID PAYLOAD_TYPE_RADIANCE = OPTIX_PAYLOAD_TYPE_ID_0;
constexpr OptixPayloadTypeID PAYLOAD_TYPE_AGENT = OPTIX_PAYLOAD_TYPE_ID_1;

struct RadiancePRD
{
    // these are produced by the caller, passed into trace, consumed/modified by CH and MS and consumed again by the caller after trace returned.
    float3       attenuation;
    unsigned int seed;
    int          depth;

    // these are produced by CH and MS, and consumed by the caller after trace returned.
    float3       emitted;
    float3       radiance;
    float3       origin;
    float3       direction;
    float        distance;
    int          done;
};

struct AgentPRD
{
    // these are produced by the caller, passed into trace, consumed/modified by CH and MS and consumed again by the caller after trace returned.
	unsigned int rotation_count;
    int          depth;

    // these are produced by CH and MS, and consumed by the caller after trace returned.
    float3       origin;
    float3       direction;
    float        distance;
    int          done;
};


const unsigned int radiancePayloadSemantics[18] =
{
    // RadiancePRD::attenuation
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::seed
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::depth
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::emitted
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    // RadiancePRD::radiance
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    // RadiancePRD::origin
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    // RadiancePRD::direction
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    // RadiancePRD::done
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE
};

const unsigned int agentPayloadSemantics[9] =
{
	//AgentPRD::rotation_count
	OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // AgentPRD::depth
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // AgentPRD::origin
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    // AgentPRD::direction
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    // AgentPRD::done
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE
};


struct ParallelogramLight
{
    float3 corner;
    float3 v1, v2;
    float3 normal;
    float3 emission;
};

struct Agent
{
    float3 position;
    float3 direction;

    float3 eye_position;
    float3       U;
    float3       V;
    float3       W;

    // (Device) distances to the nearest obstacle in each of 4 directions (Front, Right, Back, Left)
	float* d_distances;

    // (Host) distances to the nearest obstacle in each of 4 directions (Front, Right, Back, Left)
    float* h_distances;

    // (Device) relative rotation angles to turn away from obstacles in each of 4 directions (Front, Right, Back, Left)
	float3* d_rotations;

    // (Host) relative rotation angles to turn away from obstacles in each of 4 directions (Front, Right, Back, Left)
    float3* h_rotations;

    // Store agent's positions throughout the program's runtime
    float3* agent_positions;

    float3* agent_directions;

    float fov;
    unsigned int max_steps;
    unsigned int step_count;
    float march_distance;
    float turn_angle;

	int maximum_distance_index; // Index of the direction with the maximum distance
    int previous_maximum_distance_index;
    bool find_new_direction;
	bool first_launch;
	float obstacle_threshold_distance;
};

struct Params
{
    unsigned int subframe_index;
    float4* accum_buffer;
    uchar4* frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;
    unsigned long scene_triangle_count;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;
    
	float* d_triangle_weights; // (Device) one float weight per triangle in the scene   
	
    unsigned long* d_triangle_not_occluded_hits_count; // (Device) Hits that were not occluded - one int count per triangle in the scene
    unsigned long* d_triangle_total_hits_count; // (Device) Total triangle hits (occluded or not) - one int count per triangle in the scene

    float* h_triangle_weights; // (Host) one float weight per triangle in the scene   
    
    unsigned long* h_triangle_not_occluded_hits_count; // (Host) Hits that were not occluded - one int count per triangle in the scene
    unsigned long* h_triangle_total_hits_count; // (Host) Hits that were not occluded - one int count per triangle in the scene

	// Total rays launched during the entire rendering process
	unsigned long total_rays_launched;

    ParallelogramLight     light; // TODO: make light list
    OptixTraversableHandle handle;
    Agent agent;
    bool agent_mode;
    bool accumulate_weights;
};


struct RayGenData
{
};


struct MissData
{
    float4 bg_color;
};


struct HitGroupData
{
    // Data for building SBT
    GeometryData geometry_data;
    MaterialData material_data;
	unsigned int triangle_offset;

};
