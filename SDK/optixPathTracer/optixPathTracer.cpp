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

#include <glad/glad.h>  // Needs to be included before gl_interop

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
 // Optional. define TINYOBJLOADER_USE_MAPBOX_EARCUT gives robust triangulation. Requires C++11
 //#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tiny_obj_loader.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <optix_stack_size.h>

#include <GLFW/glfw3.h>

#include "optixPathTracer.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
 // Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tinygltf/tiny_gltf.h>
#include <sutil/Scene.h>

bool resize_dirty = false;
bool minimized = false;

// Camera state
bool             camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// Mouse state
int32_t mouse_button = -1;


int32_t samples_per_launch = 16;

bool weight_accumulation = false;
bool agent_mode = false;
bool agent_step_move = false;
bool agent_looping_move = false;

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;

struct PathTracerState
{
    OptixDeviceContext context = 0;

    OptixTraversableHandle         gas_handle = 0;  // Traversable handle for triangle AS
    CUdeviceptr                    d_gas_output_buffer = 0;  // Triangle AS memory
    CUdeviceptr                    d_vertices = 0;

    OptixModule                    ptx_module = 0;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    OptixPipeline                  pipeline = 0;
	OptixPipeline				   agent_pipeline = 0;

    OptixProgramGroup              raygen_prog_group = 0;
    OptixProgramGroup              radiance_miss_group = 0;
    OptixProgramGroup              radiance_hit_group = 0;
    OptixProgramGroup              agent_raygen_prog_group = 0;
    OptixProgramGroup              agent_hit_group = 0;
    OptixProgramGroup              agent_miss_group = 0;

    CUstream                       stream = 0;
    Params                         params;
    Params* d_params;

    OptixShaderBindingTable        sbt = {};
    OptixShaderBindingTable        sbtAgent = {};
};

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	// Disable mouse control in agent mode
    if (agent_mode)
    {
        return;
    }

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    // Disable mouse control in agent mode
    if (agent_mode)
    {
        return;
    }

    Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));

    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
        camera_changed = true;
    }
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
        camera_changed = true;
    }
}


static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    // Keep rendering at the current resolution when the window is minimized.
    if (minimized)
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize(res_x, res_y);

    Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));
    params->width = res_x;
    params->height = res_y;
    camera_changed = true;
    resize_dirty = true;
}


static void windowIconifyCallback(GLFWwindow* window, int32_t iconified)
{
    minimized = (iconified > 0);
}


static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(window, true);
        }
    }
    else if (key == GLFW_KEY_G)
    {
		// Toggle triange weight accumulation
        weight_accumulation = !weight_accumulation;
        std::cout << "Triangle weight accumulation " << (weight_accumulation ? "enabled" : "disabled") << std::endl;
    }
    else if (key == GLFW_KEY_N)
    {
        agent_looping_move = false;
        agent_step_move = true;
    }
    else if (key == GLFW_KEY_L)
    {
        agent_looping_move = true;
        agent_step_move = false;
    }
}


static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
    // Disable mouse control in agent mode
    if (agent_mode)
    {
        return;
    }

    if (trackball.wheelEvent((int)yscroll))
        camera_changed = true;
}


//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit(0);
}


void initLaunchParams(PathTracerState& state, const sutil::Scene& scene)
{
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.accum_buffer),
        state.params.width * state.params.height * sizeof(float4)
    ));

	// Allocate space for one float weight per triangle in the scene
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.d_triangle_weights),
        state.params.scene_triangle_count * sizeof(float)
    ));

    // Allocate space for one integer count per triangle in the scene
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.d_triangle_hits_count),
        state.params.scene_triangle_count * sizeof(unsigned int)
    ));

    CUDA_CHECK(cudaMallocHost(&state.params.h_triangle_weights, state.params.scene_triangle_count * sizeof(float)));

    CUDA_CHECK(cudaMallocHost(&state.params.h_triangle_hits_count, state.params.scene_triangle_count * sizeof(unsigned int)));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.agent.d_distances),
        4 * sizeof(float)
	));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.agent.d_rotations),
        4 * sizeof(float3)
    ));

    CUDA_CHECK(cudaMallocHost(&state.params.agent.h_distances, 4 * sizeof(float)));

    CUDA_CHECK(cudaMallocHost(&state.params.agent.h_rotations, 4 * sizeof(float3)));

    state.params.accumulate_weights = true;

    weight_accumulation = state.params.accumulate_weights;

    state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    state.params.samples_per_launch = samples_per_launch;
    state.params.subframe_index = 0u;

    state.params.light.emission = make_float3(15.0f, 15.0f, 5.0f);
    state.params.light.corner = make_float3(343.0f, 548.5f, 227.0f);
    state.params.light.v1 = make_float3(0.0f, 0.0f, 105.0f);
    state.params.light.v2 = make_float3(-130.0f, 0.0f, 0.0f);
    state.params.light.normal = normalize(cross(state.params.light.v1, state.params.light.v2));
    state.params.handle = scene.traversableHandle();

    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));
}


void handleCameraUpdate(Params& params)
{
    if (!camera_changed)
        return;
    camera_changed = false;

    camera.setAspectRatio(static_cast<float>(params.width) / static_cast<float>(params.height));
    params.eye = camera.eye();
    camera.UVWFrame(params.U, params.V, params.W);
}


void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params)
{
    if (!resize_dirty)
        return;
    resize_dirty = false;

    output_buffer.resize(params.width, params.height);

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        params.width * params.height * sizeof(float4)
    ));
}


void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params)
{
    // Update params on device
    if (camera_changed || resize_dirty)
        params.subframe_index = 0;

    handleCameraUpdate(params);
    handleResize(output_buffer, params);
}


void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, PathTracerState& state)
{
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(state.d_params),
        &state.params, sizeof(Params),
        cudaMemcpyHostToDevice, state.stream
    ));

    OPTIX_CHECK(optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast<CUdeviceptr>(state.d_params),
        sizeof(Params),
        &state.sbt,
        state.params.width,   // launch width
        state.params.height,  // launch height
        1                     // launch depth
    ));
    output_buffer.unmap();
    CUDA_SYNC_CHECK();

    OPTIX_CHECK(optixLaunch(
        state.agent_pipeline,
        state.stream,
        reinterpret_cast<CUdeviceptr>(state.d_params),
        sizeof(Params),
        &state.sbtAgent,
        1,   // launch width
        1,  // launch height
        1                     // launch depth
    ));
    CUDA_SYNC_CHECK();
}


void displaySubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window)
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}


static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}


void initCameraState(const sutil::Scene& scene, const PathTracerState& state)
{
    camera = scene.camera();
	camera.setEye(state.params.agent.eye_position);
	//camera.setDirection(state.params.agent.direction);
    camera_changed = true;

    trackball.setCamera(&camera);
    trackball.setViewMode(sutil::Trackball::EyeFixed);
    trackball.setMoveSpeed(0.0f);
    trackball.setReferenceFrame(
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 1.0f, 0.0f)
    );
    trackball.setGimbalLock(true);
}


void updateCameraState(const PathTracerState& state)
{
    if (state.params.agent_mode)
    {
        camera.setEye(state.params.agent.eye_position);
		camera.setDirection(state.params.agent.direction);
        camera_changed = true;
    }
}

void initAgent(PathTracerState& state)
{
    // Place the agent in the center of the scene
    state.params.agent.position = make_float3(15.0f, 0.0f, 10.0f);

    // Place the camera slightly above the agent's position
    state.params.agent.eye_position = state.params.agent.position + make_float3(0.0f, 1.5f, 0.0f);

    // Have the agent look straight ahead initially
    state.params.agent.direction = normalize(make_float3(-1.0f, 0.0f, 0.0f));

    state.params.agent.step_count = 0;
    state.params.agent.max_steps = 1000;
    state.params.agent.march_distance = 0.80f; // 80% of the way to the ray target each step
    state.params.agent.fov = 90.0f;
    state.params.agent_mode = true;
    state.params.agent.find_new_direction = true;
	state.params.agent.obstacle_threshold_distance = 5.f; // Minimum distance to an obstacle before changing direction
    state.params.agent.previous_maximum_distance_index = 0;
}


//-----------------------------------------------------------------------------------
// Move the agent forward and randomly rotate it a bit
// 1. Find new maximum distance direction.
// 2. Keep moving towards the maximum distance direction until a threshold is reached.
// 3. When the threshold is reached, rotate 180 degrees with a -/+ 90 degree deviation and
//    find a new maximum distance direction.
// 4. Repeat steps 1-3 until max steps reached.
//-----------------------------------------------------------------------------------
void agentMoveStep(Agent& agent)
{
    // Check if the minimum distance threshold has been reached
    if (agent.find_new_direction)
    {
        float maximum_distance = agent.h_distances[0]; // Maximum distance to check for obstacles
        float3 rotation_angle = agent.h_rotations[0]; // Rotation angle corresponding to the maximum distance
        agent.maximum_distance_index = 0;
        agent.previous_maximum_distance_index = agent.maximum_distance_index;

        for (int i = 1; i < 4; i++)
        {
            if (agent.h_distances[i] > maximum_distance) // Check if this is the new maximum
            {
                maximum_distance = agent.h_distances[i];
                rotation_angle = agent.h_rotations[i];
                agent.maximum_distance_index = i;
            }
        }

		agent.find_new_direction = false;
    }


    if ( (agent.h_distances[agent.maximum_distance_index] < agent.obstacle_threshold_distance) || (agent.h_distances[agent.previous_maximum_distance_index] < agent.obstacle_threshold_distance))
    {
        // Ensure the rotation is only around the Y-axis and is normalized
        float3 new_rotation_angle = normalize(make_float3(agent.direction.x, 0.f, agent.direction.z));

        // Generate a random fraction of 30� rotation
        unsigned int seed = agent.step_count;
        srand(seed);

        // Rotate the agent by 180 degrees
        new_rotation_angle = normalize(make_float3(-new_rotation_angle.x, 0.f, -new_rotation_angle.z));

        // Add a andom rotation angle between -90 and 90 degrees ( rand range: [0, 1) )
        float random_angle = (rand() / (float)RAND_MAX - 0.5f) * 180.0f;
        float rad = random_angle * M_PIf / 180.0f;    // Convert to radians

        // Rotate the current direction by the random angle around Y-axis
        float3 new_dir = normalize(make_float3(
            cos(rad) * new_rotation_angle.x - sin(rad) * new_rotation_angle.z,
            new_rotation_angle.y,
            sin(rad) * new_rotation_angle.x + cos(rad) * new_rotation_angle.z
        ));

        // Update the agent's facing direction (normalized)
        agent.direction = normalize(make_float3(new_dir.x, 0.f, new_dir.z));

        agent.find_new_direction = true;

        return;
    }

    // Move the agent along the XZ plane (keep Y constant) to march forward
    agent.position += agent.direction * (agent.h_distances[agent.maximum_distance_index] * agent.march_distance);
}


void updateAgentState(PathTracerState& state)
{
    if (state.params.agent_mode && state.params.agent.step_count < state.params.agent.max_steps && (agent_step_move || agent_looping_move))
    {
		agentMoveStep(state.params.agent);
        state.params.agent.eye_position += state.params.agent.direction * state.params.agent.march_distance;
        state.params.agent.step_count++;
        camera_changed = true;
    }

    agent_step_move = false;
}

void writeWeightsToFile(const float* triangle_weights, const unsigned int* triangle_hit_counts, const unsigned int scene_triangle_count, const std::string& filepath)
{
    std::ofstream outFile(filepath);

    if (!outFile.is_open())
    {
        std::cerr << "Error: Could not open file " << filepath << " for writing." << std::endl;
        return;
    }

    outFile << std::fixed << std::setprecision(5);

    // Calculate normalized values and write to file
    for (size_t i = 0; i < scene_triangle_count; i++)
    {
        float normalizedValue = 0.0f;

        // Avoid division by zero
        if (triangle_hit_counts[i] != 0)
        {
            normalizedValue = triangle_weights[i] / static_cast<float>(triangle_hit_counts[i]);
        }

        outFile << normalizedValue << std::endl;
    }

    outFile.close();
}


void createContext(PathTracerState& state)
{
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
#ifdef DEBUG
    // This may incur significant performance cost and should only be done during development.
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

    state.context = context;
}

void createModule(PathTracerState& state)
{
    OptixPayloadType payloadType[2] = {};

    // radiance prd
    payloadType[0].numPayloadValues = sizeof(radiancePayloadSemantics) / sizeof(radiancePayloadSemantics[0]);
    payloadType[0].payloadSemantics = radiancePayloadSemantics;

    // agent prd
    payloadType[1].numPayloadValues = sizeof(agentPayloadSemantics) / sizeof(agentPayloadSemantics[0]);
    payloadType[1].payloadSemantics = agentPayloadSemantics;

    OptixModuleCompileOptions module_compile_options = {};
#if OPTIX_DEBUG_DEVICE_CODE
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
    module_compile_options.numPayloadTypes = 2;
    module_compile_options.payloadTypes = payloadType;

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 0;
    state.pipeline_compile_options.numAttributeValues = 2;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t      inputSize = 0;
    const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixPathTracer.cu", inputSize);

    OPTIX_CHECK_LOG(optixModuleCreate(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input,
        inputSize,
        LOG, &LOG_SIZE,
        &state.ptx_module
    ));
}


void createProgramGroups(PathTracerState& state)
{
    OptixProgramGroupOptions  program_group_options = {};

    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = state.ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.raygen_prog_group
        ));
    }

    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = state.ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.radiance_miss_group
        ));
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &hit_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.radiance_hit_group
        ));
    }

    OptixProgramGroupOptions  agent_program_group_options = {};

    {
        OptixProgramGroupDesc agent_raygen_prog_group_desc = {};
        agent_raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        agent_raygen_prog_group_desc.raygen.module = state.ptx_module;
        agent_raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__agent";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &agent_raygen_prog_group_desc,
            1,  // num program groups
            &agent_program_group_options,
            LOG, &LOG_SIZE,
            &state.agent_raygen_prog_group
        ));
    }

    {
        OptixProgramGroupDesc agent_miss_prog_group_desc = {};
        agent_miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        agent_miss_prog_group_desc.miss.module = state.ptx_module;
        agent_miss_prog_group_desc.miss.entryFunctionName = "__miss__agent";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &agent_miss_prog_group_desc,
            1,  // num program groups
            &agent_program_group_options,
            LOG, &LOG_SIZE,
            &state.agent_miss_group
        ));
    }

    {
        OptixProgramGroupDesc agent_hit_prog_group_desc = {};
        agent_hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        agent_hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
        agent_hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__agent";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &agent_hit_prog_group_desc,
            1,  // num program groups
            &agent_program_group_options,
            LOG, &LOG_SIZE,
			&state.agent_hit_group
        ));
    }
}


void createPipeline(PathTracerState& state)
{
    OptixProgramGroup program_groups[] =
    {
        state.raygen_prog_group,
        state.radiance_miss_group,
        state.radiance_hit_group,

		// Agent programs
        state.agent_raygen_prog_group,
        state.agent_miss_group,
        state.agent_hit_group,
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;

    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        LOG, &LOG_SIZE,
        &state.pipeline
    ));

    OptixProgramGroup agent_program_groups[] =
    {
        // Agent programs
        state.agent_raygen_prog_group,
        state.agent_miss_group,
        state.agent_hit_group,
    };

    OptixPipelineLinkOptions agent_pipeline_link_options = {};
    agent_pipeline_link_options.maxTraceDepth = 2;

    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &agent_pipeline_link_options,
        agent_program_groups,
        sizeof(agent_program_groups) / sizeof(agent_program_groups[0]),
        LOG, &LOG_SIZE,
        &state.agent_pipeline
    ));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.raygen_prog_group, &stack_sizes, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_miss_group, &stack_sizes, state.pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_hit_group, &stack_sizes, state.pipeline));

	// Agent pipeline stack sizes
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.agent_raygen_prog_group, &stack_sizes, state.agent_pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.agent_miss_group, &stack_sizes, state.agent_pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.agent_hit_group, &stack_sizes, state.agent_pipeline));


    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));

    const uint32_t max_traversal_depth = 1;
    OPTIX_CHECK(optixPipelineSetStackSize(
        state.pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth
    ));

    OPTIX_CHECK(optixPipelineSetStackSize(
        state.agent_pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth
    ));
}


void createSBT(PathTracerState& state, const sutil::Scene& scene)
{
    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr  d_agent_raygen_record;
    const size_t agent_raygen_record_size = sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_agent_raygen_record), agent_raygen_record_size));

    RayGenRecord agent_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.agent_raygen_prog_group, &agent_sbt));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_agent_raygen_record),
        &agent_sbt,
        agent_raygen_record_size,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr  d_miss_records;
    const size_t miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), miss_record_size * RAY_TYPE_COUNT));

    MissRecord ms_sbt[1];
    OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_miss_group, &ms_sbt[0]));
    ms_sbt[0].data.bg_color = make_float4(0.0f);

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_miss_records),
        ms_sbt,
        miss_record_size * RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice
    ));

	// Agent miss record
    CUdeviceptr  d_agent_miss_records;
    const size_t agent_miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_agent_miss_records), agent_miss_record_size * RAY_TYPE_COUNT));

    MissRecord agent_ms_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.agent_miss_group, &agent_ms_sbt));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_agent_miss_records),
        &agent_ms_sbt,
        agent_miss_record_size * RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice
    ));

	CUdeviceptr  d_hitgroup_records;
    CUdeviceptr  d_agent_hitgroup_records;

    state.params.scene_triangle_count = 0u;
    
	// Create hitgroup records for each mesh in the scene
    std::vector<HitGroupRecord> hitgroup_records;
    std::vector<HitGroupRecord> agent_hitgroup_records;

	// Debug: Store triangle offsets for each mesh part
	std::vector<unsigned int> triangle_offsets;

    unsigned int global_triangle_offset = 0u;

    for (const auto& instance : scene.instances())
    {
        const auto mesh = scene.meshes()[instance->mesh_idx];
        for (size_t i = 0; i < mesh->material_idx.size(); ++i)
        {
            HitGroupRecord rec;
			HitGroupRecord rec_agent;
            
			// Set the SBT record headers
            OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_hit_group, &rec));
            OPTIX_CHECK(optixSbtRecordPackHeader(state.agent_hit_group, &rec_agent));

            GeometryData::TriangleMesh triangle_mesh = {};
            triangle_mesh.normals = mesh->normals[i];
            triangle_mesh.positions = mesh->positions[i];
            for (size_t j = 0; j < GeometryData::num_texcoords; ++j)
                triangle_mesh.texcoords[j] = mesh->texcoords[j][i];
            triangle_mesh.colors = mesh->colors[i];
            triangle_mesh.indices = mesh->indices[i];

            rec.data.geometry_data.setTriangleMesh(triangle_mesh);
            rec.data.material_data = scene.materials()[mesh->material_idx[i]];
            rec.data.triangle_offset = global_triangle_offset;

            rec_agent.data.geometry_data.setTriangleMesh(triangle_mesh);
            rec_agent.data.material_data = scene.materials()[mesh->material_idx[i]];
			rec_agent.data.triangle_offset = global_triangle_offset;

            // Count mesh triangles and add it to the total number of triangles
            unsigned int current_primitive_triangles = mesh->indices[i].count / 3;
            state.params.scene_triangle_count += current_primitive_triangles;

            // Store the offset for debugging
            triangle_offsets.push_back(global_triangle_offset);

			global_triangle_offset += current_primitive_triangles;

            hitgroup_records.push_back(rec);
			agent_hitgroup_records.push_back(rec_agent);
        }
    }

	std::cout << "Total number of triangles in the scene: " << state.params.scene_triangle_count << std::endl;

    std::cout << "Number of mesh parts: " << hitgroup_records.size() << std::endl;

    // Debug: Print triangle distribution
    for (size_t i = 0; i < triangle_offsets.size(); ++i)
    {
        std::cout << "Mesh part " << i+1 << " starts at triangle " << triangle_offsets[i] << std::endl;
    }

    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_hitgroup_records),
        hitgroup_record_size * RAY_TYPE_COUNT * hitgroup_records.size()
    ));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hitgroup_records),
        hitgroup_records.data(),
        hitgroup_record_size * RAY_TYPE_COUNT * hitgroup_records.size(),
        cudaMemcpyHostToDevice
    ));

    state.sbt.raygenRecord = d_raygen_record;
    state.sbt.missRecordBase = d_miss_records;
    state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    state.sbt.missRecordCount = RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(hitgroup_record_size);
    state.sbt.hitgroupRecordCount = static_cast<unsigned int> (hitgroup_records.size());
    
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_agent_hitgroup_records),
        hitgroup_record_size * RAY_TYPE_COUNT * agent_hitgroup_records.size()
    ));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_agent_hitgroup_records),
        agent_hitgroup_records.data(),
        hitgroup_record_size * RAY_TYPE_COUNT * agent_hitgroup_records.size(),
        cudaMemcpyHostToDevice
    ));

	// Agent SBT
    state.sbtAgent.raygenRecord = d_agent_raygen_record;
    state.sbtAgent.missRecordBase = d_agent_miss_records;
    state.sbtAgent.missRecordStrideInBytes = static_cast<uint32_t>(agent_miss_record_size);
    state.sbtAgent.missRecordCount = RAY_TYPE_COUNT;
    state.sbtAgent.hitgroupRecordBase = d_agent_hitgroup_records;
    state.sbtAgent.hitgroupRecordStrideInBytes = static_cast<unsigned int>(hitgroup_record_size);
    state.sbtAgent.hitgroupRecordCount = static_cast<unsigned int> (agent_hitgroup_records.size());
}


void displayAgentDistances(const float* agent_distances)
{
    constexpr std::chrono::duration<double> display_update_min_interval_time(0.1);
    static int32_t                          total_subframe_count = 0;
    static int32_t                          last_update_frames = 0;
    static auto                             last_update_time = std::chrono::steady_clock::now();
    static char                             display_text[128];

    const auto cur_time = std::chrono::steady_clock::now();

    sutil::beginFrameImGui();
    last_update_frames++;

    typedef std::chrono::duration<double, std::milli> durationMs;

    if (cur_time - last_update_time > display_update_min_interval_time || total_subframe_count == 0)
    {
        sprintf(display_text,
            "-- Agent Distances --\n"
            "Forward: %.5f\n"
            "Right: %.5f\n"
            "Back: %.5f\n"
            "Left: %.5f\n",
            agent_distances[0],
            agent_distances[1],
            agent_distances[2],
            agent_distances[3]
        );

        last_update_time = cur_time;
        last_update_frames = 0;
    }
    sutil::displayText(display_text, 10.0f, 100.0f);
    sutil::endFrameImGui();

    ++total_subframe_count;
}

void cleanupState(PathTracerState& state)
{
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixPipelineDestroy(state.agent_pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_miss_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_hit_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.agent_raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.agent_miss_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.agent_hit_group));
    OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));


    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbtAgent.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbtAgent.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbtAgent.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_params)));
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    PathTracerState state;
    state.params.width = 768;
    state.params.height = 768;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    //
    // Parse command line options
    //
    std::string model_filepath = "UnrealLevel/level_enclosed_v2/";
	std::string model_name = "level_enclosed_v2";
    std::string model_extension = ".gltf";
    std::string filename = model_filepath + model_name + model_extension;

    std::string outfile;
    std::string infile = sutil::sampleDataFilePath(filename.c_str());

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--no-gl-interop")
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if (arg == "--file" || arg == "-f")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            outfile = argv[++i];
        }
        else if (arg.substr(0, 6) == "--dim=")
        {
            const std::string dims_arg = arg.substr(6);
            int w, h;
            sutil::parseDimensions(dims_arg.c_str(), w, h);
            state.params.width = w;
            state.params.height = h;
        }
        else if (arg == "--launch-samples" || arg == "-s")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            samples_per_launch = atoi(argv[++i]);
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    try
    {
        sutil::Scene scene;
        sutil::loadScene(infile, scene);
		//loadObjScene(infile, scene);
        scene.finalize(false, RAY_TYPE_COUNT);

        OPTIX_CHECK(optixInit()); // Need to initialize function table
        initAgent(state);
        initCameraState(scene, state);

		// Store agent mode in params to determine if mouse control is enabled
        agent_mode = state.params.agent_mode;

        //
        // Set up OptiX state
        //
        createContext(state);
        createModule(state);
        createProgramGroups(state);
        createPipeline(state);
        createSBT(state, scene);
        initLaunchParams(state, scene);


        if (outfile.empty())
        {
            GLFWwindow* window = sutil::initUI("optixPathTracer", state.params.width, state.params.height);
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorPosCallback);
            glfwSetWindowSizeCallback(window, windowSizeCallback);
            glfwSetWindowIconifyCallback(window, windowIconifyCallback);
            glfwSetKeyCallback(window, keyCallback);
            glfwSetScrollCallback(window, scrollCallback);
            glfwSetWindowUserPointer(window, &state.params);

            //
            // Render loop
            //
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                );

                output_buffer.setStream(state.stream);
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time(0.0);
                std::chrono::duration<double> render_time(0.0);
                std::chrono::duration<double> display_time(0.0);

                do
                {
                    if (weight_accumulation)
                        state.params.accumulate_weights = true;
                    else
						state.params.accumulate_weights = false;

                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState(output_buffer, state.params);
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe(output_buffer, state);
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe(output_buffer, gl_display, window);
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats(state_update_time, render_time, display_time);

                    displayAgentDistances(state.params.agent.h_distances);

                    glfwSwapBuffers(window);

                    CUDA_CHECK(cudaMemcpyAsync(
                        reinterpret_cast<void*>(state.params.agent.h_distances),
                        reinterpret_cast<void*>(state.params.agent.d_distances),
                        4 * sizeof(float),
                        cudaMemcpyDeviceToHost
					));
                    CUDA_SYNC_CHECK();

                    CUDA_CHECK(cudaMemcpyAsync(
                        reinterpret_cast<void*>(state.params.agent.h_rotations),
                        reinterpret_cast<void*>(state.params.agent.d_rotations),
                        4 * sizeof(float3),
                        cudaMemcpyDeviceToHost
                    ));
					CUDA_SYNC_CHECK();

					updateAgentState(state);
					updateCameraState(state);

                    if (state.params.agent.step_count >= state.params.agent.max_steps)
                    {
                        std::cout << "Agent reached maximum step count of " << state.params.agent.max_steps << ". Stopping movement." << std::endl;
                        state.params.agent_mode = false;

                        // Memcpy for triangle weights
                        CUDA_CHECK(cudaMemcpyAsync(
                            reinterpret_cast<void*>(state.params.h_triangle_weights),
                            reinterpret_cast<void*>(state.params.d_triangle_weights),
                            state.params.scene_triangle_count * sizeof(float),
                            cudaMemcpyDeviceToHost
                        ));
                        CUDA_SYNC_CHECK();

                        // Memcpy for hit counts
                        CUDA_CHECK(cudaMemcpyAsync(
                            reinterpret_cast<void*>(state.params.h_triangle_hits_count),
                            reinterpret_cast<void*>(state.params.d_triangle_hits_count),
                            state.params.scene_triangle_count * sizeof(unsigned int),
                            cudaMemcpyDeviceToHost
                        ));
                        CUDA_SYNC_CHECK();

                        // Create buffer with triangle weights and hits
                        // and write to .txt file
                        std::string output_filename = std::string(getenv("USERPROFILE")) + "\\Desktop\\" + model_name + "_weights.txt";
                        writeWeightsToFile(state.params.h_triangle_weights, state.params.h_triangle_hits_count, state.params.scene_triangle_count, output_filename);

                        std::cout << "Successfully written out to '" << output_filename << "'." << std::endl << "Exiting...\n" << std::endl;

                        return 0;
                    }

                    ++state.params.subframe_index;
                } while (!glfwWindowShouldClose(window));
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI(window);
        }
        else
        {
            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                sutil::initGLFW();  // For GL context
                sutil::initGL();
            }

            {
                // this scope is for output_buffer, to ensure the destructor is called bfore glfwTerminate()

                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                );

                handleCameraUpdate(state.params);
                handleResize(output_buffer, state.params);
                launchSubframe(output_buffer, state);

                sutil::ImageBuffer buffer;
                buffer.data = output_buffer.getHostPointer();
                buffer.width = output_buffer.width();
                buffer.height = output_buffer.height();
                buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

                sutil::saveImage(outfile.c_str(), buffer, false);
            }

            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                glfwTerminate();
            }
        }

        cleanupState(state);
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}