# Mesh Decimation via Weighted Importance Mapping — OptiX Module

**Undergraduate Thesis | Athens University of Economics and Business**  
Department of Informatics, School of Information Sciences & Technology  
C. Chrysafi · A. A. Manetas · Athens, 2025

📄 **[Read the full thesis](doc/Mesh%20Decimation%20via%20Weighted%20Importance%20Mapping.pdf)**

---

> **This is the OptiX (visibility analysis) half of the pipeline.**  
> The companion simplification repository lives at [MeshDecimation-libigl](https://github.com/Heart-lex/MeshDecimation-libigl).

---

## Overview

Traditional mesh decimation treats all regions of a 3D model equally, often destroying visually important features while preserving unseen geometry. This project introduces a **visibility-driven importance mapping** approach that uses GPU-accelerated ray tracing to identify which parts of a mesh matter perceptually, then feeds that data into a weighted edge-collapse decimation algorithm.

This repository implements **Stages 1–2** of the pipeline: ray-based visibility analysis using NVIDIA OptiX, producing a per-triangle importance map that drives the downstream simplification.

---

## Pipeline

![Pipeline overview](docs/images/pipeline_diagram.png)

The full pipeline goes from scene creation to optimized output via two stages:

| Stage | Tool | Input | Output |
|---|---|---|---|
| Data Preparation | UE5 / Blender | Scene | glTF + OBJ |
| **Visibility Analysis** | **OptiX (GPU)** | **glTF** | **Per-triangle weight `.txt`** |
| Importance Map | OptiX (CPU) | Raw hits | Normalized weight file |
| Weighted Simplification | libigl (CPU) | OBJ + weights | Simplified LOD meshes |

---

## What This Module Does

### 1. Ray Generation

Rays are cast from a virtual camera or a simulated agent traversing the scene. Each ray originates from a pixel on a virtual image plane with stochastic subpixel jitter to reduce aliasing:

```
for each pixel:
    ray = generate_ray(pixel_coords + rnd(seed))
    trace(ray, RadiancePRD)
```

### 2. Intersection Recording

Only the **first** intersection per ray is recorded — mirroring how occlusion works in human vision. Each hit is mapped to a global triangle index shared across all meshes in the scene:

```
global_triangle_id = primitive_id + geometry_offset
```

### 3. Perceptual Weight Computation

Each intersection accumulates a weight derived from two biologically inspired components.

**Distance Weighting** — logarithmic inverse-square falloff emphasising nearby geometry:

$$w_d = \frac{1}{\log_{10}(1 + d^2)}$$

**Foveated Weighting** — models the drop in human visual acuity from the foveal centre outward, using cosine similarity between the camera forward direction and the ray direction:

| Region | Angle | Weight |
|---|---|---|
| Foveal | ~5° | 1.0 |
| Parafoveal | ~20° | 0.51 – 0.99 |
| Peripheral | ~45° | 0.1 – 0.5 |

The final accumulated weight per intersection:

$$w_i = \log(1 + w_d \cdot w_f)$$

### 4. Agent-Based Visibility Sampling

A simulated agent traverses the scene autonomously, emulating realistic player movement or pre-recorded camera paths. At each step, four rays are cast in cardinal directions (0°, 90°, 180°, 270°). The agent moves toward the direction of longest unobstructed distance, turning away from blockages when a proximity threshold is met.

Early prototype sponza scene explored by the agent during visibility sampling:

![Final prototype scene](docs/images/final_prototype_ue5.jpeg)

### 5. Importance Map Construction and Export

After the ray-tracing pass, triangle weights are copied from GPU to CPU, **standardized** (to prevent extreme values from dominating), then **normalized** to `[0, 1]`:

- **1.0** → most frequently observed triangle  
- **0.0** → never visible / fully occluded

The result is visualized as a heatmap — warm colours (yellow/green) indicate high importance, cool colours (blue) indicate low importance — and exported as a plain `.txt` file for the libigl module.

First heatmap run in libigl on a prototype scene (statues and benches):

![Importance heatmap – libigl](docs/images/heatmap_libigl_scene.jpeg)

An early weight map on UE5's Classic Sponza (this scene was ultimately discarded due to non-manifold geometry):

![Sponza weight map](docs/images/sponza_weightmap.jpeg)

Importance heatmap overlaid on the apartment scene:

![Importance heatmap – apartment](docs/images/heatmap_apartment.jpeg)

---

## Results

The pipeline was evaluated on a custom apartment scene (112,591 triangles). The importance-weighted approach produced visually indistinguishable results from the agent's camera positions at significant polygon reductions.

### Triangle Count Progression

| Reduction | Triangles | Visual Impact (agent POV) |
|---|---|---|
| 0% (original) | 112,591 | Baseline |
| 50% | 56,261 | Negligible — major surfaces intact |
| 85% | 18,765 | Minimal — only silhouette edges affected |

### Original (0% reduction, 112,591 triangles)

![Original mesh – solid](docs/images/original_mesh_blender.jpeg)
![Original mesh – wireframe](docs/images/original_mesh_wireframe.jpeg)

### 50% Reduction (56,261 triangles)

![Decimated 50% – solid](docs/images/decimated_50_solid.jpeg)
![Decimated 50% – wireframe](docs/images/decimated_50_wireframe.jpeg)

### 85% Reduction (18,765 triangles)

At 85% reduction the algorithm's visibility-awareness becomes clearly visible: the elephant toy in the lower room (never seen by the agent) is almost entirely removed, the lower body of the large camel (occluded by a wall) is heavily simplified, while the upper/visible portions remain intact.

![Decimated 85% – solid](docs/images/decimated_85_solid.jpeg)
![Decimated 85% – wireframe](docs/images/decimated_85_wireframe.jpeg)

### NVIDIA FLIP Comparisons

[NVIDIA FLIP](https://github.com/NVlabs/flip) measures perceptual difference between the original and decimated renders from the agent's exact point of view. Bright pixels = visible change; dark = unchanged.

**From the agent's POV (hallway)** — original vs. decimated:

| Original | Decimated | FLIP Difference |
|---|---|---|
| ![Hallway original](docs/images/hallway_original.jpeg) | ![Hallway decimated](docs/images/hallway_decimated.jpeg) | ![FLIP hallway](docs/images/flip_hallway.jpeg) |

The difference is confined to silhouette edges — imperceptible to the naked eye.

**From an angle the agent never saw** — the camel's unseen front face is aggressively decimated while the rear (which the agent did observe) is preserved:

| Original | Decimated |
|---|---|
| ![Camel unseen – original](docs/images/camel_unseen_original.jpeg) | ![Camel unseen – decimated](docs/images/camel_unseen_decimated.jpeg) |

**Occluded room** — the camel's lower body (hidden behind a wall from the agent) is removed entirely, while the head and hump (visible over the wall) remain:

| Original | Decimated |
|---|---|
| ![Obstructed room – original](docs/images/room_obstructed_original.jpeg) | ![Obstructed room – decimated](docs/images/room_obstructed_decimated.jpeg) |

**Toy room** — the occluded elephant and the rear of the cow are decimated; the cow's visible front face is preserved:

| Original | Decimated |
|---|---|
| ![Toy room – original](docs/images/toy_room_original.jpeg) | ![Toy room – decimated](docs/images/toy_room_decimated.jpeg) |

**Final decimated scene overview:**

![Final decimated scene](docs/images/final_decimated_scene.jpeg)

---

## Requirements

### Hardware
- **NVIDIA RTX-capable GPU** with CUDA Compute Capability 5.0+

### Software
| Dependency | Version |
|---|---|
| NVIDIA OptiX SDK | 9.0.0 |
| CUDA Toolkit | 11.0+ |
| CMake | 3.10+ |
| C++ Compiler | C++17 compatible (MSVC 2019+, GCC 9+) |

### Scene Preparation
- Input scenes must be exported as **glTF 2.0** (`.gltf` + `.bin`)
- All geometry must be **manifold** (each edge shared by exactly two faces)
- All objects must be **joined into a single mesh** before export
- Non-manifold edges can be corrected with Blender's *3D Printing Toolkit*

The glTF format is required because it preserves complete scene hierarchy, node transformations, and material properties in a GPU-friendly layout — and is the only format supported by both NVIDIA OptiX (via the `sutil` helpers) and Unreal Engine 5:

![glTF hierarchy](docs/images/gltf_hierarchy.png)

---

## Building

### Windows

Follow `SDK/INSTALL-WIN.txt` for the full setup. The short version:

```powershell
git clone https://github.com/Heart-lex/MeshDecimation-OptiX.git
cd MeshDecimation-OptiX

cmake -B build -S SDK `
      -DOptiX_INSTALL_DIR="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0" `
      -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```

### Linux

```bash
cmake -B build -S SDK \
      -DOptiX_INSTALL_DIR=/path/to/optix \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc)
```

---

## Usage

After building, run the visibility analysis executable against your glTF scene:

```bash
# Basic usage
./MeshDecimation --scene path/to/scene.gltf --output weights.txt

# Configure ray count and agent steps
./MeshDecimation --scene path/to/scene.gltf \
                 --rays 1024 \
                 --agent-steps 500 \
                 --output weights.txt
```

The output `weights.txt` is consumed directly by the [libigl repository](https://github.com/Heart-lex/MeshDecimation-libigl) to perform weighted decimation.

---

## Limitations

| Constraint | Detail |
|---|---|
| **Manifold geometry required** | Non-manifold meshes cause incorrect intersection tests and must be repaired before processing |
| **Static scenes only** | Geometry and lighting must remain fixed during analysis; dynamic objects are not supported |
| **Offline pipeline** | An 85% decimation of ~120k triangles takes ~80 minutes (serial libigl simplification) |
| **Single-layer visibility** | Only the first ray intersection is recorded; translucency and secondary reflections are not modelled |
| **RTX GPU required** | NVIDIA OptiX requires CUDA-capable hardware with hardware ray tracing support |
| **glTF input only** | Other formats (FBX, STL, PLY, OFF) alter face indexing and break triangle–weight correspondence |

---

## Repository Structure

```
MeshDecimation-OptiX/
├── SDK/
│   ├── CMakeLists.txt          # Build configuration
│   ├── INSTALL-WIN.txt
│   ├── INSTALL-LINUX.txt
│   ├── cuda/
│   │   ├── camera.cu           # Ray generation programs
│   │   ├── geometry.cu         # Geometry intersection
│   │   ├── shading.cu          # Per-hit weight accumulation
│   │   └── whitted.cu          # Path tracing base (modified for visibility)
│   ├── CMake/                  # FindOptiX, FindCUDA, compiler flags
│   └── data/                   # Sample scenes (Duck, ApartmentFloor, ...)
├── docs/images/                # Figures from the thesis
├── OptiX_EndUserLicense.pdf
└── README.md
```

---

## Related Work

- **Xanthopoulos (2021)** — *[Evaluation of Illumination-Based Mesh Simplification](http://graphics.cs.aueb.gr/graphics/docs/Light_driven_mesh_simplification.pdf)* — direct predecessor using lighting-driven importance via OptiX + libigl
- **Liu et al.** — *[Seamless: Seam erasure and seam-aware decoupling of shape from mesh resolution](https://www.cs.toronto.edu/~jacobson/images/seamless-seam-erasure-and-seam-aware-decoupling-of-shape-from-mesh-resolution-siggraph-asia-2017-compressed-liu-et-al.pdf)* — seam-aware decimation extending QSlim
- **Lindstrom & Turk (2000)** — *[Image-Driven Simplification](https://faculty.cc.gatech.edu/~turk/my_papers/image_simp_tog2000.pdf)* — multi-view image metric for visibility-aware simplification
- **Garland & Heckbert (1997)** — *[Surface Simplification Using Quadric Error Metrics](https://www.cs.cmu.edu/~garland/Papers/quadrics.pdf)* — the foundational QSlim edge-collapse algorithm

---

## Authors

**C. Chrysafi** and **A. A. Manetas**  
Athens University of Economics and Business — Department of Informatics  
Athens, 2025

---

## License

This repository is based on the **NVIDIA OptiX SDK 9.0.0** sample framework.  
See [`OptiX_EndUserLicense.pdf`](OptiX_EndUserLicense.pdf) and [`OptiX_ThirdParty_Licenses.txt`](OptiX_ThirdParty_Licenses.txt) for the terms covering SDK components.

Custom pipeline code (visibility analysis, agent sampling, importance map export) is the work of the authors above.
