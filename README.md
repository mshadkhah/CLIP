[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


<p align="center">
  <img src="https://raw.githubusercontent.com/mshadkhah/CLIP/gh-pages/assets/ClipLogo.png" alt="CLIP Logo" width="250">
</p>

<h1 align="center"><strong> CLIP: A CUDA-Accelerated Lattice Boltzmann Framework for Interfacial Phenomena</strong></h1>

**CLIP** is a high-performance, extensible Lattice Boltzmann Method (LBM) library developed in C++/CUDA, designed to simulate interfacial phenomena such as bubbles, drops, and jets in both 2D and 3D domains. It features a modular, GPU-accelerated architecture that enables easy integration of additional physical models and equations. Thanks to its flexible design, adding new models or equations requires minimal effort, making it ideal for extending to novel research problems.

Moreover, **CLIP is highly optimized**, allowing users to run many simulations — including 3D multiphase problems — **on a personal computer without requiring a high-end GPU**. For memory-constrained or low-end GPUs, users can **enable single precision** to significantly reduce memory usage and improve performance, making it possible to simulate larger domains efficiently. With its fully configurable input system, users can explore diverse simulation scenarios without modifying code, making **CLIP** an ideal tool for rapid prototyping, educational use, and scalable research.






## Table of Contents
- [Key Features](#key-features)
- [Examples](#examples)
- [Build Instructions](#build-instructions)
- [Running a Case](#running-a-case)
- [Configuration Format](#configuration-format)
- [Capabilities](#capabilities)
<!-- - [Citation](#citation) -->
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Contact](#contact)




---

<!-- \anchor key_features -->
## Key Features


- ⚡ **CUDA-Accelerated Kernels**
- 🌊 **Two-Phase Interfacial Flow**
- 🧱 **Versatile Boundary Conditions**
- 🔁 **Checkpointing Support**
- 🧪 **Config-Driven Simulation**
- ⚖️ **WMRT Collision Stability**
- 🌗 **Allen–Cahn Phase Field**
- 🧩 **2D/3D Lattice Support**
- 🔧 **Single/Double Precision Toggle**



---

<!-- \anchor examples -->
## Examples

- ✅ **Rayleigh–Taylor Instability** (2D & 3D)
- ✅ **Bubble Dynamics** (2D & 3D)
- ✅ **Drop Dynamics** (2D & 3D)
- ✅ **Jet Breakup & Pinch-Off** (2D & 3D)


<h3 align="center"><strong> Liquid Jet Simulation</strong></h3>
<p align="center">
  <img src="assets/Jet3D.gif" width="700">
</p>


<p align="center">
Figure: 3D simulation of a liquid jet undergoing primary and secondary breakup due to interfacial instabilities. The jet destabilizes under the influence of shear and surface tension forces, leading to fragmentation into droplets — a hallmark of multiphase jet atomization dynamics.
</p>

<h3 align="center"><strong> Rayleigh–Taylor Instability</strong></h3>

<p align="center">
  <img src="https://raw.githubusercontent.com/mshadkhah/CLIP/gh-pages/assets/RTI2D.gif" width="300" style="margin-right: 0px;">
  <img src="https://raw.githubusercontent.com/mshadkhah/CLIP/gh-pages/assets/RTI3D.gif" width="350">
</p>

<p align="center">
Figure: Visualization of 2D and 3D Rayleigh–Taylor instability simulations performed using CLIP, a GPU-accelerated Lattice Boltzmann Method (LBM) solver.
</p>


<h3 align="center"><strong> Drop/Bubble Dynamics</strong></h3>


<p align="center">
  <img src="https://raw.githubusercontent.com/mshadkhah/CLIP/gh-pages/assets/drop.gif" width="700">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mshadkhah/CLIP/gh-pages/assets/bubble.gif" width="700">
</p>

<p align="center">
Figure: Bubble and drop dynamics in a periodic domain under varying Reynolds and Weber numbers, highlighting the influence of inertial, viscous, and surface tension forces on deformation, breakup, and coalescence processes.

</p>


<!-- \anchor build -->
## Build Instructions

### Prerequisites

- CUDA 8.0+
- CMake ≥ 3.18
- GCC ≥ 9
- NVIDIA GPU (e.g., V100, A100, RTX, or even entry-level GPUs)
- Python for post-processing (optional)

### Build

**1. Clone the repository**
```bash
git clone https://github.com/your-org/CLIP.git
cd CLIP
```

**2. Create and enter a build directory**
```bash
 2. Create and enter a build directory
mkdir build && cd build
```

**3. Configure with CMake**

 2D simulation in double precision (default)
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_2D=ON
```

3D simulation in double precision
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_3D=ON
```
On low-end GPUs or memory-constrained systems, you can enable single precision to reduce memory usage and fit larger meshes within the available GPU memory.
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_2D=ON -DUSE_SINGLE_PRECISION=ON
```

**4. Build the executable**
```bash
make -j
```

---
<!-- \anchor running -->
## Running a Case

```bash
./clip examples/2D/Bubble/config.txt
```

Visit the `examples/` folder for examples including:

- Bubble dynamics
- Drop dynamics
- Rayleigh–Taylor instability
- Jet simulaions

---
<!-- \anchor config -->
## Configuration Format

CLIP simulations are driven by a flexible, text-based `config.txt` file. Here's a sample configuration for a 2D drop impact simulation:

```ini
case = drop
finalStep = 100000
outputInterval = 1000
reportInterval = 1000
checkpointInterval = 1000
checkpointCopy = 5

N = [64, 128, 1]         # Grid size (X, Y, Z)
referenceLength = 25     # Used for dimensionless parameter scaling

We = 5                   # Weber number
Re = 100                 # Reynolds number
gravity = [0, -1e-6, 0]  # Gravity vector
mobility = 0.02
interfaceWidth = 4
muRatio = 100            # Viscosity ratio (μ_high / μ_low)
rhoRatio = 1000          # Density ratio (ρ_high / ρ_low)

geometry = [
  {
    type = "circle"
    center = [32, 20, 0]
    radius = 10
    id = 0
  }
]

boundary = [
  { side = "x-", type = "periodic" },
  { side = "x+", type = "periodic" },
  { side = "y-", type = "wall" },
  { side = "y+", type = "wall" }
]
```

Each simulation can define geometry, boundary conditions, and physical parameters without recompiling the code. This makes CLIP well-suited for quick experimentation and parameter sweeps.

For more examples, visit the `examples/` directory.


---
<!-- \anchor capabilities -->
### Capabilities
**CLIP** is a flexible and high-performance Lattice Boltzmann Method (LBM) framework designed to simulate a variety of interfacial phenomena. The framework is built with modularity in mind, making it easy to extend with new physical models, equations, and simulations. The code is written in such a way that users can seamlessly integrate additional equations or models, allowing for rapid prototyping and adaptation to new research areas.



| Category              | Support                        |
|-----------------------|---------------------------------|
| **LB Equations**       | Navier–Stokes, Allen–Cahn       |
| **LBM Models**         | Weighted MRT                   |
| **Lattice Structures** | D2Q9 (2D), D3Q19 (3D)           |
| **Geometry via SDF**   | Circle, Sphere, Square, Cube, Perturbation |
| **Initial Conditions** | Bubble, Drop, Jet, RTI, etc.   |
| **Boundary Conditions**| Periodic, Wall, Slip-Wall, Neumann, Velocity, Free Convective, Do-Nothing |
| **Restart / Checkpoint** | ✅ Yes                        |
| **Outputs**            | ASCII VTK, Binary Checkpoints  |


We have implemented a comprehensive set of boundary conditions in **CLIP**, covering most of the typical scenarios encountered in LBM simulations:

- **Periodic Boundaries** – For domains with repeating patterns (e.g., channel or jet flows).
- **Wall Boundaries** – Includes **half-way bounce-back** modeling for enhanced accuracy and stability, especially at **high Reynolds numbers**.
- **Slip-Wall Boundaries** – Allow tangential fluid motion with no normal penetration; useful for idealized smooth-wall flows.
- **Neumann Boundaries** – Prescribed gradient (zero-gradient) at the boundary; used in thermal or scalar transport simulations.
- **Velocity Boundaries** – Supports imposed velocity profiles on inlets/outlets or moving boundaries.
- **Free Convective Boundaries** – Models fluid-wall interaction with convective behavior (useful in natural convection setups).
- **Do-Nothing Boundaries** – Apply when the simulation does not need to enforce any condition on the boundary.

These options ensure **CLIP** can handle a wide range of real-world and research-driven multiphase flow scenarios. **CLIP** utilizes a **Weighted MRT** model for better numerical stability and accuracy over standard BGK. It supports **Navier–Stokes** and **Allen–Cahn** equations, making it suitable for simulating two-phase flows with interfacial dynamics. With **D2Q9** and **D3Q19** lattices, the framework is adaptable to complex domains and phenomena. Output is provided in VTK for visualization and binary format for efficient checkpointing and restart.


---


## Roadmap

This section outlines current development efforts and future goals for the CLIP framework:

| Feature                          | Status       | Description                                                                 |
|----------------------------------|--------------|-----------------------------------------------------------------------------|
| **Adaptive Mesh Refinement**     | In Progress  | Dynamic grid refinement based on flow features like interface or vorticity. |
| **STL & Mesh Geometry Import**   | In Progress  | Support for importing external geometry files for realistic domains.        |
| **Multi-GPU via NCCL**           | In Progress  | Parallel execution across GPUs using NCCL or MPI for large-scale problems.  |
| **Thermal & Compressible LBM**   | Wishlist     | Extend CLIP to handle heat transfer and compressible flows.                 |
| **Lattice BGK / Entropic LBM**   | Planned      | Additional collision models for flexibility and improved stability.         |
| **Pre/Post-Processing Tools**    | Wishlist     | Built-in tools for initialization, conversion, and visualization.           |
| **Python Bindings**              | Wishlist     | Expose CLIP APIs for interactive scripting and integration in Python.       |
| **AMGX Integration**             | Wishlist     | Use NVIDIA's GPU-based solvers for coupled systems or hybrid methods.       |

---

<!-- \anchor contributing -->
## Contributing

We welcome contributions of all kinds — from bug reports and feature suggestions to pull requests. If you'd like to contribute, feel free to fork the repository and open a pull request. For larger changes or ideas, please start a discussion or open an issue first. If you're unsure where to begin, don't hesitate to reach out via email.


---

<!-- \anchor citation -->
<!-- ## Citation

If you use **CLIP** in your work, please cite:

```bibtex
@software{clip2025,
  author  = {Mehdi Shadkhah},
  title   = {CLIP: A CUDA-Accelerated Lattice Boltzmann Framework for Interfacial Phenomena},
  year    = {2025},
  url     = {https://github.com/your-org/culbm}
}
```
```bibtex
@paper{culbm2024,
  author  = {M.Shadkhah, M. Rahni, ....},
  title   = {CLIP: A CUDA-Accelerated Lattice Boltzmann Framework for Interfacial Phenomena},
  year    = {2025},
  url     = {https://github.com/your-org/culbm}
}
``` -->


## Contact

Developed and maintained by **Mehdi Shadkhah**  
📧 mshadkhah@gmail.com  

---
