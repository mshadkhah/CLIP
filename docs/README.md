CLIP: A CUDA-Accelerated Lattice Boltzmann Framework for Interfacial Phenomena

**CLIP** is a high-performance, extensible Lattice Boltzmann Method (LBM) library developed in C++/CUDA, designed to simulate interfacial phenomena such as bubbles, drops, and jets in both 2D and 3D domains.

It features a modular, GPU-accelerated architecture that enables **easy integration of additional physical models and equations**. Thanks to its flexible design, **adding new models or equations requires minimal effort**, making it ideal for extending to novel research problems.

Moreover, **CLIP is highly optimized**, allowing users to run many simulations — including 3D multiphase problems — **on a personal computer without requiring a high-end GPU**. For memory-constrained or low-end GPUs, users can **enable single precision** to significantly reduce memory usage and improve performance. With its fully configurable input system, CLIP is ideal for rapid prototyping, educational use, and scalable research.

---

## Table of Contents

- [Key Features](\ref key_features)
- [Examples](\ref examples)
- [Build Instructions](\ref build_instructions)
- [Contact](\ref contact)

---

\anchor key_features
## 🔧 Key Features

- **CUDA-Accelerated Kernels**
- **Two-Phase Interfacial Flow**
- **Versatile Boundary Conditions**
- **Checkpointing Support**
- **Config-Driven Simulation**
- **WMRT Collision Stability**
- **Allen–Cahn Phase Field**
- **2D/3D Lattice Support**
- **Single/Double Precision Toggle**

---

\anchor examples
## 🧪 Examples

CLIP supports the following types of simulations:

- **Rayleigh–Taylor Instability** (2D & 3D)
- **Bubble Dynamics** (2D & 3D)
- **Drop Dynamics** (2D & 3D)
- **Jet Breakup & Pinch-Off** (2D & 3D)

Animations are available in the GitHub version of the README.

---

\anchor build_instructions
## ⚙️ Build Instructions

### Prerequisites

- CUDA 8.0+
- CMake ≥ 3.18
- GCC ≥ 9
- NVIDIA GPU (e.g., V100, A100, RTX, or even entry-level GPUs)
- Python (optional, for post-processing)

### Steps


**Clone the repository**
```bash
git clone https://github.com/your-org/CLIP.git
cd CLIP
```


**Create a build directory**
```bash
mkdir build && cd build
```

**Configure with CMake**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_2D=ON                 # 2D double precision
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_3D=ON                 # 3D double precision
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_2D=ON -DUSE_SINGLE_PRECISION=ON  # 2D single precision
```


**Build**
```bash
make -j
```

---

\anchor contact
## Contact

Developed and maintained by **Mehdi Shadkhah**  
📧 mshadkhah@gmail.com  