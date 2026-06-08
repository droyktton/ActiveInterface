---

# Active-Nonlinear-Interfaces: High-Order Interface Growth & Active Noise Solver

This repository contains a high-performance CUDA/Thrust-accelerated spectral and real-space solver for simulating non-equilibrium, active nonlinear interface growth dynamics. The core framework focuses on modeling multi-scale crossovers and roughening regimes under the combined influence of **non-Hookean elasticity** and temporally correlated **active Ornstein–Uhlenbeck (OU) noise**.

This code is optimized for generating large-scale ensemble data matching analytical predictions (e.g., self-consistent Hartree approximations) detailed in our accompanying manuscript.

---

## 🔬 Mathematical Framework

The simulation numerically integrates the generalized stochastic partial differential equation governing the interface height profile $h(x,t)$:

$$\partial_t h(x,t) = c_2 \partial_x^2 h(x,t) + c_{2n} \partial_x \left[ \left(\partial_x h(x,t)\right)^{2n-1} \right] + \eta(x,t)$$

Where:

* **$c_2 > 0$**: The linear elasticity constant (Edwards-Wilkinson baseline surface tension).
* **$c_{2n} > 0$**: The anharmonic elasticity coefficient governing high-order gradient penalties ($n > 1$), which explicitly **breaks Statistical Tilt Symmetry (STS)**.
* **$\eta(x,t)$**: Active colored noise satisfying $\langle \eta(x,t) \rangle = 0$ with an exponential temporal correlation:

$$\langle \eta(x,t)\eta(x',t') \rangle = \frac{T}{\tau} \delta(x-x') e^{-\frac{|t-t'|}{\tau}}$$

where $\tau$ represents the active persistence timescale and $T$ is the noise temperature.

---

## ⚡ Features & Performance Optimizations

* **Parallelized Noise Update**: Implements decoupled, custom Thrust device functors utilizing highly vectorized pseudo-random number generation via `cuRAND` or `Random123 (Philox4x32)`.
* **Hybrid Real/Spectral Solver**: Efficiently switches execution spaces utilizing the `cuFFT` engine to compute exact structure factors $S(q,t)$ on the GPU without costly device-to-host memory overhead.
* **Multi-Precision Support**: Compile seamlessly in either `float` (single) or `double` precision depending on conservation law tolerances and hardware architectures.
* **Thermalization Warmup Loop**: Exposes an isolated, non-equilibrium initialization layer that pre-thermalizes the active colored noise field to its correct steady-state distribution prior to interface evolution.

---

## 🛠️ Installation & Dependencies

### Prerequisites

Ensure your local system has the following installed:

* **NVIDIA CUDA Toolkit (11.0 or higher)**
* **Thrust Template Library** (Bundled natively with CUDA)
* **Random123 Library** *(Optional, required only if compiling with `-DRANDOM123`)*
* A C++ compiler compatible with your CUDA version (`gcc`, `clang`, or `MSVC`)

### Cloning the Repository

```bash
git clone https://github.com/your-username/Active-Nonlinear-Interfaces.git
cd Active-Nonlinear-Interfaces

```

### Compilation

Compile the project using standard `nvcc` flags.

**For Double Precision (Recommended for long-time saturation limits):**

```bash
nvcc -O3 -arch=sm_70 -DDOUBLE ew.cu -lcufft -o interface_sim

```

**For Single Precision with Random123 Engine:**

```bash
nvcc -O3 -arch=sm_70 -DRANDOM123 ew.cu -lcufft -I /path/to/random123/include -o interface_sim

```

---

## 🚀 Running the Simulation

The executable parses runtime parameters defining both the lattice properties and physical scaling coefficients:

```bash
./interface_sim [L] [dt] [seed] [c2] [c2n] [tau] [temp]

```

### Example Command

To simulate a system of size $L = 4096$ with a time step $dt = 0.001$, a persistence time $\tau = 10.0$, and non-Hookean constraints:

```bash
./interface_sim 4096 0.001 42 1.0 0.1 10.0 0.1

```

### Output Artifacts

The program outputs standard real-space and spectral files for downstream post-processing and scaling analysis:

* `config.dat`: Final real-space height configurations $h(x)$.
* `sofq.dat`: Time-averaged steady-state Structure Factor grid $S(q)$.
* `roughness.dat`: Temporal evolution of global interface width $W(L,t) = \langle \overline{(h-\bar{h})^2} \rangle^{1/2}$ to measure dynamic and roughness exponents.

---

## 📊 Expected Scaling Regimes

This software is calibrated to capture three distinct self-affine regimes across spatial scales, depending on the coupling parameter 
$\Gamma_{2n} = \frac{c_{2n} T}{c_2^{(3n-1)/2} \sqrt{\tau}}$:

1. **Larkin Regimes**.
2. **Anharmonic Larkin Regime**.
3. **Edwards-Wilkinson**.

---

## 🤝 Contributing & Feedback

If you find a bug, missing factor, or wish to contribute optimization loops (such as alternate boundary condition arrays for tilted configurations), feel free to open an **Issue** or submit a **Pull Request**.

---

## 📜 Citation

If this codebase assists your research or analytical data processing, please cite the corresponding paper:

```bibtex
@article{Camara2026Roughening,
  title={Roughening of active nonlinear interfaces with broken tilt symmetry},
  author={C{\'a}mara, A. M. and Kolton, A. B. and Igua{\'\i}n, J. L.},
  journal={Physical Review B / E},
  year={2026},
  publisher={APS}
}

```
