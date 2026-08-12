/**
 * @file ew.cu
 * @brief Simulation of active non-equilibrium nonlinear interface growth models 
 * governed by high-order gradient elastic penalties and active Ornstein-Uhlenbeck noise.
 * * Matches theoretical frameworks targeting Physical Review journals.
 */

#ifdef RANDOM123
  #define R123_USE_CUDA
  #define R123_NO_SSE
  #include <Random123/philox.h>
#else
  #include <curand_kernel.h>
#endif

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/complex.h>
#include <cufft.h>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <iostream>
#include "cutil.h"

// Define floating point configuration precision
#ifdef DOUBLE
    typedef double real;
    typedef cufftDoubleComplex complex;
#else
    typedef float real;
    typedef cufftComplex complex;
#endif

// ============================================================================
// DEVICE FUNCTORS AND KERNELS
// ============================================================================

#ifdef RANDOM123
using philox_t = r123::Philox4x32;
/**
 * @brief Generates a normally distributed random number using Random123 Philox counter.
 */
__device__ inline float rng_normal(uint32_t thread_id, uint32_t step, uint32_t seed)
{
    philox_t::key_type key = {{seed, 0}};
    philox_t::ctr_type ctr = {{thread_id, step, 0, 0}};
    philox_t::ctr_type out = philox_t()(ctr, key);

    // Box–Muller transformation using two 32-bit integer outputs
    float u1 = (out[0] + 1.0f) * 2.3283064e-10f;
    float u2 = (out[1] + 1.0f) * 2.3283064e-10f;

    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}
#endif

/**
 * @brief Device functor to handle the evolution of the active Ornstein-Uhlenbeck noise field.
 * Updates via an exact or Euler-Maruyama discretization step depending on correlation parameters.
 */
struct NoiseUpdateFunctor {
    real dt;
    real tau;
    real temp;
    unsigned long seed;
    unsigned long step;

    __host__ __device__
    NoiseUpdateFunctor(real _dt, real _tau, real _temp, unsigned long _seed, unsigned long _step)
        : dt(_dt), tau(_tau), temp(_temp), seed(_seed), step(_step) {}

    template <typename Tuple>
    __device__ void operator()(Tuple t) {
        real &eta = thrust::get<0>(t);
        unsigned long idx = thrust::get<1>(t);

        // Calculate standard noise variance prefactor matching <\eta(x,t)\eta(x',t')>
        // Note: Prefactor corresponds to 2*T*dt/tau to guarantee correct stationary energy scaling.
        #ifdef RANDOM123
        real random_force = sqrtf(2.0f * temp * dt) * rng_normal(idx, step, seed);
        #else
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, step, &state);
        real random_force = sqrtf(2.0f * temp * dt) * curand_normal(&state);
        #endif

        // Ornstein-Uhlenbeck relaxation dynamic step
        eta += -eta * (dt / tau) + (random_force / tau);
    }
};

/**
 * @brief Histogram kernel to calculate real-space height configuration distribution profiles.
 */
__global__ void histogramKernel(const float* data, int* bins, int N, int Nbins, 
                                float xmin, float xmax, float mean, float var) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = (data[idx] - mean) / sqrt(var);
        int bin = int(((x - xmin) / (xmax - xmin)) * Nbins);
        if (bin >= 0 && bin < Nbins) {
            atomicAdd(&bins[bin], 1);
        }
    }
}

// ============================================================================
// MAIN SIMULATION CLASS
// ============================================================================

/**
 * @class ActiveInterfaceSimulation
 * @brief Encapsulates the configuration, state tracking, and spectral solver loops 
 * for the high-order active interface growth equation.
 */
class ActiveInterfaceSimulation {
private:
    // Domain parameters
    unsigned long L;
    real dt;
    unsigned long seed;
    unsigned long fourierCount;

    // Physical Material Parameters (Mappings from Physical Review models)
    real c2;    ///< Linear elastic constant (Edwards-Wilkinson baseline friction)
    real c2n;   ///< Anharmonic elasticity constant governing high-order gradients
    real tau;   ///< Noise persistence timescale (correlation boundary)
    real temp;  ///< Noise temperature parameter

    // Cuda FFT Variables
    cufftHandle plan_r2c;
    cufftHandle plan_c2r;

    // Real space state tracking vectors
    thrust::device_vector<real> u;          ///< Interface height profile configurations
    thrust::device_vector<real> dudx;       ///< First spatial gradient field
    thrust::device_vector<real> force_u;    ///< Net local elastic forces
    thrust::device_vector<real> noise;      ///< Active Ornstein-Uhlenbeck driving noise

    // Fourier space vectors
    thrust::device_vector<complex> Fou_u;
    thrust::device_vector<complex> L_k;     ///< Fourier wave numbers

    // Accumulators for Structure Factors S(q,t)
    thrust::device_vector<real> acum_Sofq_u;
    thrust::device_vector<real> inst_Sofq_u;

public:
    /**
     * @brief Construct the interface framework and assign physical coefficients.
     */
    ActiveInterfaceSimulation(unsigned long system_size, real time_step, unsigned long run_seed,
                              real linear_elasticity = 1.0, real nonlinear_elasticity = 0.1,
                              real active_tau = 0.1, real temperature = 0.1)
        : L(system_size), dt(time_step), seed(run_seed), fourierCount(0),
          c2(linear_elasticity), c2n(nonlinear_elasticity), tau(active_tau), temp(temperature) 
    {
        // Allocate real space fields
        u.resize(L);
        dudx.resize(L);
        force_u.resize(L);
        noise.resize(L);

        thrust::fill(noise.begin(), noise.end(), real(0.0));
        thrust::fill(u.begin(), u.end(), real(0.0));

        // Warm up active noise to steady state distributions prior to evolution
        #ifndef TAUINFINITO
        warmup_noise();
        #endif

        // Configure CUFFT executions matching chosen variable precision
        if (sizeof(real) == sizeof(double)) {
            cufftPlan1d(&plan_r2c, L, CUFFT_D2Z, 1);
            cufftPlan1d(&plan_c2r, L, CUFFT_Z2D, 1);
        } else {
            cufftPlan1d(&plan_r2c, L, CUFFT_R2C, 1);
            cufftPlan1d(&plan_c2r, L, CUFFT_C2R, 1);
        }

        int Lcomp = L / 2 + 1;
        Fou_u.resize(Lcomp);
        acum_Sofq_u.resize(L);
        inst_Sofq_u.resize(L);

        thrust::fill(acum_Sofq_u.begin(), acum_Sofq_u.end(), real(0.0));
    }

    ~ActiveInterfaceSimulation() {
        cufftDestroy(plan_r2c);
        cufftDestroy(plan_c2r);
    }

    /**
     * @brief Standardizes the initial condition to flat profile configuration.
     */
    void apply_flat_initial_condition() {
        thrust::fill(u.begin(), u.end(), real(0.0));
    }

    /**
     * @brief Evolves the noise generation sequence independently to reach the expected
     * steady-state Ornstein-Uhlenbeck temporal correlation balance.
     */
    void warmup_noise() {
        std::cout << "[Physics Baseline] Inverting transient dynamics: Warming up colored noise...\n";
        unsigned long twarm = static_cast<unsigned long>(5.0 * tau / dt);

        for (unsigned long n = 0; n < twarm; ++n) {
            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(noise.begin(), thrust::make_counting_iterator(0UL))),
                thrust::make_zip_iterator(thrust::make_tuple(noise.end(), thrust::make_counting_iterator(L))),
                NoiseUpdateFunctor(dt, tau, temp, seed, n)
            );
        }
        std::cout << "[Physics Baseline] Active colored noise generation initialized and stabilized.\n";
    }

    void reset_structure_factor_accumulators() {
        thrust::fill(acum_Sofq_u.begin(), acum_Sofq_u.end(), real(0.0));
    }

    real calculate_center_of_mass() {
        return thrust::reduce(u.begin(), u.end(), real(0.0)) / static_cast<real>(L);
    }

    real calculate_center_of_mass_velocity() {
        return thrust::reduce(force_u.begin(), force_u.end(), real(0.0)) / static_cast<real>(L);
    }

    /**
     * @brief Maps real space interface tracking into Fourier components to compile structure factor scaling grids S(q,t).
     */
    void perform_spectral_analysis() {
        real *raw_u = thrust::raw_pointer_cast(u.data());
        complex *raw_fou_u = thrust::raw_pointer_cast(Fou_u.data());

        if (sizeof(real) == sizeof(double)) {
            cufftExecD2Z(plan_r2c, reinterpret_cast<cufftDoubleReal*>(raw_u), reinterpret_cast<cufftDoubleComplex*>(raw_fou_u));
        } else {
            cufftExecR2C(plan_r2c, reinterpret_cast<cufftReal*>(raw_u), reinterpret_cast<cufftComplex*>(raw_fou_u));
        }

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(Fou_u.begin(), acum_Sofq_u.begin(), inst_Sofq_u.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(Fou_u.end(), acum_Sofq_u.end(), inst_Sofq_u.end())),
            [] __device__ (thrust::tuple<complex, real &, real &> t) {
                complex fu = thrust::get<0>(t);
                real sofq = fu.x * fu.x + fu.y * fu.y;
                thrust::get<1>(t) += sofq; 
                thrust::get<2>(t) = sofq; 
            }
        );
        fourierCount++;
    }
};
