#include "fast_inverse_sqrt.hpp"
#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>
#include <immintrin.h>
#include <thread>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <functional>
#include <optional>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

#ifdef USE_OPENCL
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>
#endif

namespace fastinvsqrt {

// Compile-time computation of magic number
template<typename T>
constexpr uint64_t compute_magic_number() {
    if constexpr (std::is_same_v<T, float>) {
        return 0x5f3759df;
    } else if constexpr (std::is_same_v<T, double>) {
        return 0x5fe6eb50c7b537a9;
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Unsupported floating-point type");
    }
}

// Template for different floating-point types
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
Q_rsqrt(T number) {
    if (number < std::numeric_limits<T>::min()) {
        if (number == 0) return std::numeric_limits<T>::infinity();
        if (number < 0) return std::numeric_limits<T>::quiet_NaN();
        return T(1) / std::sqrt(number);  // Handle subnormal numbers
    }
    if (number == std::numeric_limits<T>::infinity()) return 0;

    using IntType = typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;
    
    constexpr T threehalfs = T(1.5);
    T x2 = number * T(0.5);
    T y = number;
    
    IntType i = std::bit_cast<IntType>(y);
    i = compute_magic_number<T>() - (i >> 1);
    y = std::bit_cast<T>(i);
    
    y = y * (threehalfs - (x2 * y * y));
    y = y * (threehalfs - (x2 * y * y));
    y = y * (threehalfs - (x2 * y * y));  // Additional iteration for higher precision
    
    return y;
}

#ifdef __AVX512F__
// SIMD optimized version for float (AVX512)
__m512 Q_rsqrt_simd(__m512 x) {
    __m512 halfx = _mm512_mul_ps(x, _mm512_set1_ps(0.5f));
    __m512 y = x;
    __m512i i = _mm512_castps_si512(y);
    i = _mm512_sub_epi32(_mm512_set1_epi32(0x5f3759df), _mm512_srli_epi32(i, 1));
    y = _mm512_castsi512_ps(i);
    __m512 threehalfs = _mm512_set1_ps(1.5f);
    
    y = _mm512_mul_ps(y, _mm512_sub_ps(threehalfs, _mm512_mul_ps(_mm512_mul_ps(halfx, y), y)));
    y = _mm512_mul_ps(y, _mm512_sub_ps(threehalfs, _mm512_mul_ps(_mm512_mul_ps(halfx, y), y)));
    y = _mm512_mul_ps(y, _mm512_sub_ps(threehalfs, _mm512_mul_ps(_mm512_mul_ps(halfx, y), y)));
    
    return y;
}
#endif

// Multi-threaded SIMD version
void Q_rsqrt_mt_simd(float* input, float* output, size_t n, int num_threads) {
    std::vector<std::thread> threads;
    size_t chunk_size = n / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([=]() {
            size_t start = i * chunk_size;
            size_t end = (i == num_threads - 1) ? n : (i + 1) * chunk_size;

            #ifdef __AVX512F__
            for (size_t j = start; j < end; j += 16) {
                __m512 x = _mm512_loadu_ps(&input[j]);
                __m512 y = Q_rsqrt_simd(x);
                _mm512_storeu_ps(&output[j], y);
            }
            #else
            for (size_t j = start; j < end; ++j) {
                output[j] = Q_rsqrt(input[j]);
            }
            #endif
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

#ifdef USE_CUDA
// CUDA kernel for GPU acceleration
__global__ void cuda_rsqrt(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = rsqrtf(input[idx]);
    }
}

class CudaCompute {
public:
    CudaCompute(size_t size) : size_(size) {
        cudaMalloc(&d_input_, size_ * sizeof(float));
        cudaMalloc(&d_output_, size_ * sizeof(float));
    }

    ~CudaCompute() {
        cudaFree(d_input_);
        cudaFree(d_output_);
    }

    void compute(const float* input, float* output) {
        cudaMemcpy(d_input_, input, size_ * sizeof(float), cudaMemcpyHostToDevice);
        
        int block_size = 256;
        int num_blocks = (size_ + block_size - 1) / block_size;
        cuda_rsqrt<<<num_blocks, block_size>>>(d_input_, d_output_, size_);
        cudaDeviceSynchronize();
        
        cudaMemcpy(output, d_output_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
    }

private:
    size_t size_;
    float *d_input_, *d_output_;
};
#endif

#ifdef USE_OPENCL
class OpenCLCompute {
public:
    OpenCLCompute(size_t size) : size_(size) {
        // OpenCL setup code here
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms[0];

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        device_ = devices[0];

        context_ = cl::Context(device_);
        queue_ = cl::CommandQueue(context_, device_);

        // Create buffers
        input_buffer_ = cl::Buffer(context_, CL_MEM_READ_ONLY, size_ * sizeof(float));
        output_buffer_ = cl::Buffer(context_, CL_MEM_WRITE_ONLY, size_ * sizeof(float));

        // Compile the kernel
        std::string kernel_code = R"(
            __kernel void rsqrt(__global const float* input, __global float* output) {
                int i = get_global_id(0);
                output[i] = rsqrt(input[i]);
            }
        )";
        cl::Program program(context_, kernel_code);
        program.build({device_});
        kernel_ = cl::Kernel(program, "rsqrt");
    }

    void compute(const float* input, float* output) {
        queue_.enqueueWriteBuffer(input_buffer_, CL_TRUE, 0, size_ * sizeof(float), input);

        kernel_.setArg(0, input_buffer_);
        kernel_.setArg(1, output_buffer_);
        queue_.enqueueNDRangeKernel(kernel_, cl::NullRange, cl::NDRange(size_));

        queue_.enqueueReadBuffer(output_buffer_, CL_TRUE, 0, size_ * sizeof(float), output);
    }

private:
    size_t size_;
    cl::Device device_;
    cl::Context context_;
    cl::CommandQueue queue_;
    cl::Buffer input_buffer_, output_buffer_;
    cl::Kernel kernel_;
};
#endif

// Computation backend abstraction
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    virtual void compute(const float* input, float* output, size_t size) = 0;
};

class CPUBackend : public ComputeBackend {
public:
    void compute(const float* input, float* output, size_t size) override {
        int num_threads = std::thread::hardware_concurrency();
        Q_rsqrt_mt_simd(const_cast<float*>(input), output, size, num_threads);
    }
};

#ifdef USE_CUDA
class CUDABackend : public ComputeBackend {
public:
    CUDABackend(size_t size) : cuda_compute_(size) {}
    void compute(const float* input, float* output, size_t size) override {
        cuda_compute_.compute(input, output);
    }
private:
    CudaCompute cuda_compute_;
};
#endif

#ifdef USE_OPENCL
class OpenCLBackend : public ComputeBackend {
public:
    OpenCLBackend(size_t size) : opencl_compute_(size) {}
    void compute(const float* input, float* output, size_t size) override {
        opencl_compute_.compute(input, output);
    }
private:
    OpenCLCompute opencl_compute_;
};
#endif

// High-level API
class FastInvSqrt {
public:
    enum class Backend { CPU, CUDA, OpenCL };

    FastInvSqrt(size_t size, Backend backend = Backend::CPU) : size_(size) {
        switch (backend) {
            case Backend::CPU:
                backend_ = std::make_unique<CPUBackend>();
                break;
            #ifdef USE_CUDA
            case Backend::CUDA:
                backend_ = std::make_unique<CUDABackend>(size);
                break;
            #endif
            #ifdef USE_OPENCL
            case Backend::OpenCL:
                backend_ = std::make_unique<OpenCLBackend>(size);
                break;
            #endif
            default:
                throw std::runtime_error("Unsupported backend");
        }
    }

    void compute(const std::vector<float>& input, std::vector<float>& output) {
        if (input.size() != size_ || output.size() != size_) {
            throw std::invalid_argument("Input and output sizes must match the initialized size");
        }
        backend_->compute(input.data(), output.data(), size_);
    }

private:
    size_t size_;
    std::unique_ptr<ComputeBackend> backend_;
};

// Benchmark function
template<typename Func>
double benchmark(Func f, const char* name, const std::vector<float>& input, std::vector<float>& output, int iterations = 10) {
    std::vector<double> times;
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        f(input, output);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        times.push_back(diff.count());
    }
    double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::cout << name << " average time: " << avg_time << " s" << std::endl;
    return avg_time;
}

} // namespace fastinvsqrt

int main() {
    try {
        constexpr size_t N = 10'000'000;
        std::vector<float> input(N);
        std::vector<float> output(N);

        // Initialize input data
        std::generate(input.begin(), input.end(), [n = 0.0f]() mutable { return n += 1.0f; });

        // CPU
        fastinvsqrt::FastInvSqrt cpu_compute(N, fastinvsqrt::FastInvSqrt::Backend::CPU);
        fastinvsqrt::benchmark([&](const std::vector<float>& in, std::vector<float>& out) {
            cpu_compute.compute(in, out);
        }, "CPU", input, output);

        #ifdef USE_CUDA
        // CUDA
        fastinvsqrt::FastInvSqrt cuda_compute(N, fastinvsqrt::FastInvSqrt::Backend::CUDA);
        fastinvsqrt::benchmark([&](const std::vector<float>& in, std::vector<float>& out) {
            cuda_compute.compute(in, out);
        }, "CUDA", input, output);
        #endif

        #ifdef USE_OPENCL
        // OpenCL
        fastinvsqrt::FastInvSqrt opencl_compute(N, fastinvsqrt::FastInvSqrt::Backend::OpenCL);
        fastinvsqrt::benchmark([&](const std::vector<float>& in, std::vector<float>& out) {
            opencl_compute.compute(in, out);
        }, "OpenCL", input, output);
        #endif

        // Verify results
        std::cout << std::setprecision(10);
        for (size_t i = 0; i < std::min(size_t(10), N); ++i) {
            float true_value = 1.0f / std::sqrt(input[i]);
            float relative_error = std::abs(output[i] - true_value) / true_value;
            std::cout << "rsqrt(" << input[i] << ") = " << output[i] 
                      << " (true value: " << true_value 
                      << ", relative error: " << relative_error << ")" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
