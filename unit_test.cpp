#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include "fast_inverse_sqrt.hpp"

using namespace fastinvsqrt;
using namespace testing;

class FastInvSqrtTest : public Test {
protected:
    void SetUp() override {
        // Initialize with some random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(1.0, 1000.0);

        input.resize(TEST_SIZE);
        for (auto& val : input) {
            val = static_cast<float>(dis(gen));
        }

        output.resize(TEST_SIZE);
        expected_output.resize(TEST_SIZE);
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            expected_output[i] = 1.0f / std::sqrt(input[i]);
        }
    }

    static constexpr size_t TEST_SIZE = 1000000;
    std::vector<float> input;
    std::vector<float> output;
    std::vector<float> expected_output;
};

TEST_F(FastInvSqrtTest, CPUBackendAccuracy) {
    FastInvSqrt cpu_compute(TEST_SIZE, FastInvSqrt::Backend::CPU);
    cpu_compute.compute(input, output);

    for (size_t i = 0; i < TEST_SIZE; ++i) {
        EXPECT_NEAR(output[i], expected_output[i], 1e-3f);
    }
}

TEST_F(FastInvSqrtTest, CPUBackendEdgeCases) {
    std::vector<float> edge_input = {0.0f, std::numeric_limits<float>::min(), 
                                     std::numeric_limits<float>::max(), INFINITY, -1.0f};
    std::vector<float> edge_output(edge_input.size());

    FastInvSqrt cpu_compute(edge_input.size(), FastInvSqrt::Backend::CPU);
    cpu_compute.compute(edge_input, edge_output);

    EXPECT_FLOAT_EQ(edge_output[0], INFINITY);
    EXPECT_NEAR(edge_output[1], 1.0f / std::sqrt(std::numeric_limits<float>::min()), 1e-3f);
    EXPECT_NEAR(edge_output[2], 1.0f / std::sqrt(std::numeric_limits<float>::max()), 1e-3f);
    EXPECT_FLOAT_EQ(edge_output[3], 0.0f);
    EXPECT_TRUE(std::isnan(edge_output[4]));
}

TEST_F(FastInvSqrtTest, CPUBackendPerformance) {
    FastInvSqrt cpu_compute(TEST_SIZE, FastInvSqrt::Backend::CPU);

    auto start = std::chrono::high_resolution_clock::now();
    cpu_compute.compute(input, output);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "CPU Backend computation time: " << diff.count() << " s" << std::endl;

    // This is a basic performance test. You might want to set a specific
    // threshold based on your performance requirements.
    EXPECT_LT(diff.count(), 1.0);  // Expect computation to take less than 1 second
}

#ifdef USE_CUDA
TEST_F(FastInvSqrtTest, CUDABackendAccuracy) {
    FastInvSqrt cuda_compute(TEST_SIZE, FastInvSqrt::Backend::CUDA);
    cuda_compute.compute(input, output);

    for (size_t i = 0; i < TEST_SIZE; ++i) {
        EXPECT_NEAR(output[i], expected_output[i], 1e-3f);
    }
}

TEST_F(FastInvSqrtTest, CUDABackendPerformance) {
    FastInvSqrt cuda_compute(TEST_SIZE, FastInvSqrt::Backend::CUDA);

    auto start = std::chrono::high_resolution_clock::now();
    cuda_compute.compute(input, output);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "CUDA Backend computation time: " << diff.count() << " s" << std::endl;

    EXPECT_LT(diff.count(), 0.1);  // Expect CUDA to be significantly faster than CPU
}
#endif

#ifdef USE_OPENCL
TEST_F(FastInvSqrtTest, OpenCLBackendAccuracy) {
    FastInvSqrt opencl_compute(TEST_SIZE, FastInvSqrt::Backend::OpenCL);
    opencl_compute.compute(input, output);

    for (size_t i = 0; i < TEST_SIZE; ++i) {
        EXPECT_NEAR(output[i], expected_output[i], 1e-3f);
    }
}

TEST_F(FastInvSqrtTest, OpenCLBackendPerformance) {
    FastInvSqrt opencl_compute(TEST_SIZE, FastInvSqrt::Backend::OpenCL);

    auto start = std::chrono::high_resolution_clock::now();
    opencl_compute.compute(input, output);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "OpenCL Backend computation time: " << diff.count() << " s" << std::endl;

    EXPECT_LT(diff.count(), 0.1);  // Expect OpenCL to be significantly faster than CPU
}
#endif

TEST_F(FastInvSqrtTest, CompareBackends) {
    FastInvSqrt cpu_compute(TEST_SIZE, FastInvSqrt::Backend::CPU);
    std::vector<float> cpu_output(TEST_SIZE);
    cpu_compute.compute(input, cpu_output);

    #ifdef USE_CUDA
    FastInvSqrt cuda_compute(TEST_SIZE, FastInvSqrt::Backend::CUDA);
    std::vector<float> cuda_output(TEST_SIZE);
    cuda_compute.compute(input, cuda_output);

    for (size_t i = 0; i < TEST_SIZE; ++i) {
        EXPECT_NEAR(cpu_output[i], cuda_output[i], 1e-3f);
    }
    #endif

    #ifdef USE_OPENCL
    FastInvSqrt opencl_compute(TEST_SIZE, FastInvSqrt::Backend::OpenCL);
    std::vector<float> opencl_output(TEST_SIZE);
    opencl_compute.compute(input, opencl_output);

    for (size_t i = 0; i < TEST_SIZE; ++i) {
        EXPECT_NEAR(cpu_output[i], opencl_output[i], 1e-3f);
    }
    #endif
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
