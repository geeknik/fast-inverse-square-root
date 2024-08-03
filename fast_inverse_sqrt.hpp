#ifndef FAST_INVERSE_SQRT_HPP
#define FAST_INVERSE_SQRT_HPP

#include <vector>
#include <memory>

namespace fastinvsqrt {

// Forward declaration of the ComputeBackend class
class ComputeBackend;

class FastInvSqrt {
public:
    enum class Backend { CPU, CUDA, OpenCL };

    // Constructor
    FastInvSqrt(size_t size, Backend backend = Backend::CPU);

    // Destructor
    ~FastInvSqrt();

    // Compute method
    void compute(const std::vector<float>& input, std::vector<float>& output);

    // Deleted copy constructor and assignment operator
    FastInvSqrt(const FastInvSqrt&) = delete;
    FastInvSqrt& operator=(const FastInvSqrt&) = delete;

    // Move constructor and assignment operator
    FastInvSqrt(FastInvSqrt&&) noexcept;
    FastInvSqrt& operator=(FastInvSqrt&&) noexcept;

private:
    size_t size_;
    std::unique_ptr<ComputeBackend> backend_;
};

// Free function for single value computation
float q_rsqrt(float number);

} // namespace fastinvsqrt

#endif // FAST_INVERSE_SQRT_HPP
