#ifndef ADAM_CUH
#define ADAM_CUH

#include <numC/npGPUArray.cuh>

class AdamOptimiser
{
public:
    float learning_rate, beta1, beta2, epsilon;
    int t;

    // for first and second moments of adam
    np::ArrayGPU<float> m, v;

    // default + parameter constructor
    AdamOptimiser(const float learning_rate = 0.001, const float beta1 = 0.9, const float beta2 = 0.999, const float epsilon = 1e-8);

    // copy constructor
    AdamOptimiser(AdamOptimiser &A);

    // assignment operator overload for deep copy
    void operator=(AdamOptimiser &A);

    // adam step (weight update)
    void step(np::ArrayGPU<float> &param, np::ArrayGPU<float> &grad);
};
#endif