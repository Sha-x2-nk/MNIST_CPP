#ifndef ADAM_H
#define ADAM_H
#include <numC/npGPUArray.cuh>

#include <iostream>

class AdamOptimiser
{
public:
    float learning_rate, beta1, beta2, epsilon;
    int t;
    np::ArrayGPU<float> m, v;

    AdamOptimiser(const float learning_rate = 0.001, const float beta1 = 0.9, const float beta2 = 0.999, const float epsilon = 1e-8);

    AdamOptimiser(AdamOptimiser &A);

    void operator=(AdamOptimiser &A);

    void step(np::ArrayGPU<float> &param, np::ArrayGPU<float> &grad);
};
#endif