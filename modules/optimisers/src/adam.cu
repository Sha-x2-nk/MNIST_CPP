#include <optimisers/Adam.cuh>

#include <numC/npGPUArray.cuh>
#include <numC/npFunctions.cuh>

#include <cuda_runtime.h>

#include <iostream>
#include <cmath>

// default + paramterised constructor
AdamOptimiser::AdamOptimiser(const float learning_rate, const float beta1, const float beta2, const float epsilon)
{
    this->learning_rate = learning_rate;
    this->beta1 = beta1;
    this->beta2 = beta2;
    this->epsilon = epsilon;
    this->m = np::zeros<float>(1, 1);
    this->v = np::zeros<float>(1, 1);
    this->t = 0;
}

// copy constructor
AdamOptimiser::AdamOptimiser(AdamOptimiser &A)
{
    this->learning_rate = A.learning_rate;
    this->beta1 = A.beta1;
    this->beta2 = A.beta2;
    this->epsilon = A.epsilon;
    this->m = A.m;
    this->v = A.v;
    this->t = A.t;
}

// assignment operator overload for deep copy
void AdamOptimiser::operator=(AdamOptimiser &A)
{
    this->learning_rate = A.learning_rate;
    this->beta1 = A.beta1;
    this->beta2 = A.beta2;
    this->epsilon = A.epsilon;
    this->m = A.m;
    this->v = A.v;
    this->t = A.t;
}

// adam step (weight update)
void AdamOptimiser::step(np::ArrayGPU<float> &param, np::ArrayGPU<float> &grad)
{
    // time factor to take bigger steps initially
    ++this->t;

    // first momentum
    this->m = (this->m * this->beta1) + grad * (1 - this->beta1);
    auto mt = this->m / (1 - powf(this->beta1, static_cast<float>(this->t))); // bias correction

    // second momentum
    this->v = this->v * this->beta2 + np::square(grad) * (1 - this->beta2);
    auto vt = this->v / (1 - powf(this->beta2, static_cast<float>(this->t)));

    param = param - ( (mt * this->learning_rate) / (np::sqrt(vt) + this->epsilon) );

}