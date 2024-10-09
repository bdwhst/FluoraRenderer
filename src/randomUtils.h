#pragma once
#include <thrust/random.h>
#include <glm/glm.hpp>
/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__device__ inline
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

__device__ inline glm::vec2 util_sample2D(thrust::default_random_engine& rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);;
	return glm::vec2(u01(rng), u01(rng));
}