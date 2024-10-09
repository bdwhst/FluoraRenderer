#pragma once
#ifndef SAMPLING_H
#define SAMPLING_H
#include <cuda.h>
#include <glm/glm.hpp>
__device__ glm::vec2 util_math_uniform_sample_triangle(const glm::vec2& rand);

#endif // !SAMPLING_H
