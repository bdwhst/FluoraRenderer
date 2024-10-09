#include "sampling.h"

__device__ glm::vec2 util_math_uniform_sample_triangle(const glm::vec2& rand)
{
	float t = sqrt(rand.x);
	return glm::vec2(1 - t, t * rand.y);
}