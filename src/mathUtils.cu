#include "mathUtils.h"

namespace math
{
    __device__ glm::vec2 sample_uniform_disk_polar(const glm::vec2& u)
    {
        float r = sqrtf(u.x);
        float theta = 2 * pi * u[1];
        return { r * cosf(theta),r * sinf(theta) };
    }
}