#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <optional>
#include <glm/glm.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace math
{
    __device__ __host__ __inline__ bool is_nan(float x)
    {
        return x != x;
    }

    template<typename T>
    __device__ __host__ float lerp(T x, T a, T b)
    {
        return (1 - x) * a + x * b;
    }
    template<typename T>
    __device__ __host__ float sqr(T x){ return x*x; }

    template <typename T, typename U, typename V>
    __device__ __host__ inline constexpr T clamp(T val, U low, V high) {
    if (val < low)
        return T(low);
    else if (val > high)
        return T(high);
    else
        return val;
    }
    //find position i such that f returns true for all position less or equal to i
    //such that f returns false for all position greater than i
    template <typename F>
    __device__ __host__ __inline__ size_t find_interval(size_t sz, const F& f)
    {
        using ssize_t = std::make_signed_t<size_t>;
        ssize_t size = (ssize_t)sz - 2, first = 1;
        while (size > 0)
        {
            size_t half = (size_t)size >> 1;
            size_t mid = first + half;
            bool p = f(mid);
            first = p ? mid + 1 : first;
            size = p ? size - half - 1 : half;
        }
        return (size_t)clamp((ssize_t)first - 1, 0, sz - 2);
    }

    /*__device__ __host__ inline float FMA(float a, float b, float c) {
        return fma(a, b, c);
    }

    __device__ __host__ inline double FMA(double a, double b, double c) {
        return fma(a, b, c);
    }*/

    template <typename Float, typename C>
    __device__ __host__  inline constexpr Float evaluate_polynomial(Float t, C c) {
        return c;
    }


    template <typename Float, typename C, typename... Args>
    __device__ __host__ inline constexpr Float evaluate_polynomial(Float t, C c, Args... cRemaining) {
        return fma(t, evaluate_polynomial(t, cRemaining...), c);
    }

    // https://github.com/explosion/murmurhash/blob/master/murmurhash/MurmurHash2.cpp
    __device__ __host__  inline uint64_t MurmurHash64A(const unsigned char* key, size_t len,
        uint64_t seed) {
        const uint64_t m = 0xc6a4a7935bd1e995ull;
        const int r = 47;

        uint64_t h = seed ^ (len * m);

        const unsigned char* end = key + 8 * (len / 8);

        while (key != end) {
            uint64_t k;
            std::memcpy(&k, key, sizeof(uint64_t));
            key += 8;

            k *= m;
            k ^= k >> r;
            k *= m;

            h ^= k;
            h *= m;
        }

        switch (len & 7) {
        case 7:
            h ^= uint64_t(key[6]) << 48;
        case 6:
            h ^= uint64_t(key[5]) << 40;
        case 5:
            h ^= uint64_t(key[4]) << 32;
        case 4:
            h ^= uint64_t(key[3]) << 24;
        case 3:
            h ^= uint64_t(key[2]) << 16;
        case 2:
            h ^= uint64_t(key[1]) << 8;
        case 1:
            h ^= uint64_t(key[0]);
            h *= m;
        };

        h ^= h >> r;
        h *= m;
        h ^= h >> r;

        return h;
    }

    template <typename T>
    __device__ __host__ inline uint64_t HashBuffer(const T* ptr, size_t size, uint64_t seed = 0) {
        return MurmurHash64A((const unsigned char*)ptr, size, seed);
    }


    template <int N>
    class SquareMatrix {
    public:
        __device__ __host__ static SquareMatrix zero()
        {
            SquareMatrix mat;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    mat.m[i][j] = 0;
            return mat;

        }
        __device__ __host__ SquareMatrix()
        {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    m[i][j] = (i == j) ? 1 : 0;
        }

        __device__ __host__ SquareMatrix(float v)
        {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    m[i][j] = v;
        }
        __device__ __host__
            SquareMatrix(const float mat[N][N]) {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    m[i][j] = mat[i][j];
        }

        __device__ __host__
            SquareMatrix operator+(const SquareMatrix& m) const {
            SquareMatrix r = *this;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    r.m[i][j] += m.m[i][j];
            return r;
        }

        __device__ __host__
            SquareMatrix operator*(float s) const {
            SquareMatrix r = *this;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    r.m[i][j] *= s;
            return r;
        }
        __device__ __host__
            SquareMatrix operator/(float s) const {
            SquareMatrix r = *this;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    r.m[i][j] /= s;
            return r;
        }

        __device__ __host__
            bool operator==(const SquareMatrix<N>& m2) const {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    if (m[i][j] != m2.m[i][j])
                        return false;
            return true;
        }

        __device__ __host__
            bool operator!=(const SquareMatrix<N>& m2) const {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    if (m[i][j] != m2.m[i][j])
                        return true;
            return false;
        }

        __device__ __host__
            float* operator[](int i) const { return (float*)m[i]; }

        float m[N][N];
    };

    template <int N>
    __device__ __host__ inline SquareMatrix<N> operator*(const SquareMatrix<N>& m1,
        const SquareMatrix<N>& m2) {
        SquareMatrix<N> r;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                r[i][j] = 0;
                for (int k = 0; k < N; ++k)
                    r[i][j] = std::fma(m1[i][k], m2[k][j], r[i][j]);
            }
        return r;
    }

    template <int N>
    __device__ __host__ inline SquareMatrix<N> transpose(const SquareMatrix<N>& m) {
        SquareMatrix<N> r;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r[i][j] = m[j][i];
        return r;
    }

    template <int N>
    __device__ __host__ bool inverse(const SquareMatrix<N>& m, SquareMatrix<N>& out) {
        int indxc[N], indxr[N];
        int ipiv[N] = { 0 };
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                out[i][j] = m[i][j];
        for (int i = 0; i < N; i++) {
            int irow = 0, icol = 0;
            float big = 0.f;
            // Choose pivot
            for (int j = 0; j < N; j++) {
                if (ipiv[j] != 1) {
                    for (int k = 0; k < N; k++) {
                        if (ipiv[k] == 0) {
                            if (std::abs(out[j][k]) >= big) {
                                big = std::abs(out[j][k]);
                                irow = j;
                                icol = k;
                            }
                        }
                        else if (ipiv[k] > 1)
                            return false;  // singular
                    }
                }
            }
            ++ipiv[icol];
            // Swap rows _irow_ and _icol_ for pivot
            if (irow != icol) {
                for (int k = 0; k < N; ++k)
                    std::swap(out[irow][k], out[icol][k]);
            }
            indxr[i] = irow;
            indxc[i] = icol;
            if (out[icol][icol] == 0.f)
                return false;  // singular

            // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
            float pivinv = 1. / out[icol][icol];
            out[icol][icol] = 1.;
            for (int j = 0; j < N; j++)
                out[icol][j] *= pivinv;

            // Subtract this row from others to zero out their columns
            for (int j = 0; j < N; j++) {
                if (j != icol) {
                    float save = out[j][icol];
                    out[j][icol] = 0;
                    for (int k = 0; k < N; k++)
                        out[j][k] = std::fma(-out[icol][k], save, out[j][k]);
                }
            }
        }
        // Swap columns to reflect permutation
        for (int j = N - 1; j >= 0; j--) {
            if (indxr[j] != indxc[j]) {
                for (int k = 0; k < N; k++)
                    std::swap(out[k][indxr[j]], out[k][indxc[j]]);
            }
        }
        return true;
    }

    template <int N>
    __device__ __host__
    bool linear_least_squares(const float A[][N], const float B[][N], int rows, SquareMatrix<N>& out) {
        SquareMatrix<N> AtA(0.0f);
        SquareMatrix<N> AtB(0.0f);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int r = 0; r < rows; ++r) {
                    AtA[i][j] += A[r][i] * A[r][j];
                    AtB[i][j] += A[r][i] * B[r][j];
                }
            }
        }
        SquareMatrix<N> AtAi;
        if (!inverse(AtA, AtAi)) {
            return false;
        }
        out = transpose(AtAi * AtB);
        return true;
    }


};

