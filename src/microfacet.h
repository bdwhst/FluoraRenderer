#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include "utilities.h"
#include "randomUtils.h"
#include "mathUtils.h"
#include "spectrum.h"
//#include "sceneStructs.h"
//#include "interactions.h"


__device__ float fr_complex(float cos_theta_i, math::complex<float> eta);

__device__ inline SampledSpectrum fr_complex(float cos_theta_i, SampledSpectrum eta, SampledSpectrum k)
{
	SampledSpectrum res(0.0f);
	for (int i = 0; i < spec::NSpectrumSamples; i++)
	{
		res[i] = fr_complex(cos_theta_i, { eta[i],k[i] });
	}
	return res;
}

//__device__ float util_alpha_i(const glm::vec3& wi, float alpha_x, float alpha_y);
//
//__device__ inline float util_sign(float x)
//{
//	return x == 0 ? 0 : (x > 0 ? 1 : -1);
//}
//
//__device__ float util_GGX_P22(float slope_x, float slope_y, float alpha_x, float alpha_y);
//
//__device__ float util_D(const glm::vec3& wm, float alpha_x, float alpha_y);
//
//__device__ float util_GGX_lambda(const glm::vec3& wi, float alpha_x, float alpha_y);
//
//__device__ inline float util_GGX_extinction_coeff(const glm::vec3& w, float alpha_x, float alpha_y)
//{
//	return w.z * util_GGX_lambda(w, alpha_x, alpha_y);
//}
//
//
//
//__device__ float util_GGX_projectedArea(const glm::vec3& wi, float alpha_x, float alpha_y);
//
//
//__device__ float util_Dwi(const glm::vec3& wi, const glm::vec3& wm, float alpha_x, float alpha_y);
//
//
//__device__ inline float util_pow_5(float x)
//{
//	float x2 = x * x;
//	return x2 * x2 * x;
//}
//
////TODO: use wavelength dependent frensel 
//__device__ inline glm::vec3 util_fschlick(glm::vec3 f0, glm::vec3 wi, glm::vec3 wh)
//{
//	float HoV = glm::max(glm::dot(wi, wh), 0.0f);
//	return f0 + (1.0f - f0) * util_pow_5(1.0f - HoV);
//}
//
//__device__ glm::vec3 util_conductor_evalPhaseFunction(const glm::vec3& wi, const glm::vec3& wo, float alpha_x, float alpha_y, const glm::vec3& albedo);
//
//__device__ glm::vec3 util_conductor_samplePhaseFunction(const glm::vec3& wi, const glm::vec3& random, glm::vec3& throughput, float alpha_x, float alpha_y, glm::vec3 albedo);
//
//__device__ float util_fresnel(const glm::vec3& wi, const glm::vec3& wm, const float eta);
//
//__device__  glm::vec3 util_refract(const glm::vec3& wi, const glm::vec3& wm, const float eta);
//
//
//__device__ glm::vec3 util_dielectric_samplePhaseFunctionFromSide(const glm::vec3& wi, const bool wi_outside, bool& wo_outside, glm::vec3 random, glm::vec3& throughput, float alpha_x, float alpha_y, glm::vec3 albedo, float m_eta);
//
////__device__ glm::vec3 util_dielectric_samplePhaseFunction(const glm::vec3& wi, const glm::vec3& random, glm::vec3& throughput, float alpha_x, float alpha_y, glm::vec3 albedo)
////{
////	bool wo_outside;
////	throughput = glm::vec3(1.0);
////	return util_dielectric_samplePhaseFunctionFromSide(wi, wi.z > 0, wo_outside, random, throughput, alpha_x, alpha_y, albedo);
////}
//
//__device__ glm::vec3 bxdf_asymConductor_sample(glm::vec3 wo, glm::vec3& throughput, thrust::default_random_engine& rng, const asymMicrofacetInfo& mat, int order);
//
//__device__ glm::vec3 bxdf_asymConductor_eval(const glm::vec3& wo, const glm::vec3& wi, const glm::ivec3& rndSeed, const asymMicrofacetInfo& mat, int order);
//
//__device__ inline glm::vec3 bxdf_asymConductor_sample_f(const glm::vec3& wo,  glm::vec3* wi, thrust::default_random_engine& rng, float* pdf, const asymMicrofacetInfo& mat, int order)
//{
//	glm::vec3 throughput = glm::vec3(1.0f);
//	*wi = bxdf_asymConductor_sample(wo, throughput, rng, mat, order);
//	*pdf = 1.0f;
//	return throughput;
//}
//
//__device__ inline float util_flip_z(float z)
//{
//	return log(1 - exp(z));
//}
//
//__device__ glm::vec3 bxdf_asymDielectric_sample(glm::vec3 wo, glm::vec3& throughput, thrust::default_random_engine& rng, const asymMicrofacetInfo& mat, int order);
//
//__device__ inline glm::vec3 bxdf_asymDielectric_sample_f(const glm::vec3& wo, glm::vec3* wi, thrust::default_random_engine& rng, float* pdf, const asymMicrofacetInfo& mat, int order)
//{
//	glm::vec3 throughput = glm::vec3(1.0f);
//	*wi = bxdf_asymDielectric_sample(wo, throughput, rng, mat, order);
//	*pdf = 1.0f;
//	return throughput;
//}


class TRDistribution
{
public: 
	__device__ TRDistribution(float _alpha_x, float _alpha_y) :alpha_x(_alpha_x), alpha_y(_alpha_y)
	{
	}
	__device__ float D(const glm::vec3& wm) const;
	__device__ float lambda(const glm::vec3& w) const;
	__device__ float G1(const glm::vec3& w) const
	{
		return 1 / (1 + lambda(w));
	}
	__device__ float G(const glm::vec3& wo, const glm::vec3& wi) const
	{
		return 1 / (1 + lambda(wo) + lambda(wi));
	}
	__device__ float D(const glm::vec3& w, const glm::vec3& wm)
	{
		using namespace math;
		return G1(w) / abs(cos_theta_vec(w)) * D(wm) * abs(glm::dot(w, wm));
	}
	__device__ float pdf(const glm::vec3& w, const glm::vec3& wm)
	{
		return D(w, wm);
	}
	__device__ glm::vec3 sample_wm(const glm::vec3& w, const glm::vec2& random)
	{
		glm::vec3 wh = glm::normalize(glm::vec3(w.x * alpha_x, w.y * alpha_y, w.z));
		if (wh.z < 0)
			wh = -wh;
		glm::vec3 t1 = (wh.z < 0.99999f) ? glm::normalize(glm::cross(glm::vec3(0, 0, 1), wh)) : glm::vec3(1, 0, 0);
		glm::vec3 t2 = glm::cross(wh, t1);
		glm::vec2 p = math::sample_uniform_disk_polar(random);
		float h = sqrtf(1 - math::sqr(p.x));
		p.y = math::lerp((1 + wh.z) / 2, h, p.y);
		float pz = sqrtf(fmaxf(0.0f, 1 - math::l2norm_squared(p)));
		glm::vec3 nh = p.x * t1 + p.y * t2 + pz * wh;
		return glm::normalize(glm::vec3(alpha_x * nh.x, alpha_y * nh.y, fmaxf(1e-6f, nh.z)));
	}
	__device__ bool effectively_smooth() const
	{
		return fmaxf(alpha_x, alpha_y) < 1e-3f;
	}
	float alpha_x, alpha_y;
};