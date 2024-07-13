#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include "utilities.h"
#include "randomUtils.h"
#include "sceneStructs.h"
#include "interactions.h"


__device__ float util_frcomplex()
{

}

__device__ float util_alpha_i(const glm::vec3& wi, float alpha_x, float alpha_y)
{
	const float invSinTheta2 = 1.0f / (1.0f - wi.z * wi.z);
	const float cosPhi2 = wi.x * wi.x * invSinTheta2;
	const float sinPhi2 = wi.y * wi.y * invSinTheta2;
	const float alpha_i = sqrtf(cosPhi2 * alpha_x * alpha_x + sinPhi2 * alpha_y * alpha_y);
	return alpha_i;
}

__device__ inline float util_sign(float x)
{
	return x == 0 ? 0 : (x > 0 ? 1 : -1);
}

__device__ float util_GGX_P22(float slope_x, float slope_y, float alpha_x, float alpha_y)
{
	const float tmp = 1.0f + slope_x * slope_x / (alpha_x * alpha_x) + slope_y * slope_y / (alpha_y * alpha_y);
	const float value = 1.0f / (PI * alpha_x * alpha_x) / (tmp * tmp);
	return value;
}

__device__ float util_D(const glm::vec3& wm, float alpha_x, float alpha_y){
	if (wm.z <= 0.0f)
		return 0.0f;

	// slope of wm
	const float slope_x = -wm.x / wm.z;
	const float slope_y = -wm.y / wm.z;

	// value
	const float value = util_GGX_P22(slope_x, slope_y, alpha_x, alpha_y) / (wm.z * wm.z * wm.z * wm.z);
	return value;
}

__device__ float util_GGX_lambda(const glm::vec3& wi, float alpha_x, float alpha_y)
{
	if (wi.z > 0.999999f)
		return 1e-7f;
	if (wi.z < -0.999999f)
		return -1.0f;

	// a
	const float theta_i = acosf(wi.z);
	const float a = tanf(theta_i) * util_alpha_i(wi, alpha_x, alpha_y);

	// value
	const float value = 0.5f * (-1.0f + util_sign(a) * sqrtf(max(1 + (a * a),0.0f)));

	return value;
}

__device__ float util_GGX_extinction_coeff(const glm::vec3& w, float alpha_x, float alpha_y)
{
	return w.z * util_GGX_lambda(w, alpha_x, alpha_y);
}



__device__ float util_GGX_projectedArea(const glm::vec3& wi, float alpha_x, float alpha_y)
{
	if (wi.z > 0.9999f)
		return 1.0f;
	if (wi.z < -0.9999f)
		return 0.0f;

	// a
	const float theta_i = acosf(wi.z);
	const float sin_theta_i = sinf(theta_i);

	const float alphai = util_alpha_i(wi, alpha_x, alpha_y);

	// value
	const float value = 0.5f * (wi.z + sqrtf(wi.z * wi.z + sin_theta_i * sin_theta_i * alphai * alphai));

	return value;
}


__device__ float util_Dwi(const glm::vec3& wi, const glm::vec3& wm, float alpha_x, float alpha_y){
	if (wm.z <= 0.0f)
		return 0.0f;

	// normalization coefficient
	const float projectedarea = util_GGX_projectedArea(wi, alpha_x, alpha_y);
	if (projectedarea == 0)
		return 0;
	const float c = 1.0f / projectedarea;

	// value
	const float value = c * glm::max(0.0f, dot(wi, wm)) * util_D(wm, alpha_x, alpha_y);
	return value;
}


__device__ inline float util_pow_5(float x)
{
	float x2 = x * x;
	return x2 * x2 * x;
}

//TODO: use wavelength dependent frensel 
__device__ inline glm::vec3 util_fschlick(glm::vec3 f0, glm::vec3 wi, glm::vec3 wh)
{
	float HoV = glm::max(glm::dot(wi, wh), 0.0f);
	return f0 + (1.0f - f0) * util_pow_5(1.0f - HoV);
}

__device__ glm::vec3 util_conductor_evalPhaseFunction(const glm::vec3& wi, const glm::vec3& wo, float alpha_x, float alpha_y, const glm::vec3& albedo)
{
	// half vector 
	const glm::vec3 wh = normalize(wi + wo);
	if (wh.z < 0.0f)
		return glm::vec3(0.0f);

	// value
	return 0.25f * util_Dwi(wi, wh, alpha_x, alpha_y) * util_fschlick(albedo, wi, wh) / dot(wi, wh);
}

__device__ glm::vec3 util_conductor_samplePhaseFunction(const glm::vec3& wi, const glm::vec3& random, glm::vec3& throughput, float alpha_x, float alpha_y, glm::vec3 albedo)
{
	glm::vec3 wh = util_sample_ggx_vndf(wi, glm::vec2(random), alpha_x, alpha_y);

	// reflect
	glm::vec3 wo = glm::normalize(-wi + 2.0f * wh * dot(wi, wh));
	throughput *= util_fschlick(albedo, wi, wh);

	return wo;
}

__device__ float util_fresnel(const glm::vec3& wi, const glm::vec3& wm, const float eta)
{
	const float cos_theta_i = glm::dot(wi, wm);
	const float cos_theta_t2 = 1.0f - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta);

	// total internal reflection 
	if (cos_theta_t2 <= 0.0f) return 1.0f;

	const float cos_theta_t = sqrtf(cos_theta_t2);

	const float Rs = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
	const float Rp = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);

	const float F = 0.5f * (Rs * Rs + Rp * Rp);
	return F;
}

__device__  glm::vec3 util_refract(const glm::vec3& wi, const glm::vec3& wm, const float eta)
{
	const float cos_theta_i = dot(wi, wm);
	const float cos_theta_t2 = 1.0f - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta);
	const float cos_theta_t = sqrtf(max(0.0f, cos_theta_t2));

	return wm * (cos_theta_i / eta - cos_theta_t) - wi / eta;
}


__device__ glm::vec3 util_dielectric_samplePhaseFunctionFromSide(const glm::vec3& wi, const bool wi_outside, bool& wo_outside, glm::vec3 random, glm::vec3& throughput, float alpha_x, float alpha_y, glm::vec3 albedo, float m_eta)
{
	const float U1 = random.x;
	const float U2 = random.y;
	const float etaI = wi_outside ? 1.0f : m_eta;
	const float etaT = wi_outside ? m_eta : 1.0f;
	glm::vec3 wm = wi.z > 0 ? util_sample_ggx_vndf(wi, glm::vec2(random.x, random.y), alpha_x, alpha_y) : -util_sample_ggx_vndf(-wi, glm::vec2(random.x, random.y), alpha_x, alpha_y);

	const float F = util_fresnel(wi, wm, etaT / etaI);

	if (random.z < F)
	{
		const glm::vec3 wo = -wi + 2.0f * wm * dot(wi, wm); // reflect
		return glm::normalize(wo);
	}
	else
	{
		wo_outside = !wi_outside;
		const glm::vec3 wo = util_refract(wi, wm, etaT / etaI);
		if (glm::dot(wo, wi) > 0) return glm::vec3(0, 0, 1);

		return glm::normalize(wo);
	}
}

//__device__ glm::vec3 util_dielectric_samplePhaseFunction(const glm::vec3& wi, const glm::vec3& random, glm::vec3& throughput, float alpha_x, float alpha_y, glm::vec3 albedo)
//{
//	bool wo_outside;
//	throughput = glm::vec3(1.0);
//	return util_dielectric_samplePhaseFunctionFromSide(wi, wi.z > 0, wo_outside, random, throughput, alpha_x, alpha_y, albedo);
//}

__device__ glm::vec3 bxdf_asymConductor_sample(glm::vec3 wo, glm::vec3& throughput, thrust::default_random_engine& rng, const asymMicrofacetInfo& mat, int order)
{
	float z = 0;
	glm::vec3 w = glm::normalize(-wo);
	int i = 0;
	thrust::uniform_real_distribution<float> u01(0, 1);

	while (i < order)
	{	
		float U = u01(rng);
		float sigmaIn = max(z > mat.zs ? util_GGX_extinction_coeff(w, mat.alphaXA, mat.alphaYA) : util_GGX_extinction_coeff(w, mat.alphaXB, mat.alphaYB),0.0f);
		float sigmaOut = max(z > mat.zs ? util_GGX_extinction_coeff(w, mat.alphaXB, mat.alphaYB) : util_GGX_extinction_coeff(w, mat.alphaXA, mat.alphaYA),0.0f);
		float deltaZ = w.z / glm::length(w) * (-log(U) / sigmaIn);
		if (z < mat.zs != z + deltaZ < mat.zs)
		{
			deltaZ = (mat.zs - z) + (deltaZ - (mat.zs - z)) * sigmaIn / sigmaOut;
		}
		z += deltaZ;
		if (z > 0) break;
		glm::vec3 rand3 = glm::vec3(u01(rng), u01(rng), u01(rng));
		glm::vec3 nw;
		if (z > mat.zs)
		{
			w = util_conductor_samplePhaseFunction(-w, rand3, throughput, mat.alphaXA, mat.alphaYA, mat.albedo);
		}
		else
		{
			w = util_conductor_samplePhaseFunction(-w, rand3, throughput, mat.alphaXB, mat.alphaYB, mat.albedo);
		}
		if ((z != z) || (w.z != w.z))
			return glm::vec3(0.0f, 0.0f, 1.0f);
		i++;
	}
	if (z < 0)
	{
		throughput = glm::vec3(0.0);
		return glm::vec3(0, 0, 1);
	}
	return w;
}

__device__ glm::vec3 bxdf_asymConductor_eval(const glm::vec3& wo, const glm::vec3& wi, const glm::ivec3& rndSeed, const asymMicrofacetInfo& mat, int order)
{
	float z = 0;
	glm::vec3 w = glm::normalize(-wo);
	glm::vec3 result = glm::vec3(0);
	glm::vec3 throughput = glm::vec3(1.0f);
	int i = 0;
	thrust::default_random_engine rng = makeSeededRandomEngine(rndSeed.x, rndSeed.y, rndSeed.z);
	thrust::uniform_real_distribution<float> u01(0, 1);
	while (i < order)
	{
		float U = u01(rng);
		float sigmaIn = z > mat.zs ? util_GGX_extinction_coeff(w, mat.alphaXA, mat.alphaYA) : util_GGX_extinction_coeff(w, mat.alphaXB, mat.alphaYB);
		float sigmaOut = z > mat.zs ? util_GGX_extinction_coeff(w, mat.alphaXB, mat.alphaYB) : util_GGX_extinction_coeff(w, mat.alphaXA, mat.alphaYA);
		float deltaZ = w.z / glm::length(w) * (-log(U) / sigmaIn);
		if (z < mat.zs != z + deltaZ < mat.zs)
		{
			deltaZ = (mat.zs - z) + (deltaZ - (mat.zs - z)) * sigmaIn / sigmaOut;
		}
		z += deltaZ;
		if (z > 0) break;
		glm::vec3 p = z > mat.zs ? mat.fEval(-w, wi, mat.alphaXA, mat.alphaYA, mat.albedo) : mat.fEval(-w, wi, mat.alphaXB, mat.alphaYB, mat.albedo);
		float tau_exit = glm::max(z, mat.zs) * util_GGX_lambda(wi, mat.alphaXA, mat.alphaYA) + glm::min(z - mat.zs, 0.0f) * util_GGX_lambda(wi, mat.alphaXB, mat.alphaYB);
		result += throughput * exp(tau_exit) * p;
		glm::vec3 rand3 = glm::vec3(u01(rng), u01(rng), u01(rng));
		if(z> mat.zs)
		{
			w = util_conductor_samplePhaseFunction(-w, rand3, throughput, mat.alphaXA, mat.alphaYA, mat.albedo);
		}
		else
		{
			w = util_conductor_samplePhaseFunction(-w, rand3, throughput, mat.alphaXB, mat.alphaYB, mat.albedo);
		}


		if ((z != z) || (w.z != w.z))
			return glm::vec3(0.0f);
		i++;
	}
	return result;
}

__device__ inline glm::vec3 bxdf_asymConductor_sample_f(const glm::vec3& wo,  glm::vec3* wi, thrust::default_random_engine& rng, float* pdf, const asymMicrofacetInfo& mat, int order)
{
	glm::vec3 throughput = glm::vec3(1.0f);
	*wi = bxdf_asymConductor_sample(wo, throughput, rng, mat, order);
	*pdf = 1.0f;
	return throughput;
}

__device__ float util_flip_z(float z)
{
	return log(1 - exp(z));
}

__device__ glm::vec3 bxdf_asymDielectric_sample(glm::vec3 wo, glm::vec3& throughput, thrust::default_random_engine& rng, const asymMicrofacetInfo& mat, int order)
{
	constexpr float eta = 1.5f;
	float z = 0;
	glm::vec3 w = glm::normalize(-wo);
	int i = 0;
	thrust::uniform_real_distribution<float> u01(0, 1);
	bool outside = wo.z > 0;
	float zs = outside ? mat.zs : util_flip_z(mat.zs);
	bool flipped = !outside;
	if (!outside) w = -w;

	while (i < order)
	{
		float U = u01(rng);
		float alphaXA = outside ? mat.alphaXA : mat.alphaXB;
		float alphaYA = outside ? mat.alphaYA : mat.alphaYB;
		float alphaXB = outside ? mat.alphaXB : mat.alphaXA;
		float alphaYB = outside ? mat.alphaYB : mat.alphaYA;
		float sigmaIn = z > zs ? util_GGX_extinction_coeff(w, alphaXA, alphaYA) : util_GGX_extinction_coeff(w, alphaXB, alphaYB);
		float sigmaOut = z > zs ? util_GGX_extinction_coeff(w, alphaXB, alphaYB) : util_GGX_extinction_coeff(w, alphaXA, alphaYA);
		if (sigmaIn == 0.0f)
		{
			z = 0.0;
			break;
		}
		float deltaZ = w.z * (-log(U) / sigmaIn);
		if (z < zs != z + deltaZ < zs)
		{
			deltaZ = (zs - z) + (deltaZ - (zs - z)) * sigmaIn / sigmaOut;
		}
		z += deltaZ;
		if (z > 0) break;
		glm::vec3 rand3 = glm::vec3(u01(rng), u01(rng), u01(rng));
		bool n_outside = outside;
		if (z > zs)
		{
			w = util_dielectric_samplePhaseFunctionFromSide(-w, outside, n_outside, rand3, throughput, alphaXA, alphaYA, mat.albedo, eta);
		}
		else
		{
			w = util_dielectric_samplePhaseFunctionFromSide(-w, outside, n_outside, rand3, throughput, alphaXB, alphaYB, mat.albedo, eta);
		}
		if ((z != z) || (w.z != w.z))
			return glm::vec3(0.0f, 0.0f, 1.0f);
		if (n_outside!=outside)
		{
			z = util_flip_z(z);
			zs = util_flip_z(zs);
			w = -w;
			outside = !outside;
			flipped = !flipped;
		}
		i++;
	}
	if (z < 0)
	{
		throughput = glm::vec3(0.0);
		return glm::vec3(0, 0, 1);
	}
	if (flipped) w = -w;
	if (w.z * wo.z < 0)
	{
		float etap = wo.z > 0 ? 1 / eta : eta;
		throughput *= (etap * etap);
	}
	return w;
}

__device__ inline glm::vec3 bxdf_asymDielectric_sample_f(const glm::vec3& wo, glm::vec3* wi, thrust::default_random_engine& rng, float* pdf, const asymMicrofacetInfo& mat, int order)
{
	glm::vec3 throughput = glm::vec3(1.0f);
	*wi = bxdf_asymDielectric_sample(wo, throughput, rng, mat, order);
	*pdf = 1.0f;
	return throughput;
}