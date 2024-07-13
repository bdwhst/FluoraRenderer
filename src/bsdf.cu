#include "bsdf.h"
#include "mathUtils.h"
#include "spectrum.h"
#include "microfacet.h"

__device__ SampledSpectrum BxDF::sample_f(const glm::vec3& wo, glm::vec3& wi, float& pdf, thrust::default_random_engine& rng)
{
	auto op = [&](auto ptr) { return ptr->sample_f(wo, wi, pdf, rng); };
	return Dispatch(op);
}

__device__ SampledSpectrum BxDF::eval(const glm::vec3& wo, const glm::vec3& wi, thrust::default_random_engine& rng)
{
	auto op = [&](auto ptr) { return ptr->eval(wo, wi, rng); };
	return Dispatch(op);
}

__device__ float BxDF::pdf(const glm::vec3& wo, const glm::vec3& wi)
{
	auto op = [&](auto ptr) { return ptr->pdf(wo, wi); };
	return Dispatch(op);
}

__device__ inline glm::vec3 util_sample_hemisphere_uniform(const glm::vec2& random)
{
	float z = random.x;
	float sq_1_z_2 = sqrt(max(1 - z * z, 0.0f));
	float phi = TWO_PI * random.y;
	return glm::vec3(cos(phi) * sq_1_z_2, sin(phi) * sq_1_z_2, z);
}

__device__ inline glm::vec2 util_sample_disk_uniform(const glm::vec2& random)
{
	float r = sqrt(random.x);
	float theta = TWO_PI * random.y;
	return glm::vec2(r * cos(theta), r * sin(theta));
}

__device__ inline glm::vec3 util_sample_hemisphere_cosine(const glm::vec2& random)
{
	glm::vec2 t = util_sample_disk_uniform(random);
	return glm::vec3(t.x, t.y, sqrt(1 - t.x * t.x - t.y * t.y));
}

__device__ inline glm::vec2 util_sample2D(thrust::default_random_engine& rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);;
	return glm::vec2(u01(rng), u01(rng));
}

__device__ inline float util_math_tangent_space_clampedcos(const glm::vec3& w)
{
	return max(w.z, 0.0f);
}

__device__ SampledSpectrum DiffuseBxDF::sample_f(const glm::vec3& wo, glm::vec3& wi, float& pdf, thrust::default_random_engine& rng)
{
	glm::vec2 random = util_sample2D(rng);
    wi = util_sample_hemisphere_cosine(random);
	pdf = util_math_tangent_space_clampedcos(wi) * INV_PI;
	float cosWi = abs(wi.z);
	return reflectance * INV_PI * cosWi;
}

__device__ SampledSpectrum DiffuseBxDF::eval(const glm::vec3& wo, const glm::vec3& wi, thrust::default_random_engine& rng)
{
	float cosWi = abs(wi.z);
    return reflectance * INV_PI * cosWi;
}

__device__ float DiffuseBxDF::pdf(const glm::vec3& wo, const glm::vec3& wi)
{
    return util_math_tangent_space_clampedcos(wi) * INV_PI;
}

__device__ inline float util_math_sin_cos_convert(float sinOrCos)
{
	return sqrt(max(1 - sinOrCos * sinOrCos, 0.0f));
}

__device__ inline float util_math_frensel_dielectric(float cosThetaI, float etaI, float etaT)
{
	cosThetaI = math::clamp(cosThetaI, -1.0f, 1.0f);
	float sinThetaI = util_math_sin_cos_convert(cosThetaI);
	float sinThetaT = etaI / etaT * sinThetaI;
	if (sinThetaT >= 1) return 1;//total reflection
	float cosThetaT = util_math_sin_cos_convert(sinThetaT);
	float rparll = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
	float rperpe = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
	return (rparll * rparll + rperpe * rperpe) * 0.5;
}

__device__ inline bool util_geomerty_refract(const glm::vec3& wi, const glm::vec3& n, float eta, glm::vec3* wt)
{
	float cosThetaI = glm::dot(wi, n);
	float sin2ThetaI = max(0.0f, 1 - cosThetaI * cosThetaI);
	float sin2ThetaT = eta * eta * sin2ThetaI;
	if (sin2ThetaT >= 1) return false;
	float cosThetaT = sqrt(1 - sin2ThetaT);
	*wt = eta * (-wi) + (eta * cosThetaI - cosThetaT) * n;
	return true;
}


__device__ SampledSpectrum DielectricBxDF::sample_f(const glm::vec3& wo, glm::vec3& wi, float& pdf, thrust::default_random_engine& rng)
{
	glm::vec2 random = util_sample2D(rng);
    float frensel = util_math_frensel_dielectric(abs(wo.z), 1.0f, eta);
	if (random.x < frensel)
	{
		wi = glm::vec3(-wo.x, -wo.y, wo.z);
		pdf = frensel;
		return SampledSpectrum(frensel);
	}
	else
	{
		glm::vec3 n = glm::dot(wo, glm::vec3(0, 0, 1)) > 0 ? glm::vec3(0, 0, 1) : glm::vec3(0, 0, -1);
		glm::vec3 refractedRay;
		if (!util_geomerty_refract(wo, n, 1 / eta, &refractedRay)) return SampledSpectrum(0.0f);
		wi = refractedRay;
		pdf = 1 - frensel;
		//todo: account for albedo
		//todo: account for the difference when tracing importance rather than radiance
		SampledSpectrum val((1 - frensel) / (eta * eta));
		return val;
	}
}

__device__ SampledSpectrum DielectricBxDF::eval(const glm::vec3& wo, const glm::vec3& wi, thrust::default_random_engine& rng)
{
    return SampledSpectrum(0.0f);
}

__device__ float DielectricBxDF::pdf(const glm::vec3& wo, const glm::vec3& wi)
{
    return 0.0f;
}

__device__ SampledSpectrum ConductorBxDF::sample_f(const glm::vec3& wo, glm::vec3& wi, float& pdf, thrust::default_random_engine& rng)
{
	//todo
	pdf = 0.0f;
	return SampledSpectrum(0.0f);
}



__device__ SampledSpectrum ConductorBxDF::eval(const glm::vec3& wo, const glm::vec3& wi, thrust::default_random_engine& rng)
{
	//todo
	glm::vec3 wh = wo + wi;
	if (wh.x * wh.x + wh.y * wh.y + wh.z * wh.z < 1e-9f) return SampledSpectrum(0.0f);
	wh = glm::normalize(wh);
	if (wi.z > 0.0f && glm::dot(wh, wi) > 0.0f)
	{
		float a2 = roughness * roughness;
		glm::vec3 F = util_math_fschlick(baseColor, glm::max(glm::dot(wo, wh), 0.0f));
		float D = util_math_ggx_normal_distribution(wh, a2);
		float G2 = util_math_smith_ggx_shadowing_masking(wi, wo, a2);
		return (F * G2 * D) / (max(4 * util_math_tangent_space_clampedcos(wo) * util_math_tangent_space_clampedcos(wi), 1e-9f));
	}
	else
	{
		return glm::vec3(0);
	}
	return SampledSpectrum(0.0f);
}

__device__ float ConductorBxDF::pdf(const glm::vec3& wo, const glm::vec3& wi)
{
	//todo
	return 0.0f;
}