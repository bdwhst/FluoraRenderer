#pragma once
#include "taggedptr.h"
#include "bsdf.h"
#include "spectrum.h"
#include "color.h"
#include <unordered_map>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include "memoryUtils.h"
class DiffuseMaterial;
class DielectricMaterial;
class ConductorMaterial;
class EmissiveMaterial;

class MaterialParams
{
public:
	MaterialParams() = default;

	void insert_float(const std::string& key, float value) { float_data.insert({ key,value }); }
	void insert_vec3(const std::string& key, const glm::vec3& value) { vec3_data.insert({ key,value }); }
	void insert_texture(const std::string& key, cudaTextureObject_t value) { tex_data.insert({ key,value }); }
	void insert_ptr(const std::string& key, void* value) { ptr_data.insert({ key,value }); }
	void insert_spectrum(const std::string& key, Spectrum value) { spec_data.insert({ key,value }); }
	void insert_string(const std::string& key, const std::string& value) { str_data.insert({ key,value }); }
	float get_float(const std::string& key) const
	{
		if (float_data.count(key))
		{
			return float_data.at(key);
		}
		return 0.0f;
	}
	glm::vec3 get_vec3(const std::string& key)  const
	{ 
		if (vec3_data.count(key))
		{
			return vec3_data.at(key);
		}
		return glm::vec3(0.0f);
	}
	cudaTextureObject_t get_texture(const std::string& key) const
	{
		if (tex_data.count(key))
		{
			return tex_data.at(key);
		}
		return 0;
	}
	void* get_ptr(const std::string& key) const
	{
		if (ptr_data.count(key))
		{
			return ptr_data.at(key);
		}
		return nullptr;
	}
	Spectrum get_spec(const std::string& key) const
	{
		if (spec_data.count(key))
		{
			return spec_data.at(key);
		}
		return nullptr;
	}
	std::string get_string(const std::string& key) const
	{
		if (str_data.count(key))
		{
			return str_data.at(key);
		}
		return {};
	}
	std::unordered_map<std::string, Spectrum>& get_specs()
	{
		return spec_data;
	}
private:
	std::unordered_map<std::string, float> float_data;
	std::unordered_map<std::string, cudaTextureObject_t> tex_data;
	std::unordered_map<std::string, glm::vec3> vec3_data;
	std::unordered_map<std::string, void*> ptr_data;
	std::unordered_map<std::string, Spectrum> spec_data;
	std::unordered_map<std::string, std::string> str_data;
};

struct MaterialEvalInfo
{
	glm::vec3 wo;
	glm::vec2 uv;
	SampledWavelengths& swl;
	__device__ MaterialEvalInfo(const glm::vec3& wo, const glm::vec2& uv, SampledWavelengths& swl) :wo(wo), uv(uv), swl(swl){}
};

class Material : public TaggedPointer<DiffuseMaterial, DielectricMaterial, ConductorMaterial, EmissiveMaterial>
{
public:
	using TaggedPointer::TaggedPointer;
	static Material create(const std::string& name, const MaterialParams& params, Allocator alloc);
	
	// Assume local allocated size is greater than any of the bxdf class
	__device__ BxDF get_bxdf(MaterialEvalInfo& info, void* localMem);
};

class DiffuseMaterial
{
public:
	static DiffuseMaterial* create(const MaterialParams& params, Allocator alloc)
	{
		glm::vec3  albedo = params.get_vec3("albedo");
		cudaTextureObject_t albedoMap = params.get_texture("albedoMap");
		RGBColorSpace* colorSpace = (RGBColorSpace*)params.get_ptr("colorSpace");
		if (!colorSpace)
			throw std::runtime_error("No color space specified for DiffuseMaterial");
		return alloc.new_object<DiffuseMaterial>(albedo, albedoMap, colorSpace);
	}
	DiffuseMaterial(const glm::vec3& albedo, cudaTextureObject_t albedoMap, RGBColorSpace* colorSpace):albedo(albedo), albedoMap(albedoMap), colorSpace(colorSpace){}
	__device__ BxDF get_bxdf(MaterialEvalInfo& info, void* localMem);
private:
	glm::vec3   albedo = glm::vec3(0.5f);
	cudaTextureObject_t albedoMap = 0;
	RGBColorSpace* colorSpace;
};

class DielectricMaterial
{
public:
	static DielectricMaterial* create(const MaterialParams& params, Allocator alloc)
	{
		Spectrum eta = params.get_spec("eta");
		return alloc.new_object<DielectricMaterial>(eta);
	}
	DielectricMaterial(Spectrum eta):eta(eta){}
	__device__ BxDF get_bxdf(MaterialEvalInfo& info, void* localMem)
	{
		DielectricBxDF* bxdfPtr = (DielectricBxDF*)localMem;
		if (!eta) return nullptr;
		float sampledEta = eta(info.swl[0]);
		sampledEta = info.wo.z < 0 ? 1.0 / sampledEta : sampledEta;
		if (!eta.template Is<ConstantSpectrum>())
		{
			info.swl.terminate_secondary();
		}
		bxdfPtr->eta = sampledEta;
		return bxdfPtr;
	}
private:
	Spectrum eta;
};

class ConductorMaterial
{
public:
	static ConductorMaterial* create(const MaterialParams& params, Allocator alloc)
	{
		Spectrum eta = params.get_spec("eta");
		Spectrum k = params.get_spec("k");
		float roughness = params.get_float("roughness");
		return alloc.new_object<ConductorMaterial>(eta, k, roughness);
	}
	ConductorMaterial(Spectrum eta, Spectrum k, float roughness):eta(eta),k(k), roughness(roughness){}
	__device__ BxDF get_bxdf(MaterialEvalInfo& info, void* localMem)
	{
		ConductorBxDF* bxdfPtr = (ConductorBxDF*)localMem;
		if (!eta || !k) return nullptr;
		SampledSpectrum sampledEta = eta.sample(info.swl);
		SampledSpectrum sampledK = k.sample(info.swl);
		bxdfPtr->eta = sampledEta;
		bxdfPtr->k = sampledK;
		bxdfPtr->roughness = roughness;
		return bxdfPtr;
	}
private:
	Spectrum eta, k;
	float roughness;
};

// Diffuse emissive material
class EmissiveMaterial
{
public:
	static EmissiveMaterial* create(const MaterialParams& params, Allocator alloc)
	{
		glm::vec3 albedo = params.get_vec3("albedo");
		float emittance = params.get_float("emittance");
		RGBColorSpace* colorSpace = (RGBColorSpace*)params.get_ptr("colorSpace");
		if (!colorSpace)
			throw std::runtime_error("No color space specified for EmissiveMaterial");
		return alloc.new_object<EmissiveMaterial>(albedo * emittance, colorSpace);
	}
	EmissiveMaterial(const glm::vec3& rgb, RGBColorSpace* colorSpace):rgb(rgb), colorSpace(colorSpace){}
	//should not be called
	__device__ BxDF get_bxdf(MaterialEvalInfo& info, void* localMem)
	{
		return nullptr;
	}
	__device__ SampledSpectrum Le(SampledWavelengths& swl)
	{
		RGBIlluminantSpectrum illum(*colorSpace, rgb);
		return illum.sample(swl);
	}
private:
	glm::vec3 rgb;
	RGBColorSpace* colorSpace;
};

constexpr uint32_t BxDFMaxSize = std::max({ sizeof(DiffuseBxDF), sizeof(DielectricBxDF), sizeof(ConductorMaterial), sizeof(EmissiveMaterial)});