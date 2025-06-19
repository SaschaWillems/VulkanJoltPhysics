/* Copyright (c) 2025, Sascha Willems
 *
 * SPDX-License-Identifier: MIT
 *
 */

#include <SFML/Graphics.hpp>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <vector>
#include <array>
#include <string>
#include <iostream>
#include <thread>
#include <fstream>
#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>
#include "slang/slang.h"
#include "slang/slang-com-ptr.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "physicsworld.h"

static inline void chk(VkResult result) {
	if (result != VK_SUCCESS) {
		std::cerr << "Call returned an error\n";
		exit(result);
	}
}

static inline void chk(HRESULT result) {
	if (FAILED(result)) {
		std::cerr << "Call returned an error\n";
		exit(result);
	}
}

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

const uint32_t maxFramesInFlight{ 2 };
const vk::SampleCountFlagBits sampleCount = vk::SampleCountFlagBits::e4;
uint32_t imageIndex{ 0 };
uint32_t frameIndex{ 0 };
uint32_t semaphoreIndex{ 0 };
vk::Result result;
vk::Instance instance;
vk::Device device;
vk::Queue queue;
vk::SurfaceKHR surface{ VK_NULL_HANDLE };
vk::SwapchainKHR swapchain{ VK_NULL_HANDLE };
vk::CommandPool commandPool{ VK_NULL_HANDLE };
vk::Pipeline pipeline;
vk::PipelineLayout pipelineLayout;
vk::Image renderImage;
vk::ImageView renderImageView;
VmaAllocation renderImageAllocation;
vk::Image depthImage;
vk::ImageView depthImageView;
VmaAllocation depthImageAllocation;
std::vector<vk::Image> swapchainImages;
std::vector<vk::ImageView> swapchainImageViews;
std::vector<vk::CommandBuffer> commandBuffers(maxFramesInFlight);
std::vector<vk::Fence> fences(maxFramesInFlight);
std::vector<vk::Semaphore> presentSemaphores;
std::vector<vk::Semaphore> renderSemaphores;
VmaAllocator allocator{ VK_NULL_HANDLE };
VmaAllocation vBufferAllocation{ VK_NULL_HANDLE };
VmaAllocation iBufferAllocation{ VK_NULL_HANDLE };
VmaAllocation sBufferAllocation{ VK_NULL_HANDLE };
VmaAllocation uBufferAllocation{ VK_NULL_HANDLE };
VmaAllocationInfo sBufferAllocInfo{};
VmaAllocationInfo uBufferAllocInfo{};
vk::Buffer vBuffer{ VK_NULL_HANDLE };
vk::Buffer iBuffer{ VK_NULL_HANDLE };
vk::Buffer sBuffer{ VK_NULL_HANDLE };
vk::Buffer uBuffer{ VK_NULL_HANDLE };
vk::DescriptorPool descPool{ VK_NULL_HANDLE };
vk::DescriptorSetLayout sceneDescLayout{ VK_NULL_HANDLE };
vk::DescriptorSet sceneDescSet{ VK_NULL_HANDLE };
Slang::ComPtr<slang::IGlobalSession> slangGlobalSession;

JPH::PhysicsSystem physicsSystem;
PhysicsWorld::BPLayerInterfaceImpl broad_phase_layer_interface;
PhysicsWorld::ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;
PhysicsWorld::ObjectLayerPairFilterImpl object_vs_object_layer_filter;
JPH::Body* floorBody;

const uint32_t physMaxBodies = 1024;
const uint32_t physNumBodyMutexes = 0;
const uint32_t physMaxBodyPairs = 1024;
const uint32_t physMaxContactConstraints = 1024;
const uint32_t physCollisionSteps = 1;

JPH::Vec3 cubeDim{ 0.5, 1.5, 1.0 };

struct ObjectShaderData {
	JPH::Mat44 model;
};
std::vector<ObjectShaderData> objectShaderData;

struct SceneShaderData
{
	glm::mat4 projection;
	glm::mat4 view;
} sceneShaderData;

struct Camera
{
	glm::vec3 position{ 0.0f, 0.0f, -2.5f };
	glm::vec3 rotation{ 45.0f, 0.0f, 0.0f };
} camera;

bool paused{ false };

// Unit cube
float vertices[] = {
	// Front
	-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
	 0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
	 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,
	-0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,

	// Back
	-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
	 0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
	 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,
	-0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,

	// Left
	-0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
	-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,
	-0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,
	-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,

	// Right
	 0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
	 0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,
	 0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,
	 0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,

	 // Top
	 -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
	  0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,
	  0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,
	 -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,

	 // Bottom
	 -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
	  0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,
	  0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,
	 -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f
};

uint32_t indices[] = {
	0,1,2,		2,3,0,		// F
	4,5,6,		6,7,4,		// B
	8,9,10,		10,11,8,	// L
	12,13,14,	14,15,12,	// R
	16,17,18,	18,19,16,	// T
	20,21,22,	22,23,20	// B
};

void updatePerspective(sf::RenderWindow& window)
{
	sceneShaderData.projection = glm::perspective(glm::radians(60.0f), (float)static_cast<uint32_t>(window.getSize().x) / (float)static_cast<uint32_t>(window.getSize().y), 0.1f, 512.0f);
}

void updateViewMatrix()
{
	glm::mat4 rotM = glm::mat4(1.0f);
	glm::mat4 transM;
	rotM = glm::rotate(rotM, glm::radians(camera.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
	rotM = glm::rotate(rotM, glm::radians(camera.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
	rotM = glm::rotate(rotM, glm::radians(camera.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
	transM = glm::translate(glm::mat4(1.0f), camera.position);
	sceneShaderData.view = transM * rotM;
};

int main()
{
	// Setup
	auto window = sf::RenderWindow(sf::VideoMode({ 1280, 720u }), "Modern Vulkan Triangle");
	updatePerspective(window);
	// Jolt physics
	JPH::RegisterDefaultAllocator();
	JPH::Factory::sInstance = new JPH::Factory();
	JPH::RegisterTypes();
	JPH::TempAllocatorImpl temp_allocator(10 * 1024 * 1024);
	JPH::JobSystemThreadPool job_system(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, std::thread::hardware_concurrency() - 1);
	physicsSystem.Init(physMaxBodies, physMaxBodies, physMaxBodies, physMaxBodies, broad_phase_layer_interface, object_vs_broadphase_layer_filter, object_vs_object_layer_filter);
	// @todo: just for testing
	physicsSystem.SetGravity(JPH::Vec3::sZero());
	PhysicsWorld::MyBodyActivationListener body_activation_listener;
	physicsSystem.SetBodyActivationListener(&body_activation_listener);
	PhysicsWorld::MyContactListener contact_listener;
	physicsSystem.SetContactListener(&contact_listener);
	JPH::BodyInterface& body_interface = physicsSystem.GetBodyInterface();
	// Physics world
	JPH::BoxShapeSettings body_shape_settings(cubeDim);
	body_shape_settings.mConvexRadius = 0.01;
	body_shape_settings.SetDensity(1000.0);
	body_shape_settings.SetEmbedded();
	JPH::ShapeSettings::ShapeResult body_shape_result = body_shape_settings.Create();
	JPH::ShapeRefC body_shape = body_shape_result.Get();
	JPH::BodyCreationSettings body_settings(body_shape, JPH::RVec3(0.0, 0.0, 0.0), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, PhysicsWorld::Layers::MOVING);
	body_settings.mMaxLinearVelocity = 10000.0;
	body_settings.mApplyGyroscopicForce = true;
	body_settings.mLinearDamping = 0.0;
	body_settings.mAngularDamping = 0.1;
	JPH::Body* body = body_interface.CreateBody(body_settings);
	body_interface.AddBody(body->GetID(), JPH::EActivation::Activate);
	body_interface.SetLinearVelocity(body->GetID(), JPH::Vec3(0.0, 0.0, 0.0));
	// @todo: test
	//body_interface.SetAngularVelocity(body->GetID(), JPH::Vec3(0.0, 0.25, 0.0));
	body_interface.SetAngularVelocity(body->GetID(), JPH::Vec3(0.3, 0.0, 5.0));

	physicsSystem.OptimizeBroadPhase();

	/*
	JPH::BoxShapeSettings floor_shape_settings(JPH::Vec3(100.0f, 1.0f, 100.0f));
	floor_shape_settings.SetEmbedded();
	JPH::ShapeSettings::ShapeResult floor_shape_result = floor_shape_settings.Create();
	JPH::ShapeRefC floor_shape = floor_shape_result.Get();
	JPH::BodyCreationSettings floor_settings(floor_shape, JPH::RVec3(0.0, -1.0, 0.0), JPH::Quat::sIdentity(), JPH::EMotionType::Static, PhysicsWorld::Layers::NON_MOVING);
	floorBody = body_interface.CreateBody(floor_settings);
	body_interface.AddBody(floorBody->GetID(), JPH::EActivation::DontActivate);
	*/

	// Physics / GFX interaction
	objectShaderData.resize(1);
	objectShaderData[0].model = JPH::Mat44::sIdentity();

	// Initialize slang compiler
	slang::createGlobalSession(slangGlobalSession.writeRef());
	auto targets{ std::to_array<slang::TargetDesc>({ {.format{SLANG_SPIRV}, .profile{slangGlobalSession->findProfile("spirv_1_6")} } }) };
	auto options{ std::to_array<slang::CompilerOptionEntry>({ { slang::CompilerOptionName::EmitSpirvDirectly, {slang::CompilerOptionValueKind::Int, 1} } }) };
	slang::SessionDesc desc{ .targets{targets.data()}, .targetCount{SlangInt(targets.size())}, .defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR, .compilerOptionEntries{options.data()}, .compilerOptionEntryCount{uint32_t(options.size())} };
	Slang::ComPtr<slang::ISession> slangSession;
	slangGlobalSession->createSession(desc, slangSession.writeRef());

	// Instance
	VULKAN_HPP_DEFAULT_DISPATCHER.init();
	vk::ApplicationInfo appInfo{ .pApplicationName = "Modern Vulkan Triangle", .apiVersion = VK_API_VERSION_1_3 };
	const std::vector<const char*> instanceExtensions{ VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME, };
	vk::InstanceCreateInfo instanceCI{ .pApplicationInfo = &appInfo, .enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size()), .ppEnabledExtensionNames = instanceExtensions.data() };
	instance = vk::createInstance(instanceCI);
	VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
	// Device
	uint32_t deviceCount{ 0 };
	std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
	const uint32_t qf{ 0 };
	const float qfpriorities{ 1.0f };
	const uint32_t deviceIndex{ 0 };
	vk::DeviceQueueCreateInfo queueCI{ .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &qfpriorities };
	vk::PhysicalDeviceVulkan13Features features{ .dynamicRendering = true };
	const std::vector<const char*> deviceExtensions{ VK_KHR_SWAPCHAIN_EXTENSION_NAME };
	vk::DeviceCreateInfo deviceCI{ .pNext = &features, .queueCreateInfoCount = 1, .pQueueCreateInfos = &queueCI, .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()), .ppEnabledExtensionNames = deviceExtensions.data() };
	device = physicalDevices[deviceIndex].createDevice(deviceCI);
	queue = device.getQueue(qf, 0);
	// VMA
	VmaVulkanFunctions vkFunctions{ .vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr, .vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr, .vkCreateImage = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage };
	VmaAllocatorCreateInfo allocatorCI{ .physicalDevice = physicalDevices[deviceIndex], .device = device, .pVulkanFunctions = &vkFunctions, .instance = instance };
	chk(vmaCreateAllocator(&allocatorCI, &allocator));
	// Presentation
	VkSurfaceKHR _surface;
	chk(window.createVulkanSurface(static_cast<VkInstance>(instance), _surface));
	surface = vk::SurfaceKHR(_surface);
	const vk::Format imageFormat{ vk::Format::eB8G8R8A8Srgb };
	vk::SwapchainCreateInfoKHR swapchainCI{
		.surface = surface,
		.minImageCount = 2,
		.imageFormat = imageFormat,
		.imageExtent = {.width = window.getSize().x, .height = window.getSize().y, },
		.imageArrayLayers = 1,
		.imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
		.presentMode = vk::PresentModeKHR::eFifo,
	};
	swapchain = device.createSwapchainKHR(swapchainCI);
	swapchainImages = device.getSwapchainImagesKHR(swapchain);
	vk::ImageCreateInfo renderImageCI{ .imageType = vk::ImageType::e2D, .format = imageFormat, .extent = {.width = window.getSize().x, .height = window.getSize().y, .depth = 1 }, .mipLevels = 1, .arrayLayers = 1, .samples = sampleCount, .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc };
	VmaAllocationCreateInfo allocCI{ .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, .usage = VMA_MEMORY_USAGE_AUTO, .priority = 1.0f };
	vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&renderImageCI), &allocCI, reinterpret_cast<VkImage*>(&renderImage), &renderImageAllocation, nullptr);
	vk::ImageViewCreateInfo viewCI{ .image = renderImage, .viewType = vk::ImageViewType::e2D, .format = imageFormat, .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1 } };
	renderImageView = device.createImageView(viewCI);
	swapchainImageViews.resize(swapchainImages.size());
	for (auto i = 0; i < swapchainImages.size(); i++) {
		viewCI.image = swapchainImages[i];
		swapchainImageViews[i] = device.createImageView(viewCI);
	}
	// Depth
	const vk::Format depthFormat{ vk::Format::eD32Sfloat };
	vk::ImageCreateInfo depthImageCI{ .imageType = vk::ImageType::e2D, .format = depthFormat, .extent = {.width = window.getSize().x, .height = window.getSize().y, .depth = 1 }, .mipLevels = 1, .arrayLayers = 1, .samples = sampleCount, .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment };
	VmaAllocationCreateInfo depthAllocCI{ .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, .usage = VMA_MEMORY_USAGE_AUTO, .priority = 1.0f };
	vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&depthImageCI), &depthAllocCI, reinterpret_cast<VkImage*>(&depthImage), &depthImageAllocation, nullptr);
	vk::ImageViewCreateInfo depthViewCI{ .image = depthImage, .viewType = vk::ImageViewType::e2D, .format = depthFormat, .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth, .levelCount = 1, .layerCount = 1 } };
	depthImageView = device.createImageView(depthViewCI);

	// Buffers
	// @todo: vert and idx single buffer

	VkBufferCreateInfo bufferCI{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizeof(vertices), .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT };
	VmaAllocationCreateInfo bufferAllocCI{ .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, .usage = VMA_MEMORY_USAGE_AUTO };
	VmaAllocationInfo bufferAllocInfo{};

	chk(vmaCreateBuffer(allocator, &bufferCI, &bufferAllocCI, reinterpret_cast<VkBuffer*>(&vBuffer), &vBufferAllocation, &bufferAllocInfo));
	memcpy(bufferAllocInfo.pMappedData, vertices, sizeof(vertices));

	bufferCI.size = sizeof(indices);
	chk(vmaCreateBuffer(allocator, &bufferCI, &bufferAllocCI, reinterpret_cast<VkBuffer*>(&iBuffer), &iBufferAllocation, &bufferAllocInfo));
	memcpy(bufferAllocInfo.pMappedData, indices, sizeof(indices));

	bufferCI.size = sizeof(ObjectShaderData) * objectShaderData.size();
	bufferCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	chk(vmaCreateBuffer(allocator, &bufferCI, &bufferAllocCI, reinterpret_cast<VkBuffer*>(&sBuffer), &sBufferAllocation, &sBufferAllocInfo));
	memcpy(sBufferAllocInfo.pMappedData, objectShaderData.data(), sizeof(ObjectShaderData) * objectShaderData.size());

	bufferCI.size = sizeof(SceneShaderData);
	bufferCI.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	chk(vmaCreateBuffer(allocator, &bufferCI, &bufferAllocCI, reinterpret_cast<VkBuffer*>(&uBuffer), &uBufferAllocation, &uBufferAllocInfo));
	memcpy(uBufferAllocInfo.pMappedData, &sceneShaderData, sizeof(SceneShaderData));

	// Descriptors
	std::vector<vk::DescriptorPoolSize> descPoolSizes = {
		{ .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1 },
		{ .type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1 }
	};
	descPool = device.createDescriptorPool({ .maxSets = maxFramesInFlight, .poolSizeCount = 2, .pPoolSizes = descPoolSizes.data()});

	std::vector<vk::DescriptorSetLayoutBinding> layoutBindings{
		{ .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eAll},
		{ .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eAll},
	};
	sceneDescLayout = device.createDescriptorSetLayout({ .bindingCount = 2, .pBindings = layoutBindings.data()});
	sceneDescSet = device.allocateDescriptorSets({ .descriptorPool = descPool, .descriptorSetCount = 1, .pSetLayouts = &sceneDescLayout })[0];
	vk::DescriptorBufferInfo sBufferInfo{ .buffer = sBuffer, .offset = 0, .range = sizeof(ObjectShaderData) * objectShaderData.size() };
	vk::DescriptorBufferInfo uBufferInfo{ .buffer = uBuffer, .offset = 0, .range = sizeof(SceneShaderData) };

	std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
		{ .dstSet = sceneDescSet, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &uBufferInfo },
		{ .dstSet = sceneDescSet, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &sBufferInfo }
	};
	device.updateDescriptorSets(writeDescriptorSets, {});

	// Sync objects
	commandPool = device.createCommandPool({ .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, .queueFamilyIndex = qf });
	commandBuffers = device.allocateCommandBuffers({ .commandPool = commandPool, .commandBufferCount = maxFramesInFlight });
	for (auto i = 0; i < maxFramesInFlight; i++) {
		fences[i] = device.createFence({ .flags = vk::FenceCreateFlagBits::eSignaled });
	}
	presentSemaphores.resize(swapchainImages.size());
	renderSemaphores.resize(swapchainImages.size());
	for (auto i = 0; i < swapchainImages.size(); i++) {
		presentSemaphores[i] = device.createSemaphore({});
		renderSemaphores[i] = device.createSemaphore({});
	}
	// Shaders
	std::ifstream shaderFile("shaders/base.slang");
	if (!shaderFile) {
		fprintf(stderr, "Could not load shader file");
		exit(-1);
	}
	std::string shaderSrc { std::istreambuf_iterator<char>(shaderFile), std::istreambuf_iterator<char>() };
	Slang::ComPtr<slang::IBlob> diagnostics;
	Slang::ComPtr<slang::IModule> slangModule{ slangSession->loadModuleFromSourceString("triangle", nullptr, shaderSrc.c_str(), diagnostics.writeRef()) };
	Slang::ComPtr<ISlangBlob> spirv;
	if (diagnostics)
	{
		fprintf(stderr, "%s\n", (const char*)diagnostics->getBufferPointer());
		exit(-1);
	}
	slangModule->getTargetCode(0, spirv.writeRef());
	vk::ShaderModule shaderModule = device.createShaderModule({ .codeSize = spirv->getBufferSize(), .pCode = (uint32_t*)spirv->getBufferPointer() });
	std::vector<vk::PipelineShaderStageCreateInfo> stages{
		{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "main"},
		{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "main" }
	};
	// Pipeline
	pipelineLayout = device.createPipelineLayout({ .setLayoutCount = 1, .pSetLayouts = &sceneDescLayout });
	vk::VertexInputBindingDescription vertexBinding{ .binding = 0, .stride = sizeof(float) * 6, .inputRate = vk::VertexInputRate::eVertex };
	std::vector<vk::VertexInputAttributeDescription> vertexAttributes{
		{.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat },
		{.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = sizeof(float) * 3},
	};
	vk::PipelineVertexInputStateCreateInfo vertexInputState{ .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vertexBinding, .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = vertexAttributes.data(), };
	vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState{ .topology = vk::PrimitiveTopology::eTriangleList };
	vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };
	vk::PipelineRasterizationStateCreateInfo rasterizationState{ .cullMode = vk::CullModeFlagBits::eNone, .lineWidth = 1.0f };
	vk::PipelineMultisampleStateCreateInfo multisampleState{ .rasterizationSamples = sampleCount };
	vk::PipelineDepthStencilStateCreateInfo depthStencilState{ .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE, .depthCompareOp = vk::CompareOp::eLessOrEqual };
	vk::PipelineColorBlendAttachmentState blendAttachment{ .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
	vk::PipelineColorBlendStateCreateInfo colorBlendState{ .attachmentCount = 1, .pAttachments = &blendAttachment };
	std::vector<vk::DynamicState> dynamicStates{ vk::DynamicState::eViewport, vk::DynamicState::eScissor };
	vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = 2, .pDynamicStates = dynamicStates.data() };
	vk::PipelineRenderingCreateInfo renderingCI{ .colorAttachmentCount = 1, .pColorAttachmentFormats = &imageFormat, .depthAttachmentFormat = depthFormat };
	vk::GraphicsPipelineCreateInfo pipelineCI{
		.pNext = &renderingCI,
		.stageCount = 2,
		.pStages = stages.data(),
		.pVertexInputState = &vertexInputState,
		.pInputAssemblyState = &inputAssemblyState,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterizationState,
		.pMultisampleState = &multisampleState,
		.pDepthStencilState = &depthStencilState,
		.pColorBlendState = &colorBlendState,
		.pDynamicState = &dynamicState,
		.layout = pipelineLayout,
	};
	std::tie(result, pipeline) = device.createGraphicsPipeline(nullptr, pipelineCI);
	device.destroyShaderModule(shaderModule, nullptr);
	// Render
	sf::Clock clock;
	while (window.isOpen())
	{
		sf::Time dT = clock.restart();
		// Build CB
		device.waitForFences(fences[frameIndex], true, UINT64_MAX);
		device.resetFences(fences[frameIndex]);
		device.acquireNextImageKHR(swapchain, UINT64_MAX, presentSemaphores[semaphoreIndex], VK_NULL_HANDLE, &imageIndex);
		updateViewMatrix();
		memcpy(uBufferAllocInfo.pMappedData, &sceneShaderData, sizeof(SceneShaderData));
		auto& cb = commandBuffers[frameIndex];
		cb.reset();
		cb.begin({ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
		vk::ImageMemoryBarrier barrier0{
			.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
			.oldLayout = vk::ImageLayout::eUndefined,
			.newLayout = vk::ImageLayout::eGeneral,
			.image = renderImage,
			.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1 }
		};
		cb.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::DependencyFlags{ 0 }, nullptr, nullptr, barrier0);
		vk::RenderingAttachmentInfo colorAttachmentInfo{
			.imageView = renderImageView,
			.imageLayout = vk::ImageLayout::eGeneral,
			.resolveMode = vk::ResolveModeFlagBits::eAverage,
			.resolveImageView = swapchainImageViews[imageIndex],
			.resolveImageLayout = vk::ImageLayout::eGeneral,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = vk::ClearValue{ vk::ClearColorValue{ std::array<float, 4>{0.0f, 0.0f, 0.2f, 1.0f} } },
		};
		vk::RenderingAttachmentInfo depthAttachmentInfo{
			.imageView = depthImageView,
			.imageLayout = vk::ImageLayout::eGeneral,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = vk::ClearValue{ vk::ClearDepthStencilValue{ .depth = 1.0f } },
		};
		vk::RenderingInfo renderingInfo{
			.renderArea = {.extent = {.width = window.getSize().x, .height = window.getSize().y }}, .layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorAttachmentInfo,
			.pDepthAttachment =&depthAttachmentInfo,
		};
		cb.beginRendering(renderingInfo);
		vk::Viewport vp{ .width = static_cast<float>(window.getSize().x), .height = static_cast<float>(window.getSize().y), .minDepth = 0.0f, .maxDepth = 1.0f };
		cb.setViewport(0, 1, &vp);
		cb.setScissor(0, 1, &renderingInfo.renderArea);
		cb.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, { sceneDescSet }, {});
		vk::DeviceSize vOffset{ 0 };
		cb.bindVertexBuffers(0, 1, &vBuffer, &vOffset);
		cb.bindIndexBuffer(iBuffer, vOffset, vk::IndexType::eUint32);
		cb.drawIndexed(sizeof(indices) / sizeof(indices[0]), 1, 0, 0, 0);
		cb.endRendering();
		vk::ImageMemoryBarrier barrier1{
			.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
			.oldLayout = vk::ImageLayout::eUndefined,
			.newLayout = vk::ImageLayout::ePresentSrcKHR,
			.image = swapchainImages[imageIndex],
			.subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1 }
		};
		cb.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe, vk::DependencyFlags{ 0 }, nullptr, nullptr, barrier1);
		cb.end();
		// Submit
		vk::PipelineStageFlags waitStages = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		vk::SubmitInfo submitInfo{
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &presentSemaphores[semaphoreIndex],
			.pWaitDstStageMask = &waitStages,
			.commandBufferCount = 1,
			.pCommandBuffers = &cb,
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &renderSemaphores[semaphoreIndex],
		};
		queue.submit(submitInfo, fences[frameIndex]);
		queue.presentKHR({ .waitSemaphoreCount = 1, .pWaitSemaphores = &renderSemaphores[semaphoreIndex], .swapchainCount = 1, .pSwapchains = &swapchain, .pImageIndices = &imageIndex });
		frameIndex++;
		if (frameIndex >= maxFramesInFlight) { frameIndex = 0; }
		frameIndex = (frameIndex + 1) % maxFramesInFlight;
		semaphoreIndex = (semaphoreIndex + 1) % static_cast<uint32_t>(swapchainImages.size());
		while (const std::optional event = window.pollEvent()) {
			if (event->is<sf::Event::Closed>()) {
				window.close();
			}
			if (event->is<sf::Event::KeyPressed>()) {
				const auto* keyPressed = event->getIf<sf::Event::KeyPressed>();
				if (keyPressed->scancode == sf::Keyboard::Scancode::P) {
					paused = !paused;
				}
				if (keyPressed->scancode == sf::Keyboard::Scancode::A) {
					body->AddAngularImpulse(JPH::Vec3(30.0, 0.0, 75.0));
				}
			}
			if (event->is<sf::Event::Resized>()) {
				device.waitIdle();
				swapchainCI.oldSwapchain = swapchain;
				swapchainCI.imageExtent = { .width = static_cast<uint32_t>(window.getSize().x), .height = static_cast<uint32_t>(window.getSize().y) };
				swapchain = device.createSwapchainKHR(swapchainCI);
				swapchainImages = device.getSwapchainImagesKHR(swapchain);
				vmaDestroyImage(allocator, renderImage, renderImageAllocation);
				device.destroyImageView(renderImageView, nullptr);
				for (auto i = 0; i < swapchainImageViews.size(); i++) {
					device.destroyImageView(swapchainImageViews[i], nullptr);
				}
				swapchainImageViews.resize(swapchainImages.size());
				renderImageCI.extent = { .width = static_cast<uint32_t>(window.getSize().x), .height = static_cast<uint32_t>(window.getSize().y), .depth = 1 };
				VmaAllocationCreateInfo allocCI = { .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, .usage = VMA_MEMORY_USAGE_AUTO, .priority = 1.0f };
				chk(vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&renderImageCI), &allocCI, reinterpret_cast<VkImage*>(&renderImage), &renderImageAllocation, nullptr));
				vk::ImageViewCreateInfo viewCI = { .image = renderImage, .viewType = vk::ImageViewType::e2D, .format = imageFormat, .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1 } };
				renderImageView = device.createImageView(viewCI);
				for (auto i = 0; i < swapchainImages.size(); i++) {
					viewCI.image = swapchainImages[i];
					swapchainImageViews[i] = device.createImageView(viewCI);
				}
				device.destroySwapchainKHR(swapchainCI.oldSwapchain, nullptr);
				depthImageCI.extent = { .width = static_cast<uint32_t>(window.getSize().x), .height = static_cast<uint32_t>(window.getSize().y), .depth = 1 };
				vmaDestroyImage(allocator, depthImage, depthImageAllocation);
				chk(vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&depthImageCI), &depthAllocCI, reinterpret_cast<VkImage*>(&depthImage), &depthImageAllocation, nullptr));
				vk::ImageViewCreateInfo depthViewCI{ .image = depthImage, .viewType = vk::ImageViewType::e2D, .format = depthFormat, .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth, .levelCount = 1, .layerCount = 1 } };
				depthImageView = device.createImageView(depthViewCI);

				updatePerspective(window);
			}
		}
		// Update world
		// @todo: buffers per frame
		if (!paused) {
			physicsSystem.Update(dT.asSeconds(), physCollisionSteps, &temp_allocator, &job_system);
		}
		JPH::Mat44 scaleMat = JPH::Mat44::sIdentity().PostScaled(cubeDim);
		objectShaderData[0].model = body_interface.GetWorldTransform(body->GetID()) * scaleMat;
		memcpy(sBufferAllocInfo.pMappedData, objectShaderData.data(), sizeof(ObjectShaderData) * objectShaderData.size());
	}
	// Tear down
	device.waitIdle();
	for (auto i = 0; i < maxFramesInFlight; i++) {
		device.destroyFence(fences[i], nullptr);
	}
	for (auto i = 0; i < presentSemaphores.size(); i++) {
		device.destroySemaphore(presentSemaphores[i], nullptr);
	}
	for (auto i = 0; i < renderSemaphores.size(); i++) {
		device.destroySemaphore(renderSemaphores[i], nullptr);
	}
	vmaDestroyImage(allocator, renderImage, renderImageAllocation);
	device.destroyImageView(renderImageView, nullptr);
	vmaDestroyImage(allocator, depthImage, depthImageAllocation);
	device.destroyImageView(depthImageView, nullptr);
	for (auto i = 0; i < swapchainImageViews.size(); i++) {
		device.destroyImageView(swapchainImageViews[i], nullptr);
	}
	vmaDestroyBuffer(allocator, vBuffer, vBufferAllocation);
	vmaDestroyBuffer(allocator, iBuffer, iBufferAllocation);
	vmaDestroyBuffer(allocator, sBuffer, sBufferAllocation);
	vmaDestroyBuffer(allocator, uBuffer, uBufferAllocation);
	device.destroyCommandPool(commandPool, nullptr);
	device.destroyPipelineLayout(pipelineLayout, nullptr);
	device.destroyPipeline(pipeline, nullptr);
	device.destroySwapchainKHR(swapchain, nullptr);
	vmaDestroyAllocator(allocator);
	device.destroy();
	instance.destroySurfaceKHR(surface, nullptr);
	instance.destroy();
	// Jolt
	JPH::UnregisterTypes();
	// Destroy the factory
	delete JPH::Factory::sInstance;
	JPH::Factory::sInstance = nullptr;
}