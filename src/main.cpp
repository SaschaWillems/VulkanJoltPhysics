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
#include "camera.hpp"

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

struct ShaderBuffer {
	vk::Buffer buffer;
	VmaAllocation alloc;
	VmaAllocationInfo allocInfo;
};

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
vk::Buffer vBuffer{ VK_NULL_HANDLE };
vk::Buffer iBuffer{ VK_NULL_HANDLE };
vk::DescriptorPool descPool{ VK_NULL_HANDLE };
vk::DescriptorSetLayout sceneDescLayout{ VK_NULL_HANDLE };
std::vector<ShaderBuffer> objectDataBuffers{};
std::vector<ShaderBuffer> uniformBuffers{};
std::vector<vk::DescriptorSet> descriptorSets;

Slang::ComPtr<slang::IGlobalSession> slangGlobalSession;

JPH::PhysicsSystem physicsSystem;
PhysicsWorld::BPLayerInterfaceImpl broad_phase_layer_interface;
PhysicsWorld::ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;
PhysicsWorld::ObjectLayerPairFilterImpl object_vs_object_layer_filter;
JPH::Body* floorBody;

const uint32_t physMaxBodies = 4096;
const uint32_t physNumBodyMutexes = 0;
const uint32_t physMaxBodyPairs = 1024;
const uint32_t physMaxContactConstraints = 1024;
const uint32_t physCollisionSteps = 1;

struct ObjectShaderData {
	JPH::Mat44 model;
	glm::vec4 color;
};
std::vector<ObjectShaderData> objectShaderData;

struct SceneShaderData
{
	glm::mat4 projection;
	glm::mat4 view;
} sceneShaderData;

bool paused{ true };

Camera camera;
sf::Vector2f mousePosition;

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
	camera.setPerspective(60.0f, (float)static_cast<uint32_t>(window.getSize().x) / (float)static_cast<uint32_t>(window.getSize().y), 0.1f, 512.0f);
	sceneShaderData.projection = camera.matrices.perspective;
}

void updateViewMatrix(float dT)
{
	camera.update(dT);
	sceneShaderData.view = camera.matrices.view;
};

int main()
{
	// Setup
	bool fullscreen = false;
	auto window = sf::RenderWindow(sf::VideoMode({ 1920u, 1080u }), "Vulkan Jolt Physics Playground", fullscreen ? sf::State::Fullscreen : sf::State::Windowed);
	updatePerspective(window);
	camera.setPosition({ 0.0f, 5.0f, -25.0f });
	camera.setRotation({ 0.0f, 0.0f, 0.0f });
	camera.movementSpeed = 10.0f;
	camera.rotationSpeed = 0.25f;

	// Jolt physics
	JPH::RegisterDefaultAllocator();
	JPH::Factory::sInstance = new JPH::Factory();
	JPH::RegisterTypes();
	JPH::TempAllocatorImpl temp_allocator(10 * 1024 * 1024);
	JPH::JobSystemThreadPool job_system(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, std::thread::hardware_concurrency() - 1);
	physicsSystem.Init(physMaxBodies, physMaxBodies, physMaxBodies, physMaxBodies, broad_phase_layer_interface, object_vs_broadphase_layer_filter, object_vs_object_layer_filter);
	physicsSystem.SetGravity({ 0.0, 9.8, 0.0 });
	PhysicsWorld::MyBodyActivationListener body_activation_listener;
	physicsSystem.SetBodyActivationListener(&body_activation_listener);
	PhysicsWorld::MyContactListener contact_listener;
	physicsSystem.SetContactListener(&contact_listener);
	JPH::BodyInterface& body_interface = physicsSystem.GetBodyInterface();

	// Physics world
	PhysicsWorld::world = new PhysicsWorld::World();

	// Add some random cubes
	std::vector<glm::vec3> colors = {
		{ 0.0, 0.0, 1.0 }, { 0.0, 1.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 1.0 }, { 1.0, 1.0, 0.0 }, { 1.0, 1.0, 1.0 }, { 1.0, 0.0, 1.0 }
	};

	/*
	for (uint32_t i = 0; i < 20; i++) {
		const JPH::Vec3 dim{ 1.0, 1.0, 1.0 };
		JPH::BoxShapeSettings body_shape_settings(dim * 0.5);
		body_shape_settings.mConvexRadius = 0.01;
		body_shape_settings.SetDensity(250.0);
		body_shape_settings.SetEmbedded();
		JPH::ShapeSettings::ShapeResult body_shape_result = body_shape_settings.Create();
		JPH::ShapeRefC body_shape = body_shape_result.Get();
		JPH::BodyCreationSettings body_settings(body_shape, JPH::RVec3(0.0 + i * 0.15, -2.0 - i * dim.GetX() * 1.5, 0.0 + i * 0.15), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, PhysicsWorld::Layers::MOVING);
		uint32_t colorIndex = i % static_cast<uint32_t>(colors.size());
		auto newBody = PhysicsWorld::world->AddNewObject(body_interface.CreateBody(body_settings), dim, colors[colorIndex]);
		body_interface.AddBody(newBody->id, JPH::EActivation::Activate);
	}
	*/
	
	// Fixed floor
	JPH::Vec3 floorDim(100.0f, 0.1f, 100.0f);
	JPH::BoxShapeSettings floor_shape_settings(floorDim);
	floor_shape_settings.SetEmbedded();
	JPH::ShapeSettings::ShapeResult floor_shape_result = floor_shape_settings.Create();
	JPH::ShapeRefC floor_shape = floor_shape_result.Get();
	JPH::BodyCreationSettings floor_settings(floor_shape, JPH::RVec3(0.0, -0.5, 0.0), JPH::Quat::sIdentity(), JPH::EMotionType::Static, PhysicsWorld::Layers::NON_MOVING);
	auto floorBody = PhysicsWorld::world->AddNewObject(body_interface.CreateBody(floor_settings), floorDim, { 0.5, 0.5, 0.5 });
	body_interface.AddBody(floorBody->id, JPH::EActivation::DontActivate);

	physicsSystem.OptimizeBroadPhase();

	// Physics / GFX interaction
	objectShaderData.resize(physMaxBodies);

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

	VkBufferCreateInfo bufferCI{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizeof(vertices), .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT };
	VmaAllocationCreateInfo bufferAllocCI{ .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, .usage = VMA_MEMORY_USAGE_AUTO };
	VmaAllocationInfo bufferAllocInfo{};

	chk(vmaCreateBuffer(allocator, &bufferCI, &bufferAllocCI, reinterpret_cast<VkBuffer*>(&vBuffer), &vBufferAllocation, &bufferAllocInfo));
	memcpy(bufferAllocInfo.pMappedData, vertices, sizeof(vertices));

	bufferCI.size = sizeof(indices);
	chk(vmaCreateBuffer(allocator, &bufferCI, &bufferAllocCI, reinterpret_cast<VkBuffer*>(&iBuffer), &iBufferAllocation, &bufferAllocInfo));
	memcpy(bufferAllocInfo.pMappedData, indices, sizeof(indices));

	// SSBOs for all physics objects

	objectDataBuffers.resize(maxFramesInFlight);
	bufferCI.size = sizeof(ObjectShaderData) * physMaxBodies;
	bufferCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	for (auto& b : objectDataBuffers) {
		chk(vmaCreateBuffer(allocator, &bufferCI, &bufferAllocCI, reinterpret_cast<VkBuffer*>(&b.buffer), &b.alloc, &b.allocInfo));
	}

	// UBOs for scene matrices

	uniformBuffers.resize(maxFramesInFlight);
	bufferCI.size = sizeof(SceneShaderData);
	bufferCI.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	for (auto& b : uniformBuffers) {
		chk(vmaCreateBuffer(allocator, &bufferCI, &bufferAllocCI, reinterpret_cast<VkBuffer*>(&b.buffer), &b.alloc, &b.allocInfo));
	}

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

	descriptorSets.resize(maxFramesInFlight);
	for (auto i = 0; i < descriptorSets.size(); i++) {
		descriptorSets[i] = device.allocateDescriptorSets({.descriptorPool = descPool, .descriptorSetCount = 1, .pSetLayouts = &sceneDescLayout})[0];
		vk::DescriptorBufferInfo uBufferInfo{ .buffer = uniformBuffers[i].buffer, .offset = 0, .range = sizeof(SceneShaderData)};
		vk::DescriptorBufferInfo sBufferInfo{ .buffer = objectDataBuffers[i].buffer, .offset = 0, .range = sizeof(ObjectShaderData) * objectShaderData.size()};
		std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
			{.dstSet = descriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &uBufferInfo },
			{.dstSet = descriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &sBufferInfo }
		};
		device.updateDescriptorSets(writeDescriptorSets, {});
	}

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
	vk::PushConstantRange pushConstantRange{ .stageFlags = vk::ShaderStageFlagBits::eAll, .size = sizeof(uint32_t) };
	pipelineLayout = device.createPipelineLayout({ .setLayoutCount = 1, .pSetLayouts = &sceneDescLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange });
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
	mousePosition = sf::Vector2f(sf::Mouse::getPosition(window));
	while (window.isOpen())
	{
		sf::Time dT = clock.restart();
		
		// Input
		sf::Vector2 newMousePos = sf::Vector2f(sf::Mouse::getPosition(window));
		sf::Vector2f mouseDelta = mousePosition - newMousePos;
		mousePosition = newMousePos;
			
		// Keys
		camera.keys.up = sf::Keyboard::isKeyPressed(sf::Keyboard::Scancode::W);
		camera.keys.down = sf::Keyboard::isKeyPressed(sf::Keyboard::Scancode::S);
		camera.keys.left = sf::Keyboard::isKeyPressed(sf::Keyboard::Scancode::A);
		camera.keys.right = sf::Keyboard::isKeyPressed(sf::Keyboard::Scancode::D);
		
		// Mouse
		if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
			camera.rotate(glm::vec3(mouseDelta.y * camera.rotationSpeed, -mouseDelta.x * camera.rotationSpeed, 0.0f));
		}

		// Build CB
		device.waitForFences(fences[frameIndex], true, UINT64_MAX);
		device.resetFences(fences[frameIndex]);
		device.acquireNextImageKHR(swapchain, UINT64_MAX, presentSemaphores[semaphoreIndex], VK_NULL_HANDLE, &imageIndex);
		updateViewMatrix(dT.asSeconds());
		memcpy(uniformBuffers[frameIndex].allocInfo.pMappedData, &sceneShaderData, sizeof(SceneShaderData));
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
		cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, { descriptorSets[frameIndex]}, {});
		vk::DeviceSize vOffset{ 0 };
		cb.bindVertexBuffers(0, 1, &vBuffer, &vOffset);
		cb.bindIndexBuffer(iBuffer, vOffset, vk::IndexType::eUint32);
		for (size_t i = 0; i < PhysicsWorld::world->bodies.size(); i++) {
			uint32_t objIndex = static_cast<uint32_t>(i);
			cb.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eAll, 0, sizeof(uint32_t), &objIndex);
			cb.drawIndexed(sizeof(indices) / sizeof(indices[0]), 1, 0, 0, 0);
		}
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

				// Shot a high velocity object at camera direction
				if (keyPressed->scancode == sf::Keyboard::Scancode::Space) {
					// Throw a new object in the camera view direction
					const JPH::Vec3 dim{ 0.5, 0.5, 0.5 };
					JPH::BoxShapeSettings body_shape_settings(dim * 0.5);
					body_shape_settings.SetDensity(1000.0);
					body_shape_settings.SetEmbedded();
					JPH::ShapeSettings::ShapeResult body_shape_result = body_shape_settings.Create();
					JPH::ShapeRefC body_shape = body_shape_result.Get();
					JPH::BodyCreationSettings body_settings(body_shape, JPH::RVec3(camera.position.x, camera.position.y, camera.position.z) * -1.0, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, PhysicsWorld::Layers::MOVING);
					auto newBody = PhysicsWorld::world->AddNewObject(body_interface.CreateBody(body_settings), dim, { 1.0, 1.0, 1.0 });
					body_interface.AddBody(newBody->id, JPH::EActivation::Activate);
					body_interface.SetMotionQuality(newBody->id, JPH::EMotionQuality::LinearCast);
					body_interface.AddLinearVelocity(newBody->id, JPH::Vec3(camera.forwardVector.x, camera.forwardVector.y, camera.forwardVector.z) * -150.0f);
				}

				// Drop some random cubes
				if (keyPressed->scancode == sf::Keyboard::Scancode::O) {
					std::default_random_engine rGen((unsigned)time(nullptr));
					std::uniform_real_distribution<float> rDist(-0.4, 0.4);
					std::uniform_real_distribution<float> cDist(0.1, 1.0);
					uint32_t idx = 0;
					for (int32_t x = -4; x < 4; x++, idx++) {
						for (int32_t y = -4; y < 4; y++, idx++) {
							const JPH::Vec3 dim{ 1.0, 1.0, 1.0 };
							JPH::BoxShapeSettings body_shape_settings(dim * 0.5);
							body_shape_settings.SetDensity(250.0);
							body_shape_settings.SetEmbedded();
							JPH::ShapeSettings::ShapeResult body_shape_result = body_shape_settings.Create();
							JPH::ShapeRefC body_shape = body_shape_result.Get();
							JPH::Vec3 pos = { (float)x * 1.5f + rDist(rGen), -15.0f + rDist(rGen), (float)y * 1.5f + rDist(rGen)};
							JPH::BodyCreationSettings body_settings(body_shape, JPH::RVec3(pos), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, PhysicsWorld::Layers::MOVING);
							glm::vec3 color = { cDist(rGen), cDist(rGen), cDist(rGen) };
							auto newBody = PhysicsWorld::world->AddNewObject(body_interface.CreateBody(body_settings), dim, color);
							body_interface.AddBody(newBody->id, JPH::EActivation::Activate);
						}
					}
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
		if (!paused) {
			physicsSystem.Update(dT.asSeconds(), physCollisionSteps, &temp_allocator, &job_system);
		}
		for (size_t i = 0; i < PhysicsWorld::world->bodies.size(); i++) {
			auto& object = PhysicsWorld::world->bodies[i];
			JPH::Mat44 scaleMat = JPH::Mat44::sIdentity().PostScaled(object.scale);
			objectShaderData[i].model = body_interface.GetWorldTransform(object.id) * scaleMat;
			objectShaderData[i].color = glm::vec4(object.color, 1.0f);
		}
		memcpy(objectDataBuffers[frameIndex].allocInfo.pMappedData, objectShaderData.data(), sizeof(ObjectShaderData) * objectShaderData.size());
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
	for (auto& b : uniformBuffers) {
		vmaDestroyBuffer(allocator, b.buffer, b.alloc);
	}
	for (auto& b : objectDataBuffers) {
		vmaDestroyBuffer(allocator, b.buffer, b.alloc);
	}

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