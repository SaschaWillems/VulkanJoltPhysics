# Vulkan and Jolt Physics

## About

Playing around with the [Jolt physics engine](https://github.com/jrouwe/JoltPhysics) and how to integrate it with Vulkan.

## How the dynamic physics objects are rendered

Probably the most interesting part of this sample is how the physics data calculated by Jolt is passed to Vulkan. Esp. as this sample allows for objects to be spawned dynamically.

The solution to this is pretty straight-forward using shader storage buffers with a fixed size and a push constant for indexing.

Physics object data is stored in a vector on the host:

```cpp
struct ObjectShaderData {
	JPH::Mat44 model;
	glm::vec4 color;
};
std::vector<ObjectShaderData> objectShaderData;
```

First we create a shader storage buffer (per max. no of frames-in-flight) that can hold the maximum amount of physics objects:

```cpp
objectDataBuffers.resize(maxFramesInFlight);

VkBufferCreateInfo bufferCI{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizeof(vertices), .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT };
VmaAllocationCreateInfo bufferAllocCI{ .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, .usage = VMA_MEMORY_USAGE_AUTO };
VmaAllocationInfo bufferAllocInfo{};
// Make it so big, that it can potentially hold the max. no. of allowed physics objects
bufferCI.size = sizeof(ObjectShaderData) * physMaxBodies;
bufferCI.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

for (auto& b : objectDataBuffers) {
		chk(vmaCreateBuffer(allocator, &bufferCI, &bufferAllocCI, reinterpret_cast<VkBuffer*>(&b.buffer), &b.alloc, &b.allocInfo));
	}
```

This way we don't need to recreated this buffer, unless we reach said max. no. of physics objects.

After running the physics simulation, relevant values are copied over to the buffer for the actual no. of physics objects:

```cpp
physicsSystem.Update(dT.asSeconds(), physCollisionSteps, &temp_allocator, &job_system);
for (size_t i = 0; i < PhysicsWorld::world->bodies.size(); i++) {
    auto& object = PhysicsWorld::world->bodies[i];
    JPH::Mat44 scaleMat = JPH::Mat44::sIdentity().PostScaled(object.scale);
    objectShaderData[i].model = body_interface.GetWorldTransform(object.id) * scaleMat;
    objectShaderData[i].color = glm::vec4(object.color, 1.0f);
}
memcpy(objectDataBuffers[frameIndex].allocInfo.pMappedData, objectShaderData.data(), sizeof(ObjectShaderData) * objectShaderData.size());
```

We then use a push constant to pass the current object's index to the shader:

```cpp
for (size_t i = 0; i < PhysicsWorld::world->bodies.size(); i++) {
    uint32_t objIndex = static_cast<uint32_t>(i);
    cb.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eAll, 0, sizeof(uint32_t), &objIndex);
    cb.drawIndexed(sizeof(indices) / sizeof(indices[0]), 1, 0, 0, 0);
}
```

With this setup on the host side, we can now simply index into the storage buffer to get the data for each object inside our shader:

```slang
struct ObjectShaderData {
    float4x4 model;
    float4 color;
};
StructuredBuffer<ObjectShaderData> objectData;

...

[shader("vertex")]
// Push constants can be passed as uniform arguments in slang, so objIndex = push constant
VSOutput main(VSInput input, uniform uint objIndex)
{
    float4 pos = mul(objectData[objIndex].model, float4(input.Pos, 1.0));

    VSOutput output;
    output.Color = objectData[objIndex].color.rgb;
    output.Normal = mul((float3x3)objectData[objIndex].model, input.Normal);
    output.Pos = mul(sceneData.projection, mul(sceneData.view, pos));

    ...

	return output;
}

```