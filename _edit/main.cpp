// Copyright 2020-2021 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vulkan/vulkan.hpp>
// Above must come first!
#include <nvvk/context_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>  // For NVVK memory allocators
#include <nvvk/structs_vk.hpp>            // For nvvk::make

static const uint64_t render_width = 800;
static const uint64_t render_height = 600;

int main(int argc, const char** argv) {
  // Create the Vulkan context, consisting of an instance, device, physical
  // device, and queues.
  nvvk::ContextCreateInfo deviceInfo;
  // Required by KHR_acceleration_structure; allows work to be offloaded onto
  // background threads and parallelized
  deviceInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  vk::PhysicalDeviceAccelerationStructureFeaturesKHR asFeatures;
  deviceInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                false, &asFeatures);
  vk::PhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures;
  deviceInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false,
                                &rayQueryFeatures);

  nvvk::Context context;     // Encapsulates device state in a single object
  context.init(deviceInfo);  // Initialize the context
  // Device must support acceleration structures and ray queries:
  assert(asFeatures.accelerationStructure == VK_TRUE &&
         rayQueryFeatures.rayQuery == VK_TRUE);

  // Create the allocator
  nvvk::ResourceAllocatorDedicated allocator;
  allocator.init(context, context.m_physicalDevice);

  vk::DeviceSize bufferSizeBytes =
      render_width * render_height * 3 * sizeof(float);

  // Create a buffer
  vk::BufferCreateInfo bufferCreateInfo(
      {}, bufferSizeBytes,
      vk::BufferUsageFlagBits::eStorageBuffer |
          vk::BufferUsageFlagBits::eTransferDst);

  // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT means that the CPU can read this
  // buffer's memory. VK_MEMORY_PROPERTY_HOST_CACHED_BIT means that the CPU
  // caches this memory. VK_MEMORY_PROPERTY_HOST_COHERENT_BIT means that the CPU
  // side of cache management is handled automatically, with potentially slower
  // reads/writes.
  nvvk::Buffer buffer = allocator.createBuffer(
      bufferCreateInfo, vk::MemoryPropertyFlagBits::eHostVisible |
                            vk::MemoryPropertyFlagBits::eHostCached |
                            vk::MemoryPropertyFlagBits::eHostCoherent);

  // Create the command pool
  VkCommandPoolCreateInfo cmdPoolInfo = nvvk::make<VkCommandPoolCreateInfo>();
  cmdPoolInfo.queueFamilyIndex = context.m_queueGCT;
  VkCommandPool cmdPool;
  NVVK_CHECK(vkCreateCommandPool(context, &cmdPoolInfo, nullptr, &cmdPool));

  // Allocate a command buffer
  VkCommandBufferAllocateInfo cmdAllocInfo =
      nvvk::make<VkCommandBufferAllocateInfo>();
  cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdAllocInfo.commandPool = cmdPool;
  cmdAllocInfo.commandBufferCount = 1;
  VkCommandBuffer cmdBuffer;
  NVVK_CHECK(vkAllocateCommandBuffers(context, &cmdAllocInfo, &cmdBuffer));

  // Begin recording
  VkCommandBufferBeginInfo beginInfo = nvvk::make<VkCommandBufferBeginInfo>();
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  NVVK_CHECK(vkBeginCommandBuffer(cmdBuffer, &beginInfo));

  // Fill the buffer
  const float fillValue = 0.5f;
  const uint32_t& fillValueU32 = reinterpret_cast<const uint32_t&>(fillValue);
  vkCmdFillBuffer(cmdBuffer, buffer.buffer, 0, bufferSizeBytes, fillValueU32);

  allocator.destroy(buffer);
  allocator.deinit();
  context.deinit();  // Don't forget to clean up at the end of the program!
}