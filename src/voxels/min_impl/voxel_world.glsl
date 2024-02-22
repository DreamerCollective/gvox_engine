#pragma once

#include <application/input.inl>
#include <application/globals.inl>

void voxel_world_startup(daxa_RWBufferPtr(GpuGlobals) globals_ptr, VoxelRWBufferPtrs ptrs) {
}

void voxel_world_perframe(daxa_BufferPtr(GpuInput) gpu_input, daxa_RWBufferPtr(GpuOutput) gpu_output, daxa_RWBufferPtr(GpuGlobals) globals_ptr, VoxelRWBufferPtrs ptrs) {
    deref(ptrs.globals).prev_offset = deref(ptrs.globals).offset;
    deref(ptrs.globals).offset = deref(gpu_input).player.player_unit_offset;
}