#pragma once

#include <shared/user_input.inl>
#include <shared/player.inl>
#include <shared/scene.inl>

#define GPU_INPUT_FLAG_INDEX_PAUSED 0
#define GPU_INPUT_FLAG_INDEX_LIMIT_EDIT_RATE 1

DAXA_DECL_BUFFER_STRUCT(GpuInput, {
    u32vec2 frame_dim;
    f32 time;
    f32 delta_time;
    Settings settings;
    MouseInput mouse;
    KeyboardInput keyboard;
});
DAXA_DECL_BUFFER_STRUCT(GpuGlobals, {
    Player player;
    Scene scene;

    f32vec3 pick_pos;
    f32vec3 pick_nrm;

    f32vec3 edit_origin;
    u32 edit_flags;
});

DAXA_DECL_BUFFER_STRUCT(GpuIndirectDispatch, {
    u32vec3 chunk_edit_dispatch;
    u32vec3 subchunk_x2x4_dispatch;
    u32vec3 subchunk_x8up_dispatch;
});

struct StartupCompPush {
    BufferRef(GpuGlobals) gpu_globals;
};
struct PerframeCompPush {
    BufferRef(GpuGlobals) gpu_globals;
    BufferRef(GpuInput) gpu_input;
    BufferRef(GpuIndirectDispatch) gpu_indirect_dispatch;
};
struct ChunkgenCompPush {
    BufferRef(GpuGlobals) gpu_globals;
    BufferRef(GpuInput) gpu_input;
};
struct ChunkOptCompPush {
    BufferRef(GpuGlobals) gpu_globals;
};
struct ChunkEditCompPush {
    BufferRef(GpuGlobals) gpu_globals;
    BufferRef(GpuInput) gpu_input;
};
struct DrawCompPush {
    BufferRef(GpuGlobals) gpu_globals;
    BufferRef(GpuInput) gpu_input;

    ImageViewId image_id;
};

#define GLOBALS push_constant.gpu_globals
#define SCENE push_constant.gpu_globals.scene
#define VOXEL_WORLD push_constant.gpu_globals.scene.voxel_world
#define VOXEL_CHUNKS push_constant.gpu_globals.scene.voxel_world.voxel_chunks
#define UNIFORMITY_CHUNKS push_constant.gpu_globals.scene.voxel_world.uniformity_chunks
#define INPUT push_constant.gpu_input
#define PLAYER push_constant.gpu_globals.player
#define INDIRECT push_constant.gpu_indirect_dispatch