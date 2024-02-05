#pragma once

#include <shared/core.inl>
#include <shared/input.inl>
#include <shared/globals.inl>
#include <shared/voxels/voxels.inl>
#include <shared/utils/prefix_scan.inl>

struct Vertex {
    daxa_f32vec3 position;
    daxa_f32vec3 normal;
};
struct VertexPacked {
    daxa_f32vec4 data0;
};
DAXA_DECL_BUFFER_PTR(VertexPacked)
struct IrcacheMetadata {
    // Same as alloc_count, but frozen at the rt dispatch args stage.
    daxa_u32 tracing_alloc_count;
    daxa_u32 _pad0;
    daxa_u32 entry_count;
    daxa_u32 alloc_count;
};
DAXA_DECL_BUFFER_PTR(IrcacheMetadata)
struct IrcacheCell {
    daxa_u32 entry_index;
    daxa_u32 flags;
};
DAXA_DECL_BUFFER_PTR(IrcacheCell)
struct IrcacheBuffers {
    daxa_RWBufferPtr(IrcacheMetadata) ircache_meta_buf;
    daxa_RWBufferPtr(IrcacheCell) ircache_grid_meta_buf;
    daxa_RWBufferPtr(daxa_u32) ircache_entry_cell_buf;
    daxa_RWBufferPtr(VertexPacked) ircache_spatial_buf;
    daxa_RWBufferPtr(daxa_f32vec4) ircache_irradiance_buf;
    daxa_RWBufferPtr(daxa_f32vec4) ircache_aux_buf;
    daxa_RWBufferPtr(daxa_u32) ircache_life_buf;
    daxa_RWBufferPtr(daxa_u32) ircache_pool_buf;
    daxa_RWBufferPtr(daxa_u32) ircache_entry_indirection_buf;
    daxa_RWBufferPtr(VertexPacked) ircache_reposition_proposal_buf;
    daxa_RWBufferPtr(daxa_u32) ircache_reposition_proposal_count_buf;
};
DAXA_DECL_BUFFER_PTR(IrcacheBuffers)

#define IRCACHE_BUFFER_USE_N 12

#define IRCACHE_USE_BUFFERS()                                                                \
    DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(IrcacheBuffers), ircache_buffers) \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_meta_buf)                              \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_grid_meta_buf)                         \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_entry_cell_buf)                        \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_spatial_buf)                           \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_irradiance_buf)                        \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_aux_buf)                               \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_life_buf)                              \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_pool_buf)                              \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_entry_indirection_buf)                 \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_reposition_proposal_buf)               \
    DAXA_TH_BUFFER(COMPUTE_SHADER_READ_WRITE, ircache_reposition_proposal_count_buf)

#define IRCACHE_USE_BUFFERS_PUSH_USES()                                                                                      \
    daxa_BufferPtr(IrcacheBuffers) ircache_buffers = push.uses.ircache_buffers;                                              \
    daxa_RWBufferPtr(IrcacheMetadata) ircache_meta_buf = deref(ircache_buffers).ircache_meta_buf;                            \
    daxa_RWBufferPtr(IrcacheCell) ircache_grid_meta_buf = deref(ircache_buffers).ircache_grid_meta_buf;                      \
    daxa_RWBufferPtr(daxa_u32) ircache_entry_cell_buf = deref(ircache_buffers).ircache_entry_cell_buf;                       \
    daxa_RWBufferPtr(VertexPacked) ircache_spatial_buf = deref(ircache_buffers).ircache_spatial_buf;                         \
    daxa_RWBufferPtr(daxa_f32vec4) ircache_irradiance_buf = deref(ircache_buffers).ircache_irradiance_buf;                   \
    daxa_RWBufferPtr(daxa_f32vec4) ircache_aux_buf = deref(ircache_buffers).ircache_aux_buf;                                 \
    daxa_RWBufferPtr(daxa_u32) ircache_life_buf = deref(ircache_buffers).ircache_life_buf;                                   \
    daxa_RWBufferPtr(daxa_u32) ircache_pool_buf = deref(ircache_buffers).ircache_pool_buf;                                   \
    daxa_RWBufferPtr(daxa_u32) ircache_entry_indirection_buf = deref(ircache_buffers).ircache_entry_indirection_buf;         \
    daxa_RWBufferPtr(VertexPacked) ircache_reposition_proposal_buf = deref(ircache_buffers).ircache_reposition_proposal_buf; \
    daxa_RWBufferPtr(daxa_u32) ircache_reposition_proposal_count_buf = deref(ircache_buffers).ircache_reposition_proposal_count_buf;

#define IRCACHE_BUFFER_USES_ASSIGN(TaskHeadName, ircache)                                                                         \
    daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_buffers, ircache.ircache_buffers}},                                     \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_meta_buf, ircache.ircache_meta_buf}},                               \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_grid_meta_buf, ircache.ircache_grid_meta_buf}},                     \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_entry_cell_buf, ircache.ircache_entry_cell_buf}},                   \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_spatial_buf, ircache.ircache_spatial_buf}},                         \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_irradiance_buf, ircache.ircache_irradiance_buf}},                   \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_aux_buf, ircache.ircache_aux_buf}},                                 \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_life_buf, ircache.ircache_life_buf}},                               \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_pool_buf, ircache.ircache_pool_buf}},                               \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_entry_indirection_buf, ircache.ircache_entry_indirection_buf}},     \
        daxa::TaskViewVariant{std::pair{TaskHeadName::ircache_reposition_proposal_buf, ircache.ircache_reposition_proposal_buf}}, \
        daxa::TaskViewVariant {                                                                                                   \
        std::pair { TaskHeadName::ircache_reposition_proposal_count_buf, ircache.ircache_reposition_proposal_count_buf }          \
    }

DAXA_DECL_TASK_HEAD_BEGIN(ClearIrcachePoolCompute, 2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_life_buf)
DAXA_DECL_TASK_HEAD_END
struct ClearIrcachePoolComputePush {
    DAXA_TH_BLOB(ClearIrcachePoolCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(IrcacheScrollCascadesCompute, 8)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(IrcacheCell), ircache_grid_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheCell), ircache_grid_meta_buf2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_entry_cell_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_f32vec4), ircache_irradiance_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_DECL_TASK_HEAD_END
struct IrcacheScrollCascadesComputePush {
    DAXA_TH_BLOB(IrcacheScrollCascadesCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(IrcachePrepareAgeDispatchCompute, 2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32vec4), dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcachePrepareAgeDispatchComputePush {
    DAXA_TH_BLOB(IrcachePrepareAgeDispatchCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(AgeIrcacheEntriesCompute, 11)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheCell), ircache_grid_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_entry_cell_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VertexPacked), ircache_spatial_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VertexPacked), ircache_reposition_proposal_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_reposition_proposal_count_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_f32vec4), ircache_irradiance_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), entry_occupancy_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct AgeIrcacheEntriesComputePush {
    DAXA_TH_BLOB(AgeIrcacheEntriesCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(IrcacheCompactEntriesCompute, 5)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), entry_occupancy_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcacheCompactEntriesComputePush {
    DAXA_TH_BLOB(IrcacheCompactEntriesCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(IrcachePrepareTraceDispatchCompute, 2)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_RWBufferPtr(daxa_u32vec4), dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcachePrepareTraceDispatchComputePush {
    DAXA_TH_BLOB(IrcachePrepareTraceDispatchCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(IrcacheResetCompute, 6)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32vec4), ircache_irradiance_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_f32vec4), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcacheResetComputePush {
    DAXA_TH_BLOB(IrcacheResetCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(IrcacheTraceAccessCompute, 8 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VertexPacked), ircache_spatial_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VertexPacked), ircache_reposition_proposal_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_f32vec4), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcacheTraceAccessComputePush {
    DAXA_TH_BLOB(IrcacheTraceAccessCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(IrcacheValidateCompute, 15 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VertexPacked), ircache_spatial_buf)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, CUBE, sky_cube_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheCell), ircache_grid_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VertexPacked), ircache_reposition_proposal_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_reposition_proposal_count_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_f32vec4), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_entry_cell_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct IrcacheValidateComputePush {
    DAXA_TH_BLOB(IrcacheValidateCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(TraceIrradianceCompute, 15 + VOXEL_BUFFER_USE_N)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_RWBufferPtr(GpuGlobals), globals)
VOXELS_USE_BUFFERS(daxa_BufferPtr, COMPUTE_SHADER_READ)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VertexPacked), ircache_spatial_buf)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, CUBE, sky_cube_tex)
DAXA_TH_IMAGE_INDEX(COMPUTE_SHADER_SAMPLED, REGULAR_2D, transmittance_lut)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheCell), ircache_grid_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(VertexPacked), ircache_reposition_proposal_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_reposition_proposal_count_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_f32vec4), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_pool_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_u32), ircache_entry_cell_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct TraceIrradianceComputePush {
    DAXA_TH_BLOB(TraceIrradianceCompute, uses)
};
DAXA_DECL_TASK_HEAD_BEGIN(SumUpIrradianceCompute, 7)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GpuInput), gpu_input)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_life_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(IrcacheMetadata), ircache_meta_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_f32vec4), ircache_irradiance_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_RWBufferPtr(daxa_f32vec4), ircache_aux_buf)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_u32), ircache_entry_indirection_buf)
DAXA_TH_BUFFER(COMPUTE_SHADER_READ, dispatch_args)
DAXA_DECL_TASK_HEAD_END
struct SumUpIrradianceComputePush {
    DAXA_TH_BLOB(SumUpIrradianceCompute, uses)
};

#if defined(__cplusplus)

#include <array>
#include <glm/glm.hpp>

struct IrcacheIrradiancePendingSummation {
    daxa::TaskBufferView indirect_args_buf;
};

struct IrcacheRenderState {
    daxa::TaskBuffer ircache_meta_buf;

    daxa::TaskBuffer ircache_grid_meta_buf;
    daxa::TaskBuffer ircache_grid_meta_buf2;

    daxa::TaskBuffer ircache_entry_cell_buf;
    daxa::TaskBuffer ircache_spatial_buf;
    daxa::TaskBuffer ircache_irradiance_buf;
    daxa::TaskBuffer ircache_aux_buf;

    daxa::TaskBuffer ircache_life_buf;
    daxa::TaskBuffer ircache_pool_buf;
    daxa::TaskBuffer ircache_entry_indirection_buf;

    daxa::TaskBuffer ircache_reposition_proposal_buf;
    daxa::TaskBuffer ircache_reposition_proposal_count_buf;

    daxa::TaskBufferView ircache_buffers;

    bool pending_irradiance_sum;

    auto trace_irradiance(RecordContext &record_ctx, VoxelWorld::Buffers &voxel_buffers, daxa::TaskImageView sky_cube, daxa::TaskImageView transmittance_lut) -> IrcacheIrradiancePendingSummation {
        auto indirect_args_buf = record_ctx.task_graph.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(uint32_t) * 4) * 4,
            .name = "ircache.trace_indirect_args_buf",
        });

        record_ctx.add(ComputeTask<IrcachePrepareTraceDispatchCompute, IrcachePrepareTraceDispatchComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"ircache/prepare_trace_dispatch_args.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{IrcachePrepareTraceDispatchCompute::ircache_meta_buf, this->ircache_meta_buf}},
                daxa::TaskViewVariant{std::pair{IrcachePrepareTraceDispatchCompute::dispatch_args, indirect_args_buf}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, IrcachePrepareTraceDispatchComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({1, 1, 1});
            },
        });

        record_ctx.add(ComputeTask<IrcacheResetCompute, IrcacheResetComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"ircache/reset_entry.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{IrcacheResetCompute::ircache_life_buf, this->ircache_life_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheResetCompute::ircache_meta_buf, this->ircache_meta_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheResetCompute::ircache_irradiance_buf, this->ircache_irradiance_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheResetCompute::ircache_aux_buf, this->ircache_aux_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheResetCompute::ircache_entry_indirection_buf, this->ircache_entry_indirection_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheResetCompute::dispatch_args, indirect_args_buf}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, IrcacheResetComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(IrcacheResetCompute::dispatch_args).ids[0],
                    .offset = sizeof(daxa_u32vec4) * 2,
                });
            },
        });

        record_ctx.add(ComputeTask<IrcacheTraceAccessCompute, IrcacheTraceAccessComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"ircache/trace_accessibility.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{IrcacheTraceAccessCompute::globals, record_ctx.task_globals_buffer}},
                VOXELS_BUFFER_USES_ASSIGN(IrcacheTraceAccessCompute, voxel_buffers),
                daxa::TaskViewVariant{std::pair{IrcacheTraceAccessCompute::ircache_spatial_buf, this->ircache_spatial_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheTraceAccessCompute::ircache_life_buf, this->ircache_life_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheTraceAccessCompute::ircache_reposition_proposal_buf, this->ircache_reposition_proposal_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheTraceAccessCompute::ircache_meta_buf, this->ircache_meta_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheTraceAccessCompute::ircache_aux_buf, this->ircache_aux_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheTraceAccessCompute::ircache_entry_indirection_buf, this->ircache_entry_indirection_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheTraceAccessCompute::dispatch_args, indirect_args_buf}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, IrcacheTraceAccessComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(IrcacheTraceAccessCompute::dispatch_args).ids[0],
                    .offset = sizeof(daxa_u32vec4) * 1,
                });
            },
        });

        record_ctx.add(ComputeTask<IrcacheValidateCompute, IrcacheValidateComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"ircache/ircache_validate.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::globals, record_ctx.task_globals_buffer}},
                VOXELS_BUFFER_USES_ASSIGN(IrcacheValidateCompute, voxel_buffers),
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::ircache_spatial_buf, this->ircache_spatial_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::sky_cube_tex, sky_cube}},
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::transmittance_lut, transmittance_lut}},
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::ircache_grid_meta_buf, this->ircache_grid_meta_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::ircache_life_buf, this->ircache_life_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::ircache_reposition_proposal_buf, this->ircache_reposition_proposal_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::ircache_reposition_proposal_count_buf, this->ircache_reposition_proposal_count_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::ircache_meta_buf, this->ircache_meta_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::ircache_aux_buf, this->ircache_aux_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::ircache_pool_buf, this->ircache_pool_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::ircache_entry_indirection_buf, this->ircache_entry_indirection_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::ircache_entry_cell_buf, this->ircache_entry_cell_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheValidateCompute::dispatch_args, indirect_args_buf}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, IrcacheValidateComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(IrcacheValidateCompute::dispatch_args).ids[0],
                    .offset = sizeof(daxa_u32vec4) * 3,
                });
            },
        });

        record_ctx.add(ComputeTask<TraceIrradianceCompute, TraceIrradianceComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"ircache/trace_irradiance.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::globals, record_ctx.task_globals_buffer}},
                VOXELS_BUFFER_USES_ASSIGN(TraceIrradianceCompute, voxel_buffers),
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::ircache_spatial_buf, this->ircache_spatial_buf}},
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::sky_cube_tex, sky_cube}},
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::transmittance_lut, transmittance_lut}},
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::ircache_grid_meta_buf, this->ircache_grid_meta_buf}},
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::ircache_life_buf, this->ircache_life_buf}},
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::ircache_reposition_proposal_buf, this->ircache_reposition_proposal_buf}},
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::ircache_reposition_proposal_count_buf, this->ircache_reposition_proposal_count_buf}},
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::ircache_meta_buf, this->ircache_meta_buf}},
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::ircache_aux_buf, this->ircache_aux_buf}},
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::ircache_pool_buf, this->ircache_pool_buf}},
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::ircache_entry_indirection_buf, this->ircache_entry_indirection_buf}},
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::ircache_entry_cell_buf, this->ircache_entry_cell_buf}},
                daxa::TaskViewVariant{std::pair{TraceIrradianceCompute::dispatch_args, indirect_args_buf}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, TraceIrradianceComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                // TODO, check if this is broken like Tom says it is.
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(TraceIrradianceCompute::dispatch_args).ids[0],
                    .offset = sizeof(daxa_u32vec4) * 0,
                });
            },
        });

        return {indirect_args_buf};
    }

    auto sum_up_irradiance_for_sampling(RecordContext &record_ctx, IrcacheIrradiancePendingSummation pending) {
        record_ctx.add(ComputeTask<SumUpIrradianceCompute, SumUpIrradianceComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"ircache/sum_up_irradiance.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{SumUpIrradianceCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{SumUpIrradianceCompute::ircache_life_buf, this->ircache_life_buf}},
                daxa::TaskViewVariant{std::pair{SumUpIrradianceCompute::ircache_meta_buf, this->ircache_meta_buf}},
                daxa::TaskViewVariant{std::pair{SumUpIrradianceCompute::ircache_irradiance_buf, this->ircache_irradiance_buf}},
                daxa::TaskViewVariant{std::pair{SumUpIrradianceCompute::ircache_aux_buf, this->ircache_aux_buf}},
                daxa::TaskViewVariant{std::pair{SumUpIrradianceCompute::ircache_entry_indirection_buf, this->ircache_entry_indirection_buf}},
                daxa::TaskViewVariant{std::pair{SumUpIrradianceCompute::dispatch_args, pending.indirect_args_buf}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, SumUpIrradianceComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(SumUpIrradianceCompute::dispatch_args).ids[0],
                    .offset = sizeof(daxa_u32vec4) * 2,
                });
            },
        });

        this->pending_irradiance_sum = false;
    }
};

inline auto temporal_storage_buffer(RecordContext &record_ctx, std::string_view name, size_t size) -> daxa::TaskBuffer {
    auto result = record_ctx.find_or_add_temporal_buffer({
        .size = static_cast<uint32_t>(size),
        .name = name,
    });
    return result.task_buffer;
}

struct IrcacheRenderer {
    bool initialized = false;
    glm::vec3 grid_center{};
    std::array<glm::ivec3, IRCACHE_CASCADE_COUNT> cur_scroll{};
    std::array<glm::ivec3, IRCACHE_CASCADE_COUNT> prev_scroll{};
    size_t parity = 0;
    bool enable_scroll = true;

    PingPongBuffer ping_pong_ircache_grid_meta_buf;

    void update_eye_position(GpuInput &gpu_input, GpuOutput const &gpu_output) {
        if (!this->enable_scroll) {
            return;
        }

        gpu_input.ircache_grid_center = daxa_f32vec3{gpu_output.player_pos.x, gpu_output.player_pos.y, gpu_output.player_pos.z};

        this->grid_center = glm::vec3(gpu_input.ircache_grid_center.x, gpu_input.ircache_grid_center.y, gpu_input.ircache_grid_center.z);

        for (size_t cascade = 0; cascade < IRCACHE_CASCADE_COUNT; ++cascade) {
            auto cell_diameter = IRCACHE_GRID_CELL_DIAMETER * static_cast<float>(1 << cascade);
            auto cascade_center = glm::ivec3(glm::floor(this->grid_center / cell_diameter));
            auto cascade_origin = cascade_center - glm::ivec3(IRCACHE_CASCADE_SIZE / 2);

            this->prev_scroll[cascade] = this->cur_scroll[cascade];
            this->cur_scroll[cascade] = cascade_origin;

            gpu_input.ircache_cascades[cascade].origin = {
                this->cur_scroll[cascade].x,
                this->cur_scroll[cascade].y,
                this->cur_scroll[cascade].z,
                0,
            };
            gpu_input.ircache_cascades[cascade].voxels_scrolled_this_frame = {
                this->cur_scroll[cascade].x - this->prev_scroll[cascade].x,
                this->cur_scroll[cascade].y - this->prev_scroll[cascade].y,
                this->cur_scroll[cascade].z - this->prev_scroll[cascade].z,
                0,
            };
        }
    }

    void next_frame() {
        ping_pong_ircache_grid_meta_buf.swap();
        this->parity = (this->parity + 1) % 2;
    }

    auto prepare(RecordContext &record_ctx) -> IrcacheRenderState {
        constexpr auto INDIRECTION_BUF_ELEM_COUNT = size_t{1024 * 1024};

        auto [ircache_grid_meta_buf_, ircache_grid_meta_buf2_] = ping_pong_ircache_grid_meta_buf.get(
            record_ctx.device,
            daxa::BufferInfo{
                .size = static_cast<uint32_t>(sizeof(IrcacheCell) * MAX_GRID_CELLS),
                .name = "ircache.grid_meta_buf",
            });
        record_ctx.task_graph.use_persistent_buffer(ircache_grid_meta_buf_);
        record_ctx.task_graph.use_persistent_buffer(ircache_grid_meta_buf2_);

        auto state = IrcacheRenderState{
            // 0: hash grid cell count
            // 1: entry count
            .ircache_meta_buf = temporal_storage_buffer(record_ctx, "ircache.meta_buf", sizeof(IrcacheMetadata)),
            .ircache_grid_meta_buf = ircache_grid_meta_buf_,
            .ircache_grid_meta_buf2 = ircache_grid_meta_buf2_,
            .ircache_entry_cell_buf = temporal_storage_buffer(
                record_ctx,
                "ircache.entry_cell_buf",
                sizeof(daxa_u32) * MAX_ENTRIES),
            .ircache_spatial_buf = temporal_storage_buffer(
                record_ctx,
                "ircache.spatial_buf",
                sizeof(daxa_f32vec4) * MAX_ENTRIES),
            .ircache_irradiance_buf = temporal_storage_buffer(
                record_ctx,
                "ircache.irradiance_buf",
                3 * sizeof(daxa_f32vec4) * MAX_ENTRIES),
            .ircache_aux_buf = temporal_storage_buffer(
                record_ctx,
                "ircache.aux_buf",
                4 * 16 * sizeof(daxa_f32vec4) * MAX_ENTRIES),
            .ircache_life_buf = temporal_storage_buffer(
                record_ctx,
                "ircache.life_buf",
                sizeof(daxa_u32) * MAX_ENTRIES),
            .ircache_pool_buf = temporal_storage_buffer(
                record_ctx,
                "ircache.pool_buf",
                sizeof(daxa_u32) * MAX_ENTRIES),
            .ircache_entry_indirection_buf = temporal_storage_buffer(
                record_ctx,
                "ircache.entry_indirection_buf",
                sizeof(daxa_u32) * INDIRECTION_BUF_ELEM_COUNT),
            .ircache_reposition_proposal_buf = temporal_storage_buffer(
                record_ctx,
                "ircache.reposition_proposal_buf",
                sizeof(daxa_f32vec4) * MAX_ENTRIES),
            .ircache_reposition_proposal_count_buf = temporal_storage_buffer(
                record_ctx,
                "ircache.reposition_proposal_count_buf",
                sizeof(daxa_u32) * MAX_ENTRIES),
            .pending_irradiance_sum = false,
        };

        if (!this->initialized) {
            auto temp_record_ctx = RecordContext{
                .device = record_ctx.device,
                .task_graph = daxa::TaskGraph({
                    .device = record_ctx.device,
                    .name = "temp_task_graph",
                }),
                .pipeline_manager = record_ctx.pipeline_manager,
                .compute_pipelines = record_ctx.compute_pipelines,
                .raster_pipelines = record_ctx.raster_pipelines,
            };

            temp_record_ctx.task_graph.use_persistent_buffer(state.ircache_pool_buf);
            temp_record_ctx.task_graph.use_persistent_buffer(state.ircache_life_buf);

            temp_record_ctx.add(ComputeTask<ClearIrcachePoolCompute, ClearIrcachePoolComputePush, NoTaskInfo>{
                .source = daxa::ShaderFile{"ircache/clear_ircache_pool.comp.glsl"},
                .views = std::array{
                    daxa::TaskViewVariant{std::pair{ClearIrcachePoolCompute::ircache_pool_buf, state.ircache_pool_buf}},
                    daxa::TaskViewVariant{std::pair{ClearIrcachePoolCompute::ircache_life_buf, state.ircache_life_buf}},
                },
                .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, ClearIrcachePoolComputePush &push, NoTaskInfo const &) {
                    ti.recorder.set_pipeline(pipeline);
                    set_push_constant(ti, push);
                    ti.recorder.dispatch({(MAX_ENTRIES + 63) / 64});
                },
            });

            temp_record_ctx.task_graph.submit({});
            temp_record_ctx.task_graph.complete({});
            temp_record_ctx.task_graph.execute({});

            this->initialized = true;
        }

        record_ctx.add(ComputeTask<IrcacheScrollCascadesCompute, IrcacheScrollCascadesComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"ircache/scroll_cascades.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::gpu_input, record_ctx.task_input_buffer}},
                daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::ircache_grid_meta_buf, state.ircache_grid_meta_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::ircache_grid_meta_buf2, state.ircache_grid_meta_buf2}},
                daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::ircache_entry_cell_buf, state.ircache_entry_cell_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::ircache_irradiance_buf, state.ircache_irradiance_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::ircache_life_buf, state.ircache_life_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::ircache_pool_buf, state.ircache_pool_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheScrollCascadesCompute::ircache_meta_buf, state.ircache_meta_buf}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, IrcacheScrollCascadesComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({(IRCACHE_CASCADE_SIZE + 31) / 32, IRCACHE_CASCADE_SIZE, IRCACHE_CASCADE_SIZE * IRCACHE_CASCADE_COUNT});
            },
        });

        std::swap(state.ircache_grid_meta_buf, state.ircache_grid_meta_buf2);

        auto indirect_args_buf = record_ctx.task_graph.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(uint32_t) * 4) * 2,
            .name = "ircache.age_indirect_args_buf",
        });

        record_ctx.add(ComputeTask<IrcachePrepareAgeDispatchCompute, IrcachePrepareAgeDispatchComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"ircache/prepare_age_dispatch_args.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{IrcachePrepareAgeDispatchCompute::ircache_meta_buf, state.ircache_meta_buf}},
                daxa::TaskViewVariant{std::pair{IrcachePrepareAgeDispatchCompute::dispatch_args, indirect_args_buf}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, IrcachePrepareAgeDispatchComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch({1, 1, 1});
            },
        });

        auto entry_occupancy_buf = record_ctx.task_graph.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(uint32_t) * MAX_ENTRIES),
            .name = "ircache.entry_occupancy_buf",
        });
        record_ctx.add(ComputeTask<AgeIrcacheEntriesCompute, AgeIrcacheEntriesComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"ircache/age_ircache_entries.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::ircache_meta_buf, state.ircache_meta_buf}},
                daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::ircache_grid_meta_buf, state.ircache_grid_meta_buf}},
                daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::ircache_entry_cell_buf, state.ircache_entry_cell_buf}},
                daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::ircache_life_buf, state.ircache_life_buf}},
                daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::ircache_pool_buf, state.ircache_pool_buf}},
                daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::ircache_spatial_buf, state.ircache_spatial_buf}},
                daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::ircache_reposition_proposal_buf, state.ircache_reposition_proposal_buf}},
                daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::ircache_reposition_proposal_count_buf, state.ircache_reposition_proposal_count_buf}},
                daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::ircache_irradiance_buf, state.ircache_irradiance_buf}},
                daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::entry_occupancy_buf, entry_occupancy_buf}},
                daxa::TaskViewVariant{std::pair{AgeIrcacheEntriesCompute::dispatch_args, indirect_args_buf}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, AgeIrcacheEntriesComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(AgeIrcacheEntriesCompute::dispatch_args).ids[0],
                    .offset = 0,
                });
            },
        });

        inclusive_prefix_scan_u32_1m(record_ctx, entry_occupancy_buf);

        record_ctx.add(ComputeTask<IrcacheCompactEntriesCompute, IrcacheCompactEntriesComputePush, NoTaskInfo>{
            .source = daxa::ShaderFile{"ircache/ircache_compact_entries.comp.glsl"},
            .views = std::array{
                daxa::TaskViewVariant{std::pair{IrcacheCompactEntriesCompute::ircache_meta_buf, state.ircache_meta_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheCompactEntriesCompute::ircache_life_buf, state.ircache_life_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheCompactEntriesCompute::entry_occupancy_buf, entry_occupancy_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheCompactEntriesCompute::ircache_entry_indirection_buf, state.ircache_entry_indirection_buf}},
                daxa::TaskViewVariant{std::pair{IrcacheCompactEntriesCompute::dispatch_args, indirect_args_buf}},
            },
            .callback_ = [](daxa::TaskInterface const &ti, daxa::ComputePipeline &pipeline, IrcacheCompactEntriesComputePush &push, NoTaskInfo const &) {
                ti.recorder.set_pipeline(pipeline);
                set_push_constant(ti, push);
                ti.recorder.dispatch_indirect({
                    .indirect_buffer = ti.get(IrcacheCompactEntriesCompute::dispatch_args).ids[0],
                    .offset = 0,
                });
            },
        });

        state.ircache_buffers = record_ctx.task_graph.create_transient_buffer({
            .size = static_cast<daxa_u32>(sizeof(IrcacheBuffers)),
            .name = "ircache.buffers",
        });
        record_ctx.task_graph.add_task({
            .attachments = {
                daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, state.ircache_buffers),
                daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_meta_buf),
                daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_grid_meta_buf),
                daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_entry_cell_buf),
                daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_spatial_buf),
                daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_irradiance_buf),
                daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_aux_buf),
                daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_life_buf),
                daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_pool_buf),
                daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_entry_indirection_buf),
                daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_reposition_proposal_buf),
                daxa::inl_attachment(daxa::TaskBufferAccess::NONE, state.ircache_reposition_proposal_count_buf),
            },
            .task = [this](daxa::TaskInterface const &ti) {
                auto staging_buffer = ti.device.create_buffer({
                    .size = sizeof(IrcacheBuffers),
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = "staging_buffer",
                });
                ti.recorder.destroy_buffer_deferred(staging_buffer);
                auto *buffer_ptr = ti.device.get_host_address_as<IrcacheBuffers>(staging_buffer).value();
                *buffer_ptr = {
                    .ircache_meta_buf = ti.device.get_device_address(ti.get(daxa::TaskBufferAttachmentIndex{1}).ids[0]).value(),
                    .ircache_grid_meta_buf = ti.device.get_device_address(ti.get(daxa::TaskBufferAttachmentIndex{2}).ids[0]).value(),
                    .ircache_entry_cell_buf = ti.device.get_device_address(ti.get(daxa::TaskBufferAttachmentIndex{3}).ids[0]).value(),
                    .ircache_spatial_buf = ti.device.get_device_address(ti.get(daxa::TaskBufferAttachmentIndex{4}).ids[0]).value(),
                    .ircache_irradiance_buf = ti.device.get_device_address(ti.get(daxa::TaskBufferAttachmentIndex{5}).ids[0]).value(),
                    .ircache_aux_buf = ti.device.get_device_address(ti.get(daxa::TaskBufferAttachmentIndex{6}).ids[0]).value(),
                    .ircache_life_buf = ti.device.get_device_address(ti.get(daxa::TaskBufferAttachmentIndex{7}).ids[0]).value(),
                    .ircache_pool_buf = ti.device.get_device_address(ti.get(daxa::TaskBufferAttachmentIndex{8}).ids[0]).value(),
                    .ircache_entry_indirection_buf = ti.device.get_device_address(ti.get(daxa::TaskBufferAttachmentIndex{9}).ids[0]).value(),
                    .ircache_reposition_proposal_buf = ti.device.get_device_address(ti.get(daxa::TaskBufferAttachmentIndex{10}).ids[0]).value(),
                    .ircache_reposition_proposal_count_buf = ti.device.get_device_address(ti.get(daxa::TaskBufferAttachmentIndex{11}).ids[0]).value(),
                };
                ti.recorder.copy_buffer_to_buffer({
                    .src_buffer = staging_buffer,
                    .dst_buffer = ti.get(daxa::TaskBufferAttachmentIndex{0}).ids[0],
                    .size = sizeof(IrcacheBuffers),
                });
            },
            .name = "UploadIrcacheBuffers",
        });

        return state;
    }
};

#endif
