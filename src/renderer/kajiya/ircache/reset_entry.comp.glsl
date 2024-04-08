#include <renderer/kajiya/ircache.inl>
#include <renderer/kajiya/inc/sh.glsl>
#include "ircache_constants.glsl"

DAXA_DECL_PUSH_CONSTANT(IrcacheResetComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_BufferPtr(uint) ircache_life_buf = push.uses.ircache_life_buf;
daxa_RWBufferPtr(IrcacheMetadata) ircache_meta_buf = push.uses.ircache_meta_buf;
daxa_BufferPtr(vec4) ircache_irradiance_buf = push.uses.ircache_irradiance_buf;
daxa_RWBufferPtr(IrcacheAux) ircache_aux_buf = push.uses.ircache_aux_buf;
daxa_BufferPtr(uint) ircache_entry_indirection_buf = push.uses.ircache_entry_indirection_buf;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint dispatch_idx = gl_GlobalInvocationID.x;
    if (IRCACHE_FREEZE) {
        return;
    }

    const uint total_alloc_count = deref(ircache_meta_buf).tracing_alloc_count;
    if (dispatch_idx >= total_alloc_count) {
        return;
    }

    const uint entry_idx = deref(advance(ircache_entry_indirection_buf, dispatch_idx));

    const bool should_reset = all(equal(vec4(0.0), deref(advance(ircache_irradiance_buf, entry_idx * IRCACHE_IRRADIANCE_STRIDE))));

    if (should_reset) {
        for (uint i = 0; i < IRCACHE_OCTA_DIMS2; ++i) {
            deref(advance(ircache_aux_buf, entry_idx)).reservoirs[i] = uvec2(0);
        }
        for (uint i = 0; i < IRCACHE_OCTA_DIMS2; ++i) {
            deref(advance(ircache_aux_buf, entry_idx)).values[i] = 0.0.xxxx;
        }
        for (uint i = 0; i < IRCACHE_OCTA_DIMS2; ++i) {
            deref(advance(ircache_aux_buf, entry_idx)).vertexes[i] = VertexPacked(vec4(0.0));
        }
    } else {
        for (uint i = 0; i < IRCACHE_OCTA_DIMS2; ++i) {
            Vertex prev_entry = unpack_vertex(deref(advance(ircache_aux_buf, entry_idx)).vertexes[i]);
            prev_entry.position -= vec3(deref(gpu_input).player.player_unit_offset - deref(gpu_input).player.prev_unit_offset);
            deref(advance(ircache_aux_buf, entry_idx)).vertexes[i] = pack_vertex(prev_entry);
        }
    }
}
