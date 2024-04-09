#define DAXA_RAY_TRACING 1
#extension GL_EXT_ray_tracing : enable

#include "trace_primary.inl"
DAXA_DECL_PUSH_CONSTANT(TracePrimaryRtPush, push)
#include <renderer/rt.glsl>

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_RAYGEN

layout(location = PAYLOAD_LOC) rayPayloadEXT RayPayload prd;

#include <renderer/kajiya/inc/camera.glsl>

void main() {
    const ivec2 index = ivec2(gl_LaunchIDEXT.xy);

    vec4 output_tex_size = vec4(deref(push.uses.gpu_input).frame_dim, 0, 0);
    output_tex_size.zw = vec2(1.0, 1.0) / output_tex_size.xy;
    vec2 uv = get_uv(gl_LaunchIDEXT.xy, output_tex_size);

    ViewRayContext vrc = vrc_from_uv(push.uses.gpu_input, uv);
    vec3 ray_d = ray_dir_ws(vrc);
    vec3 ray_o = ray_origin_ws(vrc);

    uint rayFlags = gl_RayFlagsNoneEXT;
    float tMin = 0.0001;
    float tMax = 10000.0;
    uint cull_mask = 0xFF;
    uint sbtRecordOffset = 0;
    uint sbtRecordStride = 0;
    uint missIndex = 0;

    traceRayEXT(
        accelerationStructureEXT(push.uses.tlas),
        rayFlags, cull_mask, sbtRecordOffset, sbtRecordStride, missIndex,
        ray_o, tMin, ray_d, tMax, PAYLOAD_LOC);

    if (prd.data1 == miss_ray_payload().data1) {
        imageStore(daxa_image2D(push.uses.depth_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(0));
        imageStore(daxa_uimage2D(push.uses.g_buffer_image_id), ivec2(gl_LaunchIDEXT.xy), uvec4(0));
        return;
    }

    vec3 world_pos = vec3(0);

    uvec3 chunk_n = uvec3(CHUNKS_PER_AXIS);
    PackedVoxel voxel_data = unpack_ray_payload(push.uses.geometry_pointers, push.uses.attribute_pointers, push.uses.blas_transforms, prd, Ray(ray_o, ray_d), world_pos);
    Voxel voxel = unpack_voxel(voxel_data);

    vec3 ws_nrm = voxel.normal;
    vec3 vs_nrm = (deref(push.uses.gpu_input).player.cam.world_to_view * vec4(ws_nrm, 0)).xyz;
    vec3 vs_velocity = vec3(0, 0, 0);

    vec3 vel_ws = vec3(deref(push.uses.gpu_input).player.player_unit_offset - deref(push.uses.gpu_input).player.prev_unit_offset);

    vec4 vs_pos = (deref(push.uses.gpu_input).player.cam.world_to_view * vec4(world_pos, 1));
    vec4 prev_vs_pos = (deref(push.uses.gpu_input).player.cam.world_to_view * vec4(world_pos + vel_ws, 1));
    vec4 ss_pos = (deref(push.uses.gpu_input).player.cam.view_to_sample * vs_pos);
    float depth = ss_pos.z / ss_pos.w;

    vs_velocity = (prev_vs_pos.xyz / prev_vs_pos.w) - (vs_pos.xyz / vs_pos.w);

    uvec4 output_value = uvec4(0);
    output_value.x = pack_voxel(voxel).data;
    output_value.y = nrm_to_u16(ws_nrm);
    output_value.z = floatBitsToUint(depth);

    vs_nrm *= -sign(dot(ray_dir_vs(vrc), vs_nrm));

    imageStore(daxa_uimage2D(push.uses.g_buffer_image_id), ivec2(gl_LaunchIDEXT.xy), output_value);
    imageStore(daxa_image2D(push.uses.velocity_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(vs_velocity, 0));
    imageStore(daxa_image2D(push.uses.vs_normal_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(vs_nrm * 0.5 + 0.5, 0));
    imageStore(daxa_image2D(push.uses.depth_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(depth, 0, 0, 0));
}
#endif
