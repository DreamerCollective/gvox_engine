#include <shared/app.inl>

#include <utils/math.glsl>
#include <voxels/core.glsl>

#if TRACE_DEPTH_PREPASS_COMPUTE

#define INPUT deref(gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    daxa_f32vec4 output_tex_size;
    output_tex_size.xy = deref(gpu_input).frame_dim;
    output_tex_size.zw = daxa_f32vec2(1.0, 1.0) / output_tex_size.xy;
    daxa_f32vec2 uv = get_uv(gl_GlobalInvocationID.xy * SHADING_SCL, output_tex_size);

    ViewRayContext vrc = vrc_from_uv(globals, uv_to_ss(gpu_input, uv, output_tex_size));
    daxa_f32vec3 ray_dir = ray_dir_ws(vrc);
    daxa_f32vec3 cam_pos = ray_origin_ws(vrc);
    daxa_f32vec3 ray_pos = cam_pos;

#if ENABLE_DEPTH_PREPASS
    VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, ray_dir, MAX_STEPS, MAX_DIST, 32.0 * output_tex_size.w * deref(globals).player.cam.clip_to_view[1][1], true), ray_pos);
    daxa_u32 step_n = trace_result.step_n;
#else
    daxa_u32 step_n = 0;
#endif

    daxa_f32 depth = length(ray_pos - cam_pos);

    imageStore(daxa_image2D(render_depth_prepass_image), daxa_i32vec2(gl_GlobalInvocationID.xy), daxa_f32vec4(depth, step_n, 0, 0));
}
#undef INPUT

#endif

#if TRACE_PRIMARY_COMPUTE

#include <utils/sky.glsl>

#define PIXEL_I gl_GlobalInvocationID.xy
#define INPUT deref(gpu_input)
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    daxa_f32vec4 output_tex_size = daxa_f32vec4(deref(gpu_input).frame_dim, 0, 0);
    output_tex_size.zw = daxa_f32vec2(1.0, 1.0) / output_tex_size.xy;
    daxa_f32vec2 uv = get_uv(PIXEL_I, output_tex_size);

    ViewRayContext vrc = vrc_from_uv(globals, uv_to_ss(gpu_input, uv, output_tex_size));
    daxa_f32vec3 ray_dir = ray_dir_ws(vrc);
    daxa_f32vec3 cam_pos = ray_origin_ws(vrc);

#if !ENABLE_DEPTH_PREPASS
    daxa_f32 prepass_depth = 0.0;
    daxa_f32 prepass_steps = 0.0;
#else
    daxa_f32 max_depth = MAX_DIST;
    daxa_f32 prepass_depth = max_depth;
    daxa_f32 prepass_steps = 0.0;

    for (daxa_i32 yi = -1; yi <= 1; ++yi) {
        for (daxa_i32 xi = -1; xi <= 1; ++xi) {
            daxa_i32vec2 pt = daxa_i32vec2(PIXEL_I / PREPASS_SCL) + daxa_i32vec2(xi, yi);
            pt = clamp(pt, daxa_i32vec2(0), daxa_i32vec2(INPUT.rounded_frame_dim / PREPASS_SCL));
            daxa_f32vec2 prepass_data = texelFetch(daxa_texture2D(render_depth_prepass_image), pt, 0).xy;
            daxa_f32 loaded_depth = prepass_data.x - 1.0 / VOXEL_SCL;
            prepass_depth = max(min(prepass_depth, loaded_depth), 0);
            if (prepass_depth == loaded_depth || prepass_depth == max_depth) {
                prepass_steps = prepass_data.y / 4.0;
            }
        }
    }
#endif

    daxa_f32vec3 ray_pos = cam_pos + ray_dir * prepass_depth;

    VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, ray_dir, MAX_STEPS, MAX_DIST, 0.0, true), ray_pos);
    daxa_u32 step_n = trace_result.step_n;

    daxa_u32vec4 output_value = daxa_u32vec4(0);

    // daxa_f32 depth = length(cam_pos - ray_pos);
    // transform depth to ndc space
    vec4 vs_pos = (deref(globals).player.cam.world_to_view * vec4(ray_pos, 1));
    vec4 prev_vs_pos = (deref(globals).player.cam.world_to_view * vec4(ray_pos + trace_result.vel, 1)); // when animated objects exist, this is where they'd put their velocity
    vec4 cs_pos = (deref(globals).player.cam.view_to_sample * vs_pos);
    daxa_f32 depth = cs_pos.z / cs_pos.w;
    daxa_f32vec3 vs_nrm = daxa_f32vec3(0);
    daxa_f32vec3 vs_velocity = daxa_f32vec3(0);

    daxa_u32 thread2x2_root_index = gl_SubgroupInvocationID & (~(8 | 1));
    bool is_valid = true;
    if (trace_result.dist != MAX_DIST) {
        daxa_u32 lod_index = 0;
        daxa_u32vec3 chunk_n = daxa_u32vec3(1u << LOG2_CHUNKS_PER_LEVEL_PER_AXIS);
        daxa_u32 voxel_data = sample_voxel_chunk(VOXELS_BUFFER_PTRS, chunk_n, ray_pos, lod_index, trace_result.nrm * 0.5);
        is_valid = (voxel_data & 0xff000000) == 0;
    }
    vec3 old_nrm = trace_result.nrm;
    if (!is_valid) {
        trace_result.nrm = vec3(0.0);
    }
    vec3 valid_nrm = vec3(0);
    valid_nrm.x += subgroupBroadcast(trace_result.nrm.x, thread2x2_root_index + 0);
    valid_nrm.y += subgroupBroadcast(trace_result.nrm.y, thread2x2_root_index + 0);
    valid_nrm.z += subgroupBroadcast(trace_result.nrm.z, thread2x2_root_index + 0);
    valid_nrm.x += subgroupBroadcast(trace_result.nrm.x, thread2x2_root_index + 1);
    valid_nrm.y += subgroupBroadcast(trace_result.nrm.y, thread2x2_root_index + 1);
    valid_nrm.z += subgroupBroadcast(trace_result.nrm.z, thread2x2_root_index + 1);
    valid_nrm.x += subgroupBroadcast(trace_result.nrm.x, thread2x2_root_index + 8);
    valid_nrm.y += subgroupBroadcast(trace_result.nrm.y, thread2x2_root_index + 8);
    valid_nrm.z += subgroupBroadcast(trace_result.nrm.z, thread2x2_root_index + 8);
    valid_nrm.x += subgroupBroadcast(trace_result.nrm.x, thread2x2_root_index + 9);
    valid_nrm.y += subgroupBroadcast(trace_result.nrm.y, thread2x2_root_index + 9);
    valid_nrm.z += subgroupBroadcast(trace_result.nrm.z, thread2x2_root_index + 9);
    if (!is_valid) {
        trace_result.nrm = normalize(valid_nrm);
    }
    if (dot(valid_nrm, valid_nrm) == 0.0) {
        trace_result.nrm = old_nrm;
    }

    if (trace_result.dist == MAX_DIST) {
        // daxa_f32vec3 sky_col = get_far_sky_color_sun(sky_lut, ray_dir);
        // output_value.w = daxa_f32vec3_to_uint_urgb9e5(sky_col);
        output_value.y |= nrm_to_u16(daxa_f32vec3(0, 0, 1));
        depth = 0.0;
    } else {
        output_value.x = trace_result.voxel_data;
        // daxa_f32vec3 o = abs(fract(ray_pos * 8 - 0.5) - 0.5);
        // float z = 0.1;
        // float f =
        //     min(smoothstep(0.0, z, o.x), smoothstep(0.0, z, o.y)) +
        //     min(smoothstep(0.0, z, o.x), smoothstep(0.0, z, o.z)) +
        //     min(smoothstep(0.0, z, o.y), smoothstep(0.0, z, o.z));
        // daxa_f32vec4 voxel_color = uint_rgba8_to_f32vec4(trace_result.voxel_data);
        // output_value.x = daxa_f32vec4_to_uint_rgba8(daxa_f32vec4(mix(vec3(0.02), voxel_color.rgb, f), 1)); // trace_result.voxel_data;
        output_value.y |= nrm_to_u16(trace_result.nrm);
        vs_nrm = (deref(globals).player.cam.world_to_view * daxa_f32vec4(trace_result.nrm, 0)).xyz;
        vs_velocity = (prev_vs_pos.xyz / prev_vs_pos.w) - (vs_pos.xyz / vs_pos.w);
    }
    output_value.z = floatBitsToUint(depth);

    imageStore(daxa_uimage2D(g_buffer_image_id), daxa_i32vec2(gl_GlobalInvocationID.xy), output_value);
    imageStore(daxa_image2D(vs_normal_image_id), daxa_i32vec2(gl_GlobalInvocationID.xy), daxa_f32vec4(vs_nrm * 0.5 + 0.5, 0));
    imageStore(daxa_image2D(velocity_image_id), daxa_i32vec2(gl_GlobalInvocationID.xy), daxa_f32vec4(vs_velocity, 0));
    imageStore(daxa_image2D(depth_image_id), daxa_i32vec2(gl_GlobalInvocationID.xy), daxa_f32vec4(depth, 0, 0, 0));
}
#undef INPUT

#endif
