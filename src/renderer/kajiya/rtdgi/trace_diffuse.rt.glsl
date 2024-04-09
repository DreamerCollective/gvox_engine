#define DAXA_RAY_TRACING 1
#extension GL_EXT_ray_tracing : enable

#include <renderer/kajiya/rtdgi.inl>
DAXA_DECL_PUSH_CONSTANT(RtdgiTraceRtPush, push)
#include <renderer/rt.glsl>

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_RAYGEN

layout(location = PAYLOAD_LOC) rayPayloadEXT RayPayload prd;

#include <utilities/gpu/math.glsl>
// #include <utilities/gpu/uv.glsl>
// #include <utilities/gpu/pack_unpack.glsl>
// #include <renderer/kajiya/inc/frame_constants.glsl>
#include <renderer/kajiya/inc/gbuffer.glsl>
#include <renderer/kajiya/inc/brdf.glsl>
#include <renderer/kajiya/inc/brdf_lut.glsl>
#include <renderer/kajiya/inc/layered_brdf.glsl>
// #include <utilities/gpu/blue_noise.glsl>
#include <renderer/kajiya/inc/rt.glsl>
// #include <utilities/gpu/atmosphere.glsl>
// #include <utilities/gpu/sun.glsl>
// #include <utilities/gpu/lights/triangle.glsl>
#include <renderer/kajiya/inc/reservoir.glsl>
// #include "../ircache/bindings.hlsl"
// #include "../wrc/bindings.hlsl"
#include "../rtr/rtr_settings.glsl"
#include "rtdgi_restir_settings.glsl"
#include "near_field_settings.glsl"

// #define IRCACHE_LOOKUP_DONT_KEEP_ALIVE
// #define IRCACHE_LOOKUP_KEEP_ALIVE_PROB 0.125

#include "../ircache/lookup.glsl"
// #include "../wrc/lookup.glsl"
#include "candidate_ray_dir.glsl"

#include "diffuse_trace_common.inc.glsl"
#include <renderer/kajiya/inc/downscale.glsl>
#include <renderer/kajiya/inc/safety.glsl>

void main() {
    const uvec2 px = gl_LaunchIDEXT.xy;
#undef HALFRES_SUBSAMPLE_INDEX
#define HALFRES_SUBSAMPLE_INDEX (deref(push.uses.gpu_input).frame_index & 3)
    const ivec2 hi_px_offset = ivec2(HALFRES_SUBSAMPLE_OFFSET);
    const uvec2 hi_px = px * 2 + hi_px_offset;

    float depth = safeTexelFetch(push.uses.depth_tex, ivec2(hi_px), 0).r;

    if (0.0 == depth) {
        safeImageStore(push.uses.candidate_irradiance_out_tex, ivec2(px), vec4(0));
        safeImageStore(push.uses.candidate_normal_out_tex, ivec2(px), vec4(0, 0, 1, 0));
        safeImageStore(push.uses.rt_history_invalidity_out_tex, ivec2(px), vec4(0));
        return;
    }

    const vec2 uv = get_uv(hi_px, push.gbuffer_tex_size);
    const ViewRayContext view_ray_context = vrc_from_uv_and_biased_depth(push.uses.gpu_input, uv, depth);

    const float NEAR_FIELD_FADE_OUT_END = -ray_hit_vs(view_ray_context).z * (SSGI_NEAR_FIELD_RADIUS * push.gbuffer_tex_size.w * 0.5);

#if RTDGI_INTERLEAVED_VALIDATION_ALWAYS_TRACE_NEAR_FIELD
    if (true)
#else
    if (is_rtdgi_tracing_frame(deref(push.uses.gpu_input).frame_index))
#endif
    {
        const vec3 normal_vs = safeTexelFetch(push.uses.half_view_normal_tex, ivec2(px), 0).xyz;
        const vec3 normal_ws = direction_view_to_world(push.uses.gpu_input, normal_vs);
        const mat3 tangent_to_world = build_orthonormal_basis(normal_ws);
        const vec3 outgoing_dir = rtdgi_candidate_ray_dir(push.uses.blue_noise_vec2, deref(push.uses.gpu_input).frame_index, px, tangent_to_world);

        RayDesc outgoing_ray;
        outgoing_ray.Direction = outgoing_dir;
        outgoing_ray.Origin = biased_secondary_ray_origin_ws_with_normal(view_ray_context, normal_ws);
        outgoing_ray.TMin = 0;

        if (is_rtdgi_tracing_frame(deref(push.uses.gpu_input).frame_index)) {
            outgoing_ray.TMax = SKY_DIST;
        } else {
            outgoing_ray.TMax = NEAR_FIELD_FADE_OUT_END;
        }

        uint rng = hash3(uvec3(px, deref(push.uses.gpu_input).frame_index & 31));
        TraceResult result = do_the_thing(px, normal_ws, rng, outgoing_ray);

#if RTDGI_INTERLEAVED_VALIDATION_ALWAYS_TRACE_NEAR_FIELD
        if (!is_rtdgi_tracing_frame(deref(push.uses.gpu_input).frame_index) && !result.is_hit) {
            // If we were only tracing short rays, make sure we don't try to output
            // sky color upon misses.
            result.out_value = vec3(0);
            result.hit_t = SKY_DIST;
        }
#endif

        const vec3 hit_offset_ws = outgoing_ray.Direction * result.hit_t;

        const float cos_theta = dot(normalize(outgoing_dir - ray_dir_ws(view_ray_context)), normal_ws);
        safeImageStore(push.uses.candidate_irradiance_out_tex, ivec2(px), vec4(result.out_value, rtr_encode_cos_theta_for_fp16(cos_theta)));
        safeImageStore(push.uses.candidate_hit_out_tex, ivec2(px), vec4(hit_offset_ws, result.pdf * select(is_rtdgi_tracing_frame(deref(push.uses.gpu_input).frame_index), 1, -1)));
        safeImageStore(push.uses.candidate_normal_out_tex, ivec2(px), vec4(direction_world_to_view(push.uses.gpu_input, result.hit_normal_ws), 0));
    }
    // } else {
    //     const vec4 reproj = reprojection_tex[hi_px];
    //     const ivec2 reproj_px = floor(px + gbuffer_tex_size.xy * reproj.xy / 2 + 0.5);
    //     candidate_irradiance_out_tex[px] = 0.0;
    //     candidate_hit_out_tex[px] = 0.0;
    //     candidate_normal_out_tex[px] = 0.0;
    // }

    const vec4 reproj = safeTexelFetch(push.uses.reprojection_tex, ivec2(hi_px), 0);
    const ivec2 reproj_px = ivec2(floor(vec2(px) + push.gbuffer_tex_size.xy * reproj.xy / 2.0 + 0.5));
    safeImageStore(push.uses.rt_history_invalidity_out_tex, ivec2(px), safeTexelFetch(push.uses.rt_history_invalidity_in_tex, ivec2(reproj_px), 0));
}
#endif
