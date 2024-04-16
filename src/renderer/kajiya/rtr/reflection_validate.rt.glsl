#define DAXA_RAY_TRACING 1
#extension GL_EXT_ray_tracing : enable

#include <renderer/kajiya/rtr.inl>
DAXA_DECL_PUSH_CONSTANT(RtrValidateRtPush, push)
#include <renderer/rt.glsl>

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_RAYGEN

layout(location = PAYLOAD_LOC) rayPayloadEXT RayPayload prd;

// #include <utilities/gpu/uv.glsl>
// #include <utilities/gpu/pack_unpack.glsl>
// #include <renderer/kajiya/inc/frame_constants.glsl>
#include <renderer/kajiya/inc/gbuffer.glsl>
#include <renderer/kajiya/inc/brdf.glsl>
#include <renderer/kajiya/inc/brdf_lut.glsl>
#include <renderer/kajiya/inc/layered_brdf.glsl>
#include "blue_noise.glsl"
#include <renderer/kajiya/inc/rt.glsl>
// #include <utilities/gpu/atmosphere.glsl>
// #include <utilities/gpu/sun.glsl>
// #include <utilities/gpu/lights/triangle.glsl>
#include <renderer/kajiya/inc/reservoir.glsl>
// #include "../ircache/bindings.hlsl"
// #include "../wrc/bindings.hlsl"
#include "rtr_settings.glsl"

// #define IRCACHE_LOOKUP_KEEP_ALIVE_PROB 0.125
#include "../ircache/lookup.glsl"
// #include "../wrc/lookup.hlsl"

#include "reflection_trace_common.inc.glsl"
#include <renderer/kajiya/inc/downscale.glsl>
#include <renderer/kajiya/inc/safety.glsl>

void main() {
#undef HALFRES_SUBSAMPLE_INDEX
#define HALFRES_SUBSAMPLE_INDEX (deref(push.uses.gpu_input).frame_index & 3)
    if (RTR_RESTIR_USE_PATH_VALIDATION == 0) {
        return;
    }

    // Validation at half-res
    const uvec2 px = gl_LaunchIDEXT.xy * 2 + HALFRES_SUBSAMPLE_OFFSET;
    // const uvec2 px = DispatchRaysIndex().xy;

    // Standard jitter from the other reflection passes
    const uvec2 hi_px = px * 2 + HALFRES_SUBSAMPLE_OFFSET;
    float depth = safeTexelFetch(push.uses.depth_tex, ivec2(hi_px), 0).r;

    // refl_restir_invalidity_tex[px] = 0;

    if (0.0 == depth) {
        safeImageStoreU(push.uses.refl_restir_invalidity_tex, ivec2(px), uvec4(1));
        return;
    }

    const vec2 uv = get_uv(hi_px, push.gbuffer_tex_size);

    GbufferData gbuffer = unpack(GbufferDataPacked(safeTexelFetchU(push.uses.gbuffer_tex, ivec2(hi_px), 0)));
    gbuffer.roughness = max(gbuffer.roughness, RTR_ROUGHNESS_CLAMP);

    const mat3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);

#if RTR_USE_TIGHTER_RAY_BIAS
    const ViewRayContext view_ray_context = vrc_from_uv_and_biased_depth(push.uses.gpu_input, uv, depth);
    const vec3 refl_ray_origin_ws = biased_secondary_ray_origin_ws_with_normal(view_ray_context, gbuffer.normal);
#else
    const ViewRayContext view_ray_context = vrc_from_uv_and_depth(push.uses.gpu_input, uv, depth);
    const vec3 refl_ray_origin_ws = biased_secondary_ray_origin_ws(view_ray_context);
#endif

    // TODO: frame consistency
    const uint noise_offset = deref(push.uses.gpu_input).frame_index * select(USE_TEMPORAL_JITTER, 1, 0);

    vec3 ray_orig_ws = safeImageLoad(push.uses.ray_orig_history_tex, ivec2(px)).xyz + get_prev_eye_position(push.uses.gpu_input);
    vec3 ray_hit_ws = safeImageLoad(push.uses.ray_history_tex, ivec2(px)).xyz + ray_orig_ws;

    // NOTE(grundlett): Here we fix the ray origin/hit for when the player causes the world to wrap.
    // We technically don't need to write out if the world does not wrap in this frame, but IDC for now.
    vec3 offset = vec3(deref(push.uses.gpu_input).player.player_unit_offset - deref(push.uses.gpu_input).player.prev_unit_offset);
    ray_orig_ws -= offset;
    ray_hit_ws -= offset;
    safeImageStore(push.uses.ray_orig_history_tex, ivec2(px), vec4(ray_orig_ws, 0));
    safeImageStore(push.uses.ray_history_tex, ivec2(px), vec4(ray_hit_ws, 0));

    RayDesc outgoing_ray;
    outgoing_ray.Direction = normalize(ray_hit_ws - ray_orig_ws);
    outgoing_ray.Origin = ray_orig_ws;
    outgoing_ray.TMin = 0;
    outgoing_ray.TMax = SKY_DIST;

    // uint rng = hash2(px);
    uint rng = safeTexelFetchU(push.uses.rng_history_tex, ivec2(px), 0).x;
    RtrTraceResult result = do_the_thing(px, gbuffer.normal, gbuffer.roughness, rng, outgoing_ray);

    Reservoir1spp r = Reservoir1spp_from_raw(safeImageLoadU(push.uses.reservoir_history_tex, ivec2(px)).xy);

    const vec4 prev_irradiance_packed = safeImageLoad(push.uses.irradiance_history_tex, ivec2(px));
    const vec3 prev_irradiance = max(0.0.xxx, prev_irradiance_packed.rgb * deref(push.uses.gpu_input).pre_exposure_delta);
    const vec3 check_radiance = max(0.0.xxx, result.total_radiance);

    const float rad_diff = length(abs(prev_irradiance - check_radiance) / max(vec3(1e-3), prev_irradiance + check_radiance));
    const float invalidity = smoothstep(0.1, 0.5, rad_diff / length(1.0.xxx));

    // r.M = max(0, min(r.M, exp2(log2(float(RTR_RESTIR_TEMPORAL_M_CLAMP)) * (1.0 - invalidity))));
    r.M *= 1 - invalidity;

    // TODO: also update hit point and normal
    // TODO: does this also need a hit_t check as in rtdgi restir validation?
    // TOD:: rename to radiance
    safeImageStore(push.uses.irradiance_history_tex, ivec2(px), vec4(check_radiance, prev_irradiance_packed.a));

    safeImageStore(push.uses.refl_restir_invalidity_tex, ivec2(px), vec4(invalidity));
    safeImageStoreU(push.uses.reservoir_history_tex, ivec2(px), uvec4(as_raw(r), 0, 0));

// Also reduce M of the neighbors in case we have fewer validation rays than irradiance rays.
#if 1
    for (uint i = 1; i <= 3; ++i) {
        // const uvec2 main_px = px;
        // const uvec2 px = (main_px & ~1u) + HALFRES_SUBSAMPLE_OFFSET;
        const uvec2 px = gl_LaunchIDEXT.xy * 2 + hi_px_subpixels[(deref(push.uses.gpu_input).frame_index + i) & 3];

        const vec4 neighbor_prev_irradiance_packed = safeImageLoad(push.uses.irradiance_history_tex, ivec2(px));
        {
            const vec3 a = max(0.0.xxx, neighbor_prev_irradiance_packed.rgb * deref(push.uses.gpu_input).pre_exposure_delta);
            const vec3 b = prev_irradiance;
            const float neigh_rad_diff = length(abs(a - b) / max(vec3(1e-8), a + b));

            // If the neighbor and us tracked similar radiance, assume it would also have
            // a similar change in value upon validation.
            if (neigh_rad_diff < 0.2) {
                // With this assumption, we'll replace the neighbor's old radiance with our own new one.
                safeImageStore(push.uses.irradiance_history_tex, ivec2(px), vec4(check_radiance, neighbor_prev_irradiance_packed.a));
            }
        }

        safeImageStore(push.uses.refl_restir_invalidity_tex, ivec2(px), vec4(invalidity));

        if (invalidity > 0) {
            Reservoir1spp r = Reservoir1spp_from_raw(safeImageLoadU(push.uses.reservoir_history_tex, ivec2(px)).xy);
            // r.M = max(0, min(r.M, exp2(log2(float(RTR_RESTIR_TEMPORAL_M_CLAMP)) * (1.0 - invalidity))));
            r.M *= 1 - invalidity;
            safeImageStoreU(push.uses.reservoir_history_tex, ivec2(px), uvec4(as_raw(r), 0, 0));
        }
    }
#endif
}
#endif
