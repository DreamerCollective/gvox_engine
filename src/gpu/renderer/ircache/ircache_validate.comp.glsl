#include <shared/app.inl>

#include <utils/math.glsl>
// #include <utils/pack_unpack.glsl>
// #include <utils/frame_constants.glsl>
#include <utils/gbuffer.glsl>
#include <utils/brdf.glsl>
#include <utils/brdf_lut.glsl>
#include <utils/layered_brdf.glsl>
#include <utils/rt.glsl>
#include <utils/random.glsl>
#include <utils/quasi_random.glsl>
#include <utils/reservoir.glsl>
// #include <utils/bindless_textures.glsl>
// #include <utils/atmosphere.glsl>
#include <utils/normal.glsl>
// #include <utils/lights/triangle.glsl>
#include <utils/color.glsl>

// #include "../inc/sun.hlsl"
// #include "../wrc/lookup.hlsl"

// Sample straight from the `ircache_aux_buf` instead of the SH.
#define IRCACHE_LOOKUP_PRECISE
#include "lookup.glsl"

#include "ircache_sampler_common.inc.glsl"
#include "ircache_trace_common.inc.glsl"

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
void main() {
    if (IRCACHE_FREEZE) {
        return;
    }

    const uint dispatch_idx = gl_GlobalInvocationID.x;

// AMD ray-tracing bug workaround; indirect RT seems to be tracing with the same
// ray count for multiple dispatches (???)
// Search for c804a814-fdc8-4843-b2c8-9d0674c10a6f for other occurences.
#if 1
    const uint alloc_count = deref(ircache_meta_buf[IRCACHE_META_TRACING_ALLOC_COUNT_INDEX]);
    if (dispatch_idx >= alloc_count * IRCACHE_VALIDATION_SAMPLES_PER_FRAME) {
        return;
    }
#endif

    const uint entry_idx = deref(ircache_entry_indirection_buf[dispatch_idx / IRCACHE_VALIDATION_SAMPLES_PER_FRAME]);
    const uint sample_idx = dispatch_idx % IRCACHE_VALIDATION_SAMPLES_PER_FRAME;
    const uint life = deref(ircache_life_buf[entry_idx * 4]);

    DiffuseBrdf brdf;
    brdf.albedo = 1.0.xxx;

    const SampleParams sample_params = SampleParams_from_spf_entry_sample_frame(
        IRCACHE_VALIDATION_SAMPLES_PER_FRAME,
        entry_idx,
        sample_idx,
        deref(gpu_input).frame_index);

    const uint octa_idx = octa_idx(sample_params);
    const uint output_idx = entry_idx * IRCACHE_AUX_STRIDE + octa_idx;

    float invalidity = 0;

    {
        // TODO: wz not used. slim down.
        Reservoir1spp r = Reservoir1spp_from_raw(floatBitsToUint(deref(ircache_aux_buf[output_idx]).xy));

        if (r.M > 0) {
            vec4 prev_value_and_count =
                deref(ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2]) * vec4((deref(gpu_input).pre_exposure_delta).xxx, 1);

            Vertex prev_entry = unpack_vertex(VertexPacked(deref(ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2 * 2])));

            // Validate the previous sample
            IrcacheTraceResult prev_traced = ircache_trace(prev_entry, brdf, SampleParams_from_raw(r.payload), life);

            const float prev_self_lighting_limiter =
                select(USE_SELF_LIGHTING_LIMITER, mix(0.5, 1, smoothstep(-0.1, 0, dot(prev_traced.direction, prev_entry.normal))), 1.0);

            const vec3 a = prev_traced.incident_radiance * prev_self_lighting_limiter;
            const vec3 b = prev_value_and_count.rgb;
            const vec3 dist3 = abs(a - b) / (a + b);
            const float dist = max(dist3.r, max(dist3.g, dist3.b));
            invalidity = smoothstep(0.1, 0.5, dist);
            r.M = max(0, min(r.M, exp2(log2(float(IRCACHE_RESTIR_M_CLAMP)) * (1.0 - invalidity))));

            // Update the stored value too.
            // TODO: try the update heuristics from the diffuse trace
            prev_value_and_count.rgb = a;

            deref(ircache_aux_buf[output_idx]).xy = uintBitsToFloat(as_raw(r));
            deref(ircache_aux_buf[output_idx + IRCACHE_OCTA_DIMS2]) = prev_value_and_count;
        }
    }

// Also reduce M of the neighbors in case we have fewer validation rays than irradiance rays.
#if 1
    if (IRCACHE_VALIDATION_SAMPLES_PER_FRAME < IRCACHE_SAMPLES_PER_FRAME) {
        if (invalidity > 0) {
            const uint PERIOD = IRCACHE_OCTA_DIMS2 / IRCACHE_VALIDATION_SAMPLES_PER_FRAME;
            const uint OTHER_PERIOD = IRCACHE_OCTA_DIMS2 / IRCACHE_SAMPLES_PER_FRAME;

            for (uint xor = OTHER_PERIOD; xor < PERIOD; xor *= 2) {
                const uint idx = output_idx ^ xor;
                Reservoir1spp r = Reservoir1spp_from_raw(floatBitsToUint(deref(ircache_aux_buf[idx]).xy));
                r.M = max(0, min(r.M, exp2(log2(float(IRCACHE_RESTIR_M_CLAMP)) * (1.0 - invalidity))));
                deref(ircache_aux_buf[idx]).xy = uintBitsToFloat(as_raw(r));
            }
        }
    }
#endif
}
