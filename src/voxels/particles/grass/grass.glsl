#pragma once

#include <utilities/gpu/noise.glsl>
#include <g_samplers>
#include <g_value_noise>

#include "../particle.glsl"

vec2 grass_get_rot_offset(in out GrassStrand self, float time) {
    FractalNoiseConfig noise_conf = FractalNoiseConfig(
        /* .amplitude   = */ 1.0,
        /* .persistance = */ 0.3,
        /* .scale       = */ 0.15,
        /* .lacunarity  = */ 2.5,
        /* .octaves     = */ 4);
    vec4 noise_val = fractal_noise(g_value_noise_tex, g_sampler_llr, self.origin + vec3(time * 0.5, sin(time * 1.0), 0), noise_conf);
    float rot = noise_val.x * 43.0;
    // float rot = time * 1.5 + self.origin.x * (sin(time * 0.157 + self.origin.z * 1.12) * 0.125 + 0.75) + self.origin.y * (cos(time * 0.063 + self.origin.z * 0.98) * 0.125 + 0.75);
    return vec2(sin(rot), cos(rot));
}

vec3 get_grass_offset(vec2 rot_offset, float z) {
    return vec3(rot_offset * z * 0.66, z);
}

ParticleVertex get_grass_vertex(daxa_BufferPtr(GpuInput) gpu_input, daxa_BufferPtr(GrassStrand) grass_strands, PackedParticleVertex packed_vertex) {
    // unpack indices from the vertex
    uint strand_index = (packed_vertex.data >> 0) & 0xffffff;
    uint i = (packed_vertex.data >> 24) & 0xff;

    GrassStrand self = deref(advance(grass_strands, strand_index));

    vec2 rot_offset = grass_get_rot_offset(self, deref(gpu_input).time);
    vec3 offset = get_grass_offset(rot_offset, i * VOXEL_SIZE);

    vec2 prev_rot_offset = grass_get_rot_offset(self, deref(gpu_input).time - deref(gpu_input).delta_time);
    vec3 prev_offset = get_grass_offset(prev_rot_offset, i * VOXEL_SIZE);

    Voxel voxel = unpack_voxel(self.packed_voxel);
    voxel.color = rgb2hsv(voxel.color);
    voxel.color[2] *= float(i + 1);
    voxel.color = hsv2rgb(voxel.color);

    ParticleVertex result;
    result.pos = self.origin + offset;
    result.prev_pos = self.origin + prev_offset;
    result.packed_voxel = pack_voxel(voxel);

    result.pos = get_particle_worldspace_origin(gpu_input, result.pos);
    result.prev_pos = get_particle_prev_worldspace_origin(gpu_input, result.prev_pos);

    return result;
}
