#pragma once

#include <voxels/particles/particle.glsl>

ParticleVertex get_sim_particle_vertex(daxa_BufferPtr(GpuInput) gpu_input, daxa_BufferPtr(SimulatedVoxelParticle) simulated_voxel_particles, PackedParticleVertex packed_vertex) {
    uint index = packed_vertex.data;

    SimulatedVoxelParticle self = deref(advance(simulated_voxel_particles, index));

    ParticleVertex result;
    result.pos = self.pos;
    result.prev_pos = self.pos - self.vel * deref(gpu_input).delta_time;
    result.packed_voxel = self.packed_voxel;

    result.pos = get_particle_worldspace_origin(gpu_input, result.pos);
    result.prev_pos = get_particle_prev_worldspace_origin(gpu_input, result.prev_pos);

    return result;
}
