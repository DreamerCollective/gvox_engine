#include "sim_particle.inl"
#include "sim_particle.glsl"

DAXA_DECL_PUSH_CONSTANT(SimParticleSimComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(VoxelParticlesState) particles_state = push.uses.particles_state;
daxa_RWBufferPtr(SimulatedVoxelParticle) simulated_voxel_particles = push.uses.simulated_voxel_particles;
daxa_RWBufferPtr(PackedParticleVertex) cube_rendered_particle_verts = push.uses.cube_rendered_particle_verts;
daxa_RWBufferPtr(PackedParticleVertex) shadow_cube_rendered_particle_verts = push.uses.shadow_cube_rendered_particle_verts;
daxa_RWBufferPtr(PackedParticleVertex) splat_rendered_particle_verts = push.uses.splat_rendered_particle_verts;

#include <renderer/rt.glsl>

void particle_spawn(in out SimulatedVoxelParticle self, uint index, daxa_BufferPtr(GpuInput) gpu_input) {
    rand_seed(index);

    self.duration_alive = 0.0 + rand() * 1.0;
    self.flags = PARTICLE_ALIVE_FLAG;

    vec3 look_dir = deref(gpu_input).player.forward;

    self.pos = deref(gpu_input).player.pos + deref(gpu_input).player.player_unit_offset + look_dir * 1 + deref(gpu_input).player.lateral * 0.25;
    self.vel = look_dir * 18 + rand_dir() * 2; // + deref(gpu_input).player.vel;

    vec3 col = vec3(0.3, 0.4, 0.9) * (rand() * 0.5 + 0.5); // vec3(0.5);
    vec3 nrm = vec3(0, 0, 1);
    Voxel particle_voxel = Voxel(1, 0.99, nrm, col);
    self.packed_voxel = pack_voxel(particle_voxel);
}

void particle_voxelize(daxa_RWBufferPtr(uint) placed_voxel_particles, daxa_RWBufferPtr(VoxelParticlesState) particles_state, in SimulatedVoxelParticle self, uint self_index) {
    uvec3 my_voxel_i = uvec3(self.pos * VOXEL_SCL);
    const uvec3 max_pos = uvec3(2048);
    if (my_voxel_i.x < max_pos.x && my_voxel_i.y < max_pos.y && my_voxel_i.z < max_pos.z) {
        // Commented out, since placing particles in the voxel volume is not well optimized yet.

        // uint my_place_index = atomicAdd(deref(particles_state).place_count, 1);
        // if (my_place_index == 0) {
        //     ChunkWorkItem brush_work_item;
        //     brush_work_item.i = uvec3(0);
        //     brush_work_item.brush_id = BRUSH_FLAGS_PARTICLE_BRUSH;
        //     brush_work_item.brush_input = deref(particles_state).brush_input;
        //     zero_work_item_children(brush_work_item);
        //     queue_root_work_item(particles_state, brush_work_item);
        // }
        // deref(advance(placed_voxel_particles, my_place_index)) = self_index;
        // atomicMin(deref(particles_state).place_bounds_min.x, my_voxel_i.x);
        // atomicMin(deref(particles_state).place_bounds_min.y, my_voxel_i.y);
        // atomicMin(deref(particles_state).place_bounds_min.z, my_voxel_i.z);
        // atomicMax(deref(particles_state).place_bounds_max.x, my_voxel_i.x);
        // atomicMax(deref(particles_state).place_bounds_max.y, my_voxel_i.y);
        // atomicMax(deref(particles_state).place_bounds_max.z, my_voxel_i.z);
    }
}

void particle_update(in out SimulatedVoxelParticle self, uint self_index, daxa_BufferPtr(GpuInput) gpu_input, in out bool should_place) {
    rand_seed(self_index + uint((deref(gpu_input).time) * 13741));

    // if (deref(gpu_input).frame_index == 0) {
    //     particle_spawn(self, self_index, gpu_input);
    // }

    if (deref(gpu_input).actions[GAME_ACTION_INTERACT0] != 0 && (rand() < 0.01)) {
        particle_spawn(self, self_index, gpu_input);
    }

    // if (PER_VOXEL_NORMALS != 0) {
    //     Voxel particle_voxel = unpack_voxel(self.packed_voxel);
    //     particle_voxel.normal = normalize(deref(gpu_input).player.pos - get_particle_worldspace_pos(gpu_input, self.pos));
    //     self.packed_voxel = pack_voxel(particle_voxel);
    // }

    const bool PARTICLES_PAUSED = false;

    if (!PARTICLES_PAUSED) {
        if ((self.flags & PARTICLE_ALIVE_FLAG) == 0) {
            // particle_spawn(self, self_index, gpu_input);
            return;
        }

        float dt = min(deref(gpu_input).delta_time, 0.01);
        self.duration_alive += dt;

        if ((self.flags & PARTICLE_SMOKE_FLAG) != 0) {
            self.vel += (vec3(0.0, 0.0, 1.0) + rand_dir() * 5.1) * dt;
        } else {
            self.vel += vec3(0.0, 0.0, -9.8) * dt;
        }

        float curr_speed = length(self.vel);
        float curr_dist_in_dt = curr_speed * dt;
        vec3 ray_pos = get_particle_worldspace_pos(gpu_input, self.pos);

        VoxelTraceResult trace_result = voxel_trace(VoxelRtTraceInfo(VOXELS_RT_BUFFER_PTRS, self.vel / curr_speed, curr_dist_in_dt), ray_pos);
        float dist = trace_result.dist;

        if (dist < curr_dist_in_dt) {
            vec3 nrm = trace_result.nrm;
            if (abs(dot(nrm, nrm) - 1.0) > 0.1) {
                nrm = vec3(0, 0, 1);
            }
            nrm = normalize(nrm);
            self.pos += self.vel / curr_speed * (dist - 0.001) + nrm * 0.87 * VOXEL_SIZE;

            const float bounciness = 0.35;
            const float dampening = 0.001;
            // self.vel = normalize(reflect(self.vel, nrm) * (0.5 + 0.5 * bounciness) + self.vel * (0.5 - 0.5 * bounciness)) * curr_speed * dampening;
            self.vel = reflect(self.vel, nrm) * bounciness;
            // if (good_rand(self_index) < 0.1) {
            //     self.vel *= 0.2;
            //     self.flags |= PARTICLE_SMOKE_FLAG;
            // }
        } else {
            self.pos += self.vel * dt;
        }

        if (min(dist, curr_dist_in_dt) < 0.01 * VOXEL_SIZE) {
            self.flags += 1 << PARTICLE_SLEEP_TIMER_OFFSET;
            if (((self.flags & PARTICLE_SLEEP_TIMER_MASK) >> PARTICLE_SLEEP_TIMER_OFFSET) >= 200) {
                should_place = true;
                self.flags &= ~PARTICLE_ALIVE_FLAG & ~PARTICLE_SLEEP_TIMER_MASK;
            }
        } else {
            self.flags &= ~PARTICLE_SLEEP_TIMER_MASK;
        }

        // if (self.duration_alive > 5.0) {
        //     particle_spawn(self, self_index, gpu_input);
        // }
    } else {
        self.vel = vec3(0);
        // vec3 col = pow(vec3(105, 126, 78) / 255.0, vec3(2.2));
        // Voxel particle_voxel = Voxel(1, 0.99, vec3(0, 0, 1), col);
        // self.packed_voxel = pack_voxel(particle_voxel);
        // rand_seed(self_index);
        // self.pos = deref(gpu_input).player.pos + deref(gpu_input).player.player_unit_offset + vec3(rand(), rand(), 0) * 5 + vec3(0, 0, -1); // deref(gpu_input).player.forward * 1 + vec3(0, 0, -2.5) + deref(gpu_input).player.lateral * 1.5;
    }
}

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint particle_index = gl_GlobalInvocationID.x;
    SimulatedVoxelParticle self = deref(advance(simulated_voxel_particles, particle_index));

    bool should_place = false;
    particle_update(self, particle_index, gpu_input, should_place);

    deref(advance(simulated_voxel_particles, particle_index)) = self;

    // if (should_place) {
    //     particle_voxelize(placed_voxel_particles, particles_state, self, particle_index);
    // }

    if (self.flags != 0) {
        PackedParticleVertex packed_vertex = PackedParticleVertex(particle_index);
        ParticleVertex vert = get_sim_particle_vertex(gpu_input, daxa_BufferPtr(SimulatedVoxelParticle)(as_address(simulated_voxel_particles)), packed_vertex);
        particle_render(cube_rendered_particle_verts, shadow_cube_rendered_particle_verts, splat_rendered_particle_verts, particles_state, gpu_input, vert, packed_vertex, true);
    }
}
