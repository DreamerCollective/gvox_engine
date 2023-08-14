#pragma once

#include <shared/app.inl>
#include <utils/math.glsl>
#include <voxels/core.glsl>

#define PARTICLE_ALIVE_FLAG (1 << 0)
#define PARTICLE_SMOKE_FLAG (1 << 1)

#define PARTICLE_SLEEP_TIMER_OFFSET (8)
#define PARTICLE_SLEEP_TIMER_MASK (0xff << PARTICLE_SLEEP_TIMER_OFFSET)

void particle_spawn(in out SimulatedVoxelParticle self, u32 index) {
    rand_seed(index);

    self.duration_alive = 0.0 + rand() * 0;
    self.flags = PARTICLE_ALIVE_FLAG;

    self.pos = f32vec3(good_rand(deref(gpu_input).time + 137.41) * 10 + 20, good_rand(deref(gpu_input).time + 41.137) * 10 + 20, 70.0);
    self.vel = deref(globals).player.forward * 0 + vec3(0, 0, -10);
    // self.pos = deref(globals).player.pos + deref(globals).player.forward * 1 + vec3(0, 0, -2.5) + deref(globals).player.lateral * 3.5;
    // self.vel = deref(globals).player.forward * 3 + rand_dir() * 2 + deref(globals).player.vel;
}

void particle_update(in out SimulatedVoxelParticle self, daxa_BufferPtr(GpuInput) gpu_input, in out bool should_place) {
    rand_seed(gl_GlobalInvocationID.x + uint((deref(gpu_input).time) * 13741));

    if ((self.flags & PARTICLE_ALIVE_FLAG) == 0) {
        particle_spawn(self, gl_GlobalInvocationID.x);
        return;
    }

    float dt = 0.01; // deref(gpu_input).delta_time;
    self.duration_alive += dt;

    if ((self.flags & PARTICLE_SMOKE_FLAG) != 0) {
        self.vel += (f32vec3(0.0, 0.0, 1.0) + rand_dir() * 5.1) * dt;
    } else {
        self.vel += f32vec3(0.0, 0.0, -9.8) * dt;
    }

    float curr_speed = length(self.vel);
    float curr_dist_in_dt = curr_speed * dt;
    vec3 ray_pos = self.pos;

    VoxelTraceResult trace_result = voxel_trace(VoxelTraceInfo(VOXELS_BUFFER_PTRS, self.vel / curr_speed, MAX_STEPS, curr_dist_in_dt, 0.0, true), ray_pos);
    float dist = trace_result.dist;

    if (!(dist > curr_dist_in_dt)) {
        self.pos += self.vel / curr_speed * (dist - 0.001);
        vec3 nrm = trace_result.nrm;
        const float bounciness = 0.1;
        const float dampening = 0.01;
        // self.vel = normalize(reflect(self.vel, nrm) * (0.5 + 0.5 * bounciness) + self.vel * (0.5 - 0.5 * bounciness)) * curr_speed * dampening;
        self.vel = reflect(self.vel, nrm) * bounciness;
        // if (good_rand(gl_GlobalInvocationID.x) < 0.1) {
        //     self.vel *= 0.2;
        //     self.flags |= PARTICLE_SMOKE_FLAG;
        // }
    } else {
        self.pos += self.vel * dt;
    }

    if (min(dist, curr_dist_in_dt) < 0.01 / VOXEL_SCL) {
        self.flags += 1 << PARTICLE_SLEEP_TIMER_OFFSET;
        if (((self.flags & PARTICLE_SLEEP_TIMER_MASK) >> PARTICLE_SLEEP_TIMER_OFFSET) >= 15) {
            should_place = true;
            self.flags &= ~PARTICLE_ALIVE_FLAG & ~PARTICLE_SLEEP_TIMER_MASK;
        }
    } else {
        self.flags &= ~PARTICLE_SLEEP_TIMER_MASK;
    }

    if (self.duration_alive > 0.0) { // if (self.pos.z < 0.0) {
        particle_spawn(self, gl_GlobalInvocationID.x);
    }
}