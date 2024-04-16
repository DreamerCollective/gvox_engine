#include "voxel_particles.inl"

DAXA_DECL_PUSH_CONSTANT(VoxelParticlePerframeComputePush, push)
daxa_BufferPtr(GpuInput) gpu_input = push.uses.gpu_input;
daxa_RWBufferPtr(VoxelParticlesState) particles_state = push.uses.particles_state;
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(GrassStrandAllocator, grass_allocator)
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(FlowerAllocator, flower_allocator)
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(TreeParticleAllocator, tree_particle_allocator)
SIMPLE_STATIC_ALLOCATOR_BUFFERS_PUSH_USES(FireParticleAllocator, fire_particle_allocator)

#include <renderer/kajiya/inc/camera.glsl>
#include <voxels/voxels.glsl>

#define UserAllocatorType GrassStrandAllocator
#define UserIndexType uint
#define UserMaxElementCount MAX_GRASS_BLADES
#include <utilities/allocator.glsl>

#define UserAllocatorType FlowerAllocator
#define UserIndexType uint
#define UserMaxElementCount MAX_FLOWERS
#include <utilities/allocator.glsl>

#define UserAllocatorType TreeParticleAllocator
#define UserIndexType uint
#define UserMaxElementCount MAX_TREE_PARTICLES
#include <utilities/allocator.glsl>

#define UserAllocatorType FireParticleAllocator
#define UserIndexType uint
#define UserMaxElementCount MAX_FIRE_PARTICLES
#include <utilities/allocator.glsl>

void reset_draw_params(in out IndirectDrawIndexedParams params) {
    params.index_count = 8;
    params.instance_count = 0;
    params.first_index = 0;
    params.vertex_offset = 0;
    params.first_instance = 0;
}

void reset_draw_params(in out IndirectDrawParams params) {
    params.vertex_count = 0;
    params.instance_count = 1;
    params.first_vertex = 0;
    params.first_instance = 0;
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    deref(particles_state).simulation_dispatch = uvec3(MAX_SIMULATED_VOXEL_PARTICLES / 64, 1, 1);

    deref(particles_state).place_count = 0;
    deref(particles_state).place_bounds_min = uvec3(1000000);
    deref(particles_state).place_bounds_max = uvec3(0);

    // sim particle
    reset_draw_params(deref(particles_state).sim_particle.cube_draw_params);
    reset_draw_params(deref(particles_state).sim_particle.shadow_cube_draw_params);
    reset_draw_params(deref(particles_state).sim_particle.splat_draw_params);

    // grass
    reset_draw_params(deref(particles_state).grass.cube_draw_params);
    reset_draw_params(deref(particles_state).grass.shadow_cube_draw_params);
    reset_draw_params(deref(particles_state).grass.splat_draw_params);
    GrassStrandAllocator_perframe(grass_allocator);

    // flower
    reset_draw_params(deref(particles_state).flower.cube_draw_params);
    reset_draw_params(deref(particles_state).flower.shadow_cube_draw_params);
    reset_draw_params(deref(particles_state).flower.splat_draw_params);
    FlowerAllocator_perframe(flower_allocator);

    // tree_particle
    reset_draw_params(deref(particles_state).tree_particle.cube_draw_params);
    reset_draw_params(deref(particles_state).tree_particle.shadow_cube_draw_params);
    reset_draw_params(deref(particles_state).tree_particle.splat_draw_params);
    TreeParticleAllocator_perframe(tree_particle_allocator);

    // fire_particle
    reset_draw_params(deref(particles_state).fire_particle.cube_draw_params);
    reset_draw_params(deref(particles_state).fire_particle.shadow_cube_draw_params);
    reset_draw_params(deref(particles_state).fire_particle.splat_draw_params);
    FireParticleAllocator_perframe(fire_particle_allocator);
}
