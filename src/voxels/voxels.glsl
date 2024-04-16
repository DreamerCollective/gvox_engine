#pragma once

#include <voxels/voxels.inl>
#include <voxels/brushes.inl>

struct VoxelTraceResult {
    float dist;
    vec3 nrm;
    vec3 vel;
    uint step_n;
    PackedVoxel voxel_data;
};

#include <voxels/impl/voxels.glsl>
