#pragma once

#define VOXEL_ACCEL_UNIFORMITY 1
#define ENABLE_DEPTH_PREPASS 1
#define ENABLE_TREE_GENERATION 1

#define CHUNK_FLAGS_ACCEL_GENERATED (1 << 0)

#define BRUSH_FLAGS_WORLD_BRUSH (1 << 1)
#define BRUSH_FLAGS_USER_BRUSH_A (1 << 2)
#define BRUSH_FLAGS_USER_BRUSH_B (1 << 3)
#define BRUSH_FLAGS_PARTICLE_BRUSH (1 << 4)
#define BRUSH_FLAGS_BRUSH_MASK (BRUSH_FLAGS_WORLD_BRUSH | BRUSH_FLAGS_USER_BRUSH_A | BRUSH_FLAGS_USER_BRUSH_B | BRUSH_FLAGS_PARTICLE_BRUSH)

#define VOXEL_SCL 8
#define PER_VOXEL_NORMALS 1