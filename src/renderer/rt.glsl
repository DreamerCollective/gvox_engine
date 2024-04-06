#pragma once

#include <voxels/voxels.glsl>
#include <utilities/gpu/math.glsl>
#include <application/input.inl>
#include <voxels/impl/voxels.inl>

#define PAYLOAD_LOC 0

struct HitAttribute {
    uint data;
};

struct RayPayload {
    uint data0;
    uint data1;
};

struct Ray {
    daxa_f32vec3 origin;
    daxa_f32vec3 direction;
};

HitAttribute pack_hit_attribute(ivec3 voxel_i) {
    return HitAttribute(voxel_i.x + voxel_i.y * BLAS_BRICK_SIZE + voxel_i.z * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE);
}

RayPayload pack_ray_payload(uint blas_id, uint brick_id, HitAttribute hit_attrib) {
    return RayPayload(blas_id, (brick_id * (BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * 2)) | hit_attrib.data);
}

RayPayload miss_ray_payload() {
    return pack_ray_payload(0, 0, HitAttribute(BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE));
}

// Ray-AABB intersection
float hitAabb(const Aabb aabb, const Ray r) {
    if (all(greaterThanEqual(r.origin, aabb.minimum)) && all(lessThanEqual(r.origin, aabb.maximum))) {
        return 0.0;
    }
    vec3 invDir = 1.0 / r.direction;
    vec3 tbot = invDir * (aabb.minimum - r.origin);
    vec3 ttop = invDir * (aabb.maximum - r.origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    float t0 = max(tmin.x, max(tmin.y, tmin.z));
    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    return t1 > max(t0, 0.0) ? t0 : -1.0;
}

float hitAabb_midpoint(const Aabb aabb, const Ray r) {
    vec3 invDir = 1.0 / r.direction;
    vec3 tbot = invDir * (aabb.minimum - r.origin);
    vec3 ttop = invDir * (aabb.maximum - r.origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    float t0 = max(tmin.x, max(tmin.y, tmin.z));
    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    return (t0 + t1) * 0.5;
}

PackedVoxel unpack_ray_payload(
    daxa_BufferPtr(daxa_BufferPtr(BlasGeom)) geometry_pointers,
    daxa_BufferPtr(daxa_BufferPtr(VoxelBrickAttribs)) attribute_pointers,
    daxa_BufferPtr(daxa_f32vec3) blas_transforms,
    RayPayload payload, Ray ray, out vec3 hit_pos) {
    uint blas_id = payload.data0;
    uint brick_id = payload.data1 / (BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * 2);
    uint voxel_index = payload.data1 & (BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * 2 - 1);
    daxa_BufferPtr(VoxelBrickAttribs) brick_attribs = deref(advance(attribute_pointers, blas_id));
    daxa_BufferPtr(BlasGeom) blas_geoms = deref(advance(geometry_pointers, blas_id));
    {
        // mat3x4 m = deref(advance(blas_transforms, blas_id));
        // mat4 world_to_blas = mat4(m[0], m[1], m[2], vec4(0, 0, 0, 1));
        // mat4 blas_to_world = transpose(world_to_blas);

        vec3 v = deref(advance(blas_transforms, blas_id));
        Aabb aabb = deref(advance(blas_geoms, brick_id)).aabb;
        aabb.minimum += v;
        aabb.maximum += v;
        ivec3 mapPos = ivec3(voxel_index % BLAS_BRICK_SIZE, (voxel_index / BLAS_BRICK_SIZE) % BLAS_BRICK_SIZE, voxel_index / BLAS_BRICK_SIZE / BLAS_BRICK_SIZE);
        aabb.minimum += vec3(mapPos) * VOXEL_SIZE;
        aabb.maximum = aabb.minimum + VOXEL_SIZE;
#if PER_VOXEL_NORMALS
        hit_pos = ray.origin + ray.direction * hitAabb_midpoint(aabb, ray);
#else
        hit_pos = ray.origin + ray.direction * hitAabb(aabb, ray);
#endif
        // hit_pos = (blas_to_world * vec4(hit_pos, 1)).xyz;
    }
    return deref(advance(brick_attribs, brick_id)).packed_voxels[voxel_index];
}

bool getVoxel(daxa_BufferPtr(BlasGeom) blas_geoms, uint brick_id, ivec3 c) {
    uint bit_index = c.x + c.y * BLAS_BRICK_SIZE + c.z * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE;
    uint u32_index = bit_index / 32;
    uint in_u32_index = bit_index & 0x1f;
    uint val = deref(advance(blas_geoms, brick_id)).bitmask[u32_index];
    return (val & (1 << in_u32_index)) != 0;
}

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_INTERSECTION
#define intersect_voxels(geometry_pointers, hit_attrib)                                                                                             \
    {                                                                                                                                               \
        Ray ray;                                                                                                                                    \
        ray.origin = gl_ObjectRayOriginEXT;                                                                                                         \
        ray.direction = gl_ObjectRayDirectionEXT;                                                                                                   \
        mat4 world_to_blas = mat4(                                                                                                                  \
            gl_ObjectToWorld3x4EXT[0][0], gl_ObjectToWorld3x4EXT[0][1], gl_ObjectToWorld3x4EXT[0][2], gl_ObjectToWorld3x4EXT[0][3],                 \
            gl_ObjectToWorld3x4EXT[1][0], gl_ObjectToWorld3x4EXT[1][1], gl_ObjectToWorld3x4EXT[1][2], gl_ObjectToWorld3x4EXT[1][3],                 \
            gl_ObjectToWorld3x4EXT[2][0], gl_ObjectToWorld3x4EXT[2][1], gl_ObjectToWorld3x4EXT[2][2], gl_ObjectToWorld3x4EXT[2][3],                 \
            0, 0, 0, 1.0);                                                                                                                          \
        ray.origin = (world_to_blas * vec4(ray.origin, 1)).xyz;                                                                                     \
        ray.direction = (world_to_blas * vec4(ray.direction, 0)).xyz;                                                                               \
        float tHit = -1;                                                                                                                            \
        daxa_BufferPtr(BlasGeom) blas_geoms = deref(advance(geometry_pointers, gl_InstanceCustomIndexEXT));                                         \
        Aabb aabb = deref(advance(blas_geoms, gl_PrimitiveID)).aabb;                                                                                \
        tHit = hitAabb(aabb, ray);                                                                                                                  \
        const float BIAS = uintBitsToFloat(0x3f800040);                                                                                             \
        ray.origin += ray.direction * tHit * BIAS;                                                                                                  \
        if (tHit >= 0) {                                                                                                                            \
            ivec3 bmin = ivec3(floor(aabb.minimum * VOXEL_SCL));                                                                                    \
            ivec3 mapPos = clamp(ivec3(floor(ray.origin * VOXEL_SCL)) - bmin, ivec3(0), ivec3(BLAS_BRICK_SIZE - 1));                                \
            vec3 deltaDist = abs(vec3(length(ray.direction)) / ray.direction);                                                                      \
            vec3 sideDist = (sign(ray.direction) * (vec3(mapPos + bmin) - ray.origin * VOXEL_SCL) + (sign(ray.direction) * 0.5) + 0.5) * deltaDist; \
            ivec3 rayStep = ivec3(sign(ray.direction));                                                                                             \
            bvec3 mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));                                                              \
            for (int i = 0; i < int(3 * VOXEL_SCL); i++) {                                                                                          \
                if (getVoxel(blas_geoms, gl_PrimitiveID, mapPos) == true) {                                                                         \
                    aabb.minimum += vec3(mapPos) * VOXEL_SIZE;                                                                                      \
                    aabb.maximum = aabb.minimum + VOXEL_SIZE;                                                                                       \
                    tHit += hitAabb_midpoint(aabb, ray);                                                                                            \
                    hit_attrib = pack_hit_attribute(mapPos);                                                                                        \
                    reportIntersectionEXT(tHit, 0);                                                                                                 \
                    break;                                                                                                                          \
                }                                                                                                                                   \
                mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));                                                                \
                sideDist += vec3(mask) * deltaDist;                                                                                                 \
                mapPos += ivec3(vec3(mask)) * rayStep;                                                                                              \
                if (any(lessThan(mapPos, ivec3(0))) || any(greaterThanEqual(mapPos, ivec3(BLAS_BRICK_SIZE)))) {                                     \
                    break;                                                                                                                          \
                }                                                                                                                                   \
            }                                                                                                                                       \
        }                                                                                                                                           \
    }
#endif
