#define DAXA_RAY_TRACING 1
#extension GL_EXT_ray_tracing : enable

#include <daxa/daxa.inl>

#include "trace_primary.inl"
#include <voxels/voxels.glsl>

struct hitPayload {
    PackedVoxel hit_voxel;
    vec3 pos_ws;
    vec3 vel_ws;
};

struct Ray {
    daxa_f32vec3 origin;
    daxa_f32vec3 direction;
};

DAXA_DECL_PUSH_CONSTANT(TestRtPush, p)

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_RAYGEN

layout(location = 0) rayPayloadEXT hitPayload prd;

void main() {
    const ivec2 index = ivec2(gl_LaunchIDEXT.xy);

    uint cull_mask = 0xff;

    // Camera setup
    daxa_f32mat4x4 inv_view = deref(p.uses.gpu_input).player.cam.view_to_world;
    daxa_f32mat4x4 inv_proj = deref(p.uses.gpu_input).player.cam.sample_to_view;
    inv_proj[1][1] *= -1;

    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = inUV * 2.0 - 1.0;

    vec4 origin = inv_view * vec4(0, 0, 0, 1);
    vec4 target = inv_proj * vec4(d.x, d.y, 1, 1);
    vec4 direction = inv_view * vec4(normalize(target.xyz), 0);

    uint rayFlags = gl_RayFlagsNoneEXT;
    float tMin = 0.0001;
    float tMax = 10000.0;
    uint cullMask = 0xFF;

    traceRayEXT(
        daxa_accelerationStructureEXT(p.tlas), // topLevelAccelerationStructure
        rayFlags,                              // rayFlags
        cullMask,                              // cullMask
        0,                                     // sbtRecordOffset
        0,                                     // sbtRecordStride
        0,                                     // missIndex
        origin.xyz,                            // ray origin
        tMin,                                  // ray min range
        direction.xyz,                         // ray direction
        tMax,                                  // ray max range
        0                                      // payload (location = 0)
    );

    Voxel voxel = unpack_voxel(prd.hit_voxel);
    if (voxel.material_type == 0) {
        return;
    }

    vec3 ws_nrm = voxel.normal;
    vec3 vs_nrm = (deref(p.uses.gpu_input).player.cam.world_to_view * vec4(ws_nrm, 0)).xyz;
    vec3 vs_velocity = vec3(0, 0, 0);

    vec4 vs_pos = (deref(p.uses.gpu_input).player.cam.world_to_view * vec4(prd.pos_ws, 1));
    vec4 prev_vs_pos = (deref(p.uses.gpu_input).player.cam.world_to_view * vec4(prd.pos_ws + prd.vel_ws, 1));
    vec4 ss_pos = (deref(p.uses.gpu_input).player.cam.view_to_sample * vs_pos);
    float depth = ss_pos.z / ss_pos.w;

    // TODO: remove when we get rid of the old compute RT
    float prev_depth = imageLoad(daxa_image2D(p.uses.depth_image_id), ivec2(gl_LaunchIDEXT.xy)).r;
    if (depth <= prev_depth) {
        return;
    }

    uvec4 output_value = uvec4(0);
    output_value.x = pack_voxel(voxel).data;
    output_value.y = nrm_to_u16(ws_nrm);
    output_value.z = floatBitsToUint(depth);

    imageStore(daxa_uimage2D(p.uses.g_buffer_image_id), ivec2(gl_LaunchIDEXT.xy), output_value);
    imageStore(daxa_image2D(p.uses.velocity_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(vs_velocity, 0));
    imageStore(daxa_image2D(p.uses.vs_normal_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(vs_nrm * 0.5 + 0.5, 0));
    imageStore(daxa_image2D(p.uses.depth_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(depth, 0, 0, 0));
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_INTERSECTION

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

#include <utilities/gpu/math.glsl>

daxa_BufferPtr(BlasGeom) blas_geoms;

bool getVoxel(ivec3 c) {
    c = c & (BLAS_BRICK_SIZE - 1);
    uint bit_index = c.x + c.y * BLAS_BRICK_SIZE + c.z * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE;
    uint u32_index = bit_index / 32;
    uint in_u32_index = bit_index & 0x1f;
    uint val = deref(advance(blas_geoms, gl_PrimitiveID)).bitmask[u32_index];
    return (val & (1 << in_u32_index)) != 0;
    // return length(vec3(c - 4)) < 4.0;
    // return (c & 3) == ivec3(0);
}

void main() {
    Ray ray;
    ray.origin = gl_ObjectRayOriginEXT;
    ray.direction = gl_ObjectRayDirectionEXT;

    mat4 world_to_blas = mat4(
        // Note that the ordering is transposed
        gl_ObjectToWorld3x4EXT[0][0], gl_ObjectToWorld3x4EXT[0][1], gl_ObjectToWorld3x4EXT[0][2], gl_ObjectToWorld3x4EXT[0][3],
        gl_ObjectToWorld3x4EXT[0][1], gl_ObjectToWorld3x4EXT[1][1], gl_ObjectToWorld3x4EXT[1][2], gl_ObjectToWorld3x4EXT[1][3],
        gl_ObjectToWorld3x4EXT[2][0], gl_ObjectToWorld3x4EXT[2][1], gl_ObjectToWorld3x4EXT[2][2], gl_ObjectToWorld3x4EXT[2][3],
        0, 0, 0, 1.0);

    ray.origin = (world_to_blas * vec4(ray.origin, 1)).xyz;
    ray.direction = (world_to_blas * vec4(ray.direction, 0)).xyz;
    float tHit = -1;
    // InstanceCustomIndexKHR
    blas_geoms = deref(advance(p.uses.geometry_pointers, gl_InstanceCustomIndexEXT));
    Aabb aabb = deref(advance(blas_geoms, gl_PrimitiveID)).aabb;
    tHit = hitAabb(aabb, ray);

    // Move ray to AABB surface (biased just barely inside)
    const float BIAS = uintBitsToFloat(0x3f800040); // uintBitsToFloat(0x3f800040) == 1.00000762939453125
    ray.origin += ray.direction * tHit * BIAS;

    if (tHit >= 0) {
        // DDA
        ivec3 bmin = ivec3(floor(aabb.minimum * VOXEL_SCL));
        ivec3 mapPos = ivec3(floor(ray.origin * VOXEL_SCL)) - bmin;
        vec3 deltaDist = abs(vec3(length(ray.direction)) / ray.direction);
        vec3 sideDist = (sign(ray.direction) * (vec3(mapPos + bmin) - ray.origin * VOXEL_SCL) + (sign(ray.direction) * 0.5) + 0.5) * deltaDist;
        ivec3 rayStep = ivec3(sign(ray.direction));
        bvec3 mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));

        for (int i = 0; i < int(3 * VOXEL_SCL); i++) {
            if (getVoxel(mapPos) == true) {
                if (i != 0) {
                    aabb.minimum += vec3(mapPos) * VOXEL_SIZE;
                    aabb.maximum = aabb.minimum + VOXEL_SIZE;
                    tHit += hitAabb(aabb, ray);
                }
                reportIntersectionEXT(tHit, 0);
                break;
            }
            mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
            sideDist += vec3(mask) * deltaDist;
            mapPos += ivec3(vec3(mask)) * rayStep;
            if (any(lessThan(mapPos, ivec3(0))) || any(greaterThanEqual(mapPos, ivec3(BLAS_BRICK_SIZE)))) {
                break;
            }
        }
    }
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_ANY_HIT

layout(location = 0) rayPayloadInEXT hitPayload prd;

// hardcoded dissolve
void main() {
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_CLOSEST_HIT

layout(location = 0) rayPayloadInEXT hitPayload prd;

void main() {
    const float BIAS = uintBitsToFloat(0x3f800040); // uintBitsToFloat(0x3f800040) == 1.00000762939453125
    vec3 world_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT * BIAS;

    mat4 blas_to_world = mat4(
        gl_ObjectToWorldEXT[0][0], gl_ObjectToWorldEXT[0][1], gl_ObjectToWorldEXT[0][2], 0,
        gl_ObjectToWorldEXT[0][1], gl_ObjectToWorldEXT[1][1], gl_ObjectToWorldEXT[1][2], 0,
        gl_ObjectToWorldEXT[2][0], gl_ObjectToWorldEXT[2][1], gl_ObjectToWorldEXT[2][2], 0,
        gl_ObjectToWorldEXT[3][0], gl_ObjectToWorldEXT[3][1], gl_ObjectToWorldEXT[3][2], 1.0);

    daxa_BufferPtr(BlasGeom) blas_geoms = deref(advance(p.uses.geometry_pointers, gl_InstanceCustomIndexEXT));
    Aabb aabb = deref(advance(blas_geoms, gl_PrimitiveID)).aabb;

    vec3 center = (aabb.minimum + aabb.maximum) * 0.5;
    center = (blas_to_world * vec4(center, 1)).xyz;

    vec3 world_nrm = normalize(floor((world_pos - center) * VOXEL_SCL) * VOXEL_SIZE);

    Voxel voxel;
    voxel.material_type = 1;
    voxel.roughness = 0.9;
    voxel.color = vec3(1);
    voxel.normal = world_nrm;
    prd.hit_voxel = pack_voxel(voxel);
    prd.pos_ws = world_pos;
    prd.vel_ws = vec3(0, 0, 0);
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_MISS

layout(location = 0) rayPayloadInEXT hitPayload prd;

void main() {
    prd.hit_voxel.data = 0;
}

#endif // DAXA_SHADER_STAGE
