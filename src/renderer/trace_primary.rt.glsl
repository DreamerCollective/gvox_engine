#define DAXA_RAY_TRACING 1
#extension GL_EXT_ray_tracing : enable

#define PAYLOAD_LOC 0

#include <daxa/daxa.inl>

#include "trace_primary.inl"
#include <voxels/voxels.glsl>

struct HitAttribute {
    uint data;
};

struct RayPayload {
    uint data0;
    uint data1;
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

struct Ray {
    daxa_f32vec3 origin;
    daxa_f32vec3 direction;
};

DAXA_DECL_PUSH_CONSTANT(TestRtPush, push)

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

PackedVoxel unpack_ray_payload(RayPayload payload, Ray ray, out vec3 hit_pos) {
    uint blas_id = payload.data0;
    uint brick_id = payload.data1 / (BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * 2);
    uint voxel_index = payload.data1 & (BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE * 2 - 1);
    daxa_BufferPtr(VoxelBrickAttribs) brick_attribs = deref(advance(push.uses.attribute_pointers, blas_id));
    daxa_BufferPtr(BlasGeom) blas_geoms = deref(advance(push.uses.geometry_pointers, blas_id));
    {
        // mat3x4 m = deref(advance(push.uses.blas_transforms, blas_id));
        // mat4 world_to_blas = mat4(m[0], m[1], m[2], vec4(0, 0, 0, 1));
        // mat4 blas_to_world = transpose(world_to_blas);

        vec3 v = deref(advance(push.uses.blas_transforms, blas_id));
        Aabb aabb = deref(advance(blas_geoms, brick_id)).aabb;
        aabb.minimum += v;
        aabb.maximum += v;
        ivec3 mapPos = ivec3(voxel_index % BLAS_BRICK_SIZE, (voxel_index / BLAS_BRICK_SIZE) % BLAS_BRICK_SIZE, voxel_index / BLAS_BRICK_SIZE / BLAS_BRICK_SIZE);
        aabb.minimum += vec3(mapPos) * VOXEL_SIZE;
        aabb.maximum = aabb.minimum + VOXEL_SIZE;
        hit_pos = ray.origin + ray.direction * hitAabb_midpoint(aabb, ray);
        // hit_pos = (blas_to_world * vec4(hit_pos, 1)).xyz;
    }
    return deref(advance(brick_attribs, brick_id)).packed_voxels[voxel_index];
}

#if DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_RAYGEN

layout(location = PAYLOAD_LOC) rayPayloadEXT RayPayload prd;

#include <renderer/kajiya/inc/camera.glsl>

void main() {
    const ivec2 index = ivec2(gl_LaunchIDEXT.xy);

    vec4 output_tex_size = vec4(deref(push.uses.gpu_input).frame_dim, 0, 0);
    output_tex_size.zw = vec2(1.0, 1.0) / output_tex_size.xy;
    vec2 uv = get_uv(gl_LaunchIDEXT.xy, output_tex_size);

    ViewRayContext vrc = vrc_from_uv(push.uses.gpu_input, uv);
    vec3 ray_d = ray_dir_ws(vrc);
    vec3 ray_o = ray_origin_ws(vrc);

    uint rayFlags = gl_RayFlagsNoneEXT;
    float tMin = 0.0001;
    float tMax = 10000.0;
    uint cull_mask = 0xFF;
    uint sbtRecordOffset = 0;
    uint sbtRecordStride = 0;
    uint missIndex = 0;

    traceRayEXT(
        daxa_accelerationStructureEXT(push.tlas),
        rayFlags, cull_mask, sbtRecordOffset, sbtRecordStride, missIndex,
        ray_o, tMin, ray_d, tMax, PAYLOAD_LOC);

    if (prd.data1 == miss_ray_payload().data1) {
        imageStore(daxa_image2D(push.uses.depth_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(0));
        imageStore(daxa_uimage2D(push.uses.g_buffer_image_id), ivec2(gl_LaunchIDEXT.xy), uvec4(0));
        return;
    }

    vec3 world_pos = vec3(0);

    uvec3 chunk_n = uvec3(CHUNKS_PER_AXIS);
    PackedVoxel voxel_data = unpack_ray_payload(prd, Ray(ray_o, ray_d), world_pos);
    Voxel voxel = unpack_voxel(voxel_data);

    vec3 ws_nrm = voxel.normal;
    vec3 vs_nrm = (deref(push.uses.gpu_input).player.cam.world_to_view * vec4(ws_nrm, 0)).xyz;
    vec3 vs_velocity = vec3(0, 0, 0);

    vec3 vel_ws = vec3(deref(push.uses.gpu_input).player.player_unit_offset - deref(push.uses.gpu_input).player.prev_unit_offset);

    vec4 vs_pos = (deref(push.uses.gpu_input).player.cam.world_to_view * vec4(world_pos, 1));
    vec4 prev_vs_pos = (deref(push.uses.gpu_input).player.cam.world_to_view * vec4(world_pos + vel_ws, 1));
    vec4 ss_pos = (deref(push.uses.gpu_input).player.cam.view_to_sample * vs_pos);
    float depth = ss_pos.z / ss_pos.w;

    vs_velocity = (prev_vs_pos.xyz / prev_vs_pos.w) - (vs_pos.xyz / vs_pos.w);

    uvec4 output_value = uvec4(0);
    output_value.x = pack_voxel(voxel).data;
    output_value.y = nrm_to_u16(ws_nrm);
    output_value.z = floatBitsToUint(depth);

    vs_nrm *= -sign(dot(ray_dir_vs(vrc), vs_nrm));

    imageStore(daxa_uimage2D(push.uses.g_buffer_image_id), ivec2(gl_LaunchIDEXT.xy), output_value);
    imageStore(daxa_image2D(push.uses.velocity_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(vs_velocity, 0));
    imageStore(daxa_image2D(push.uses.vs_normal_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(vs_nrm * 0.5 + 0.5, 0));
    imageStore(daxa_image2D(push.uses.depth_image_id), ivec2(gl_LaunchIDEXT.xy), vec4(depth, 0, 0, 0));
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_INTERSECTION

#include <utilities/gpu/math.glsl>

daxa_BufferPtr(BlasGeom) blas_geoms;

bool getVoxel(ivec3 c) {
    uint bit_index = c.x + c.y * BLAS_BRICK_SIZE + c.z * BLAS_BRICK_SIZE * BLAS_BRICK_SIZE;
    uint u32_index = bit_index / 32;
    uint in_u32_index = bit_index & 0x1f;
    uint val = deref(advance(blas_geoms, gl_PrimitiveID)).bitmask[u32_index];
    return (val & (1 << in_u32_index)) != 0;
}

hitAttributeEXT HitAttribute hit_attrib;

void main() {
    Ray ray;
    ray.origin = gl_ObjectRayOriginEXT;
    ray.direction = gl_ObjectRayDirectionEXT;

    mat4 world_to_blas = mat4(
        // Note that the ordering is transposed
        gl_ObjectToWorld3x4EXT[0][0], gl_ObjectToWorld3x4EXT[0][1], gl_ObjectToWorld3x4EXT[0][2], gl_ObjectToWorld3x4EXT[0][3],
        gl_ObjectToWorld3x4EXT[1][0], gl_ObjectToWorld3x4EXT[1][1], gl_ObjectToWorld3x4EXT[1][2], gl_ObjectToWorld3x4EXT[1][3],
        gl_ObjectToWorld3x4EXT[2][0], gl_ObjectToWorld3x4EXT[2][1], gl_ObjectToWorld3x4EXT[2][2], gl_ObjectToWorld3x4EXT[2][3],
        0, 0, 0, 1.0);

    ray.origin = (world_to_blas * vec4(ray.origin, 1)).xyz;
    ray.direction = (world_to_blas * vec4(ray.direction, 0)).xyz;
    float tHit = -1;
    blas_geoms = deref(advance(push.uses.geometry_pointers, gl_InstanceCustomIndexEXT));
    Aabb aabb = deref(advance(blas_geoms, gl_PrimitiveID)).aabb;
    tHit = hitAabb(aabb, ray);

    // Move ray to AABB surface (biased just barely inside)
    const float BIAS = uintBitsToFloat(0x3f800040); // uintBitsToFloat(0x3f800040) == 1.00000762939453125
    ray.origin += ray.direction * tHit * BIAS;

    if (tHit >= 0) {
        // DDA
        ivec3 bmin = ivec3(floor(aabb.minimum * VOXEL_SCL));
        ivec3 mapPos = clamp(ivec3(floor(ray.origin * VOXEL_SCL)) - bmin, ivec3(0), ivec3(BLAS_BRICK_SIZE - 1));
        vec3 deltaDist = abs(vec3(length(ray.direction)) / ray.direction);
        vec3 sideDist = (sign(ray.direction) * (vec3(mapPos + bmin) - ray.origin * VOXEL_SCL) + (sign(ray.direction) * 0.5) + 0.5) * deltaDist;
        ivec3 rayStep = ivec3(sign(ray.direction));
        bvec3 mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));

        for (int i = 0; i < int(3 * VOXEL_SCL); i++) {
            if (getVoxel(mapPos) == true) {
                aabb.minimum += vec3(mapPos) * VOXEL_SIZE;
                aabb.maximum = aabb.minimum + VOXEL_SIZE;
                tHit += hitAabb_midpoint(aabb, ray);
                hit_attrib = pack_hit_attribute(mapPos);
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

layout(location = PAYLOAD_LOC) rayPayloadInEXT RayPayload prd;

void main() {
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_CLOSEST_HIT

layout(location = PAYLOAD_LOC) rayPayloadInEXT RayPayload prd;
hitAttributeEXT HitAttribute hit_attrib;

void main() {
    prd = pack_ray_payload(gl_InstanceCustomIndexEXT, gl_PrimitiveID, hit_attrib);
}

#elif DAXA_SHADER_STAGE == DAXA_SHADER_STAGE_MISS

layout(location = PAYLOAD_LOC) rayPayloadInEXT RayPayload prd;

void main() {
    prd = miss_ray_payload();
}

#endif // DAXA_SHADER_STAGE
