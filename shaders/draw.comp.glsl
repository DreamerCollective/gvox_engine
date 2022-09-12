#version 450

#include <shared.inl>

DAXA_USE_PUSH_CONSTANT(DrawCompPush)

#define MAX_DIST 10000.0

struct Sphere {
    f32vec3 o;
    f32 r;
};

struct Box {
    f32vec3 bound_min, bound_max;
};

struct Ray {
    f32vec3 o;
    f32vec3 nrm;
    f32vec3 inv_nrm;
};

struct IntersectionRecord {
    b32 hit;
    f32 dist;
    f32vec3 nrm;
};

struct TraceRecord {
    IntersectionRecord intersection_record;
    f32vec3 color;
};

void default_init(out IntersectionRecord result) {
    result.hit = false;
    result.dist = MAX_DIST;
    result.nrm = f32vec3(0, 0, 0);
}

IntersectionRecord intersect_sphere(Ray ray, Sphere s) {
    IntersectionRecord result;
    default_init(result);

    f32vec3 so_r = ray.o - s.o;
    f32 a = dot(ray.nrm, ray.nrm);
    f32 b = 2.0f * dot(ray.nrm, so_r);
    f32 c = dot(so_r, so_r) - (s.r * s.r);
    f32 f = b * b - 4.0f * a * c;
    if (f < 0.0f)
        return result;
    result.dist = (-b - sqrt(f)) / (2.0f * a);
    result.hit = result.dist > 0.0f;
    result.nrm = normalize(ray.o + ray.nrm * result.dist - s.o);
    return result;
}

IntersectionRecord intersect_box(Ray ray, Box b) {
    IntersectionRecord result;
    default_init(result);

    f32 tx1 = (b.bound_min.x - ray.o.x) * ray.inv_nrm.x;
    f32 tx2 = (b.bound_max.x - ray.o.x) * ray.inv_nrm.x;
    f32 tmin = min(tx1, tx2);
    f32 tmax = max(tx1, tx2);
    f32 ty1 = (b.bound_min.y - ray.o.y) * ray.inv_nrm.y;
    f32 ty2 = (b.bound_max.y - ray.o.y) * ray.inv_nrm.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));
    f32 tz1 = (b.bound_min.z - ray.o.z) * ray.inv_nrm.z;
    f32 tz2 = (b.bound_max.z - ray.o.z) * ray.inv_nrm.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    result.hit = (tmax >= tmin && tmin > 0);
    result.dist = tmin;

    b32 is_x = tmin == tx1 || tmin == tx2;
    b32 is_y = tmin == ty1 || tmin == ty2;
    b32 is_z = tmin == tz1 || tmin == tz2;

    if (is_z) {
        if (ray.nrm.z < 0) {
            result.nrm = f32vec3(0, 0, 1);
        } else {
            result.nrm = f32vec3(0, 0, -1);
        }
    } else if (is_y) {
        if (ray.nrm.y < 0) {
            result.nrm = f32vec3(0, 1, 0);
        } else {
            result.nrm = f32vec3(0, -1, 0);
        }
    } else {
        if (ray.nrm.x < 0) {
            result.nrm = f32vec3(1, 0, 0);
        } else {
            result.nrm = f32vec3(-1, 0, 0);
        }
    }

    return result;
}

Ray create_view_ray(f32vec2 uv) {
    Ray result;

    result.o = push_constant.gpu_globals.player.cam.pos;
    result.nrm = normalize(f32vec3(uv.x * push_constant.gpu_globals.player.cam.tan_half_fov, 1, -uv.y * push_constant.gpu_globals.player.cam.tan_half_fov));
    result.nrm = push_constant.gpu_globals.player.cam.rot_mat * result.nrm;

    return result;
}

TraceRecord trace_scene(in Ray ray) {
    Sphere s0;
    s0.o = f32vec3(sin(push_constant.gpu_input.time) * 0, 1, 0);
    s0.r = 0.5;

    Box b0;
    b0.bound_min = f32vec3(-2.0, -2.0, -0.6);
    b0.bound_max = f32vec3(+2.0, +2.0, -0.5);

    TraceRecord trace;
    default_init(trace.intersection_record);
    trace.color = f32vec3(0.3, 0.4, 0.9);

    IntersectionRecord s0_hit = intersect_sphere(ray, s0);
    IntersectionRecord b0_hit = intersect_box(ray, b0);

    if (s0_hit.hit && s0_hit.dist < trace.intersection_record.dist) {
        trace.intersection_record = s0_hit;
        trace.color = f32vec3(1.0, 0.5, 0.5);
    }
    if (b0_hit.hit && b0_hit.dist < trace.intersection_record.dist) {
        trace.intersection_record = b0_hit;
        trace.color = f32vec3(0.5, 0.5, 1.0);
    }

    return trace;
}

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    u32vec3 pixel_i = gl_GlobalInvocationID.xyz;
    if (pixel_i.x >= push_constant.gpu_input.frame_dim.x ||
        pixel_i.y >= push_constant.gpu_input.frame_dim.y)
        return;

    f32vec2 pixel_p = pixel_i.xy;
    f32vec2 frame_dim = push_constant.gpu_input.frame_dim;
    f32vec2 inv_frame_dim = f32vec2(1.0, 1.0) / frame_dim;
    f32 aspect = frame_dim.x * inv_frame_dim.y;

    f32vec2 uv = pixel_p * inv_frame_dim;
    uv = (uv - 0.5) * f32vec2(aspect, 1.0) * 2.0;

    f32vec3 col = f32vec3(0, 0, 0);

    Ray view_ray = create_view_ray(uv);
    view_ray.inv_nrm = 1.0 / view_ray.nrm;

    TraceRecord view_trace_record = trace_scene(view_ray);
    col = view_trace_record.color;

    if (view_trace_record.intersection_record.hit) {
        f32vec3 hit_pos = view_ray.o + view_ray.nrm * view_trace_record.intersection_record.dist;
        Ray sun_ray;
        sun_ray.o = hit_pos + view_trace_record.intersection_record.nrm * 0.001;
        sun_ray.nrm = normalize(f32vec3(1, -2, 3));
        sun_ray.inv_nrm = 1.0 / sun_ray.nrm;
        f32 shade = max(dot(sun_ray.nrm, view_trace_record.intersection_record.nrm), 0.0);
        TraceRecord sun_trace_record = trace_scene(sun_ray);
        shade *= f32(!sun_trace_record.intersection_record.hit);
        col *= shade;
    }

    imageStore(
        daxa_GetRWImage(image2D, rgba32f, push_constant.image_id),
        i32vec2(pixel_i.xy),
        f32vec4(col, 1));
}
