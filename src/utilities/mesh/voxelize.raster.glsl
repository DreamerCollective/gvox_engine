#include <utilities/mesh/mesh_model.inl>

DAXA_DECL_PUSH_CONSTANT(MeshRasterPush, daxa_push_constant)
#define VERTS(i) deref(daxa_push_constant.vertex_buffer[i])
#define NORMALS(i) deref(daxa_push_constant.normal_buffer[i])
#define INPUT deref(daxa_push_constant.gpu_input)
#define VOXELS(i) deref(daxa_push_constant.voxel_buffer[i])

#if RASTER_VERT

layout(location = 0) out vec2 v_tex;
layout(location = 1) out vec3 v_nrm;
layout(location = 2) out uint v_rotation;

void main() {
    MeshVertex vert = VERTS(gl_VertexIndex);

    v_tex = vert.tex;
    uint tri_index = gl_VertexIndex / 3;
    // Vertex v0 = VERTS(tri_index * 3 + 0);
    // Vertex v1 = VERTS(tri_index * 3 + 1);
    // Vertex v2 = VERTS(tri_index * 3 + 2);
    // v_nrm = normalize(cross(v1.pos - v0.pos, v2.pos - v0.pos));
    v_nrm = -normalize(NORMALS(tri_index));

    v_rotation = vert.rotation;

    gl_Position = vec4(vert.pos, 1);
    gl_Position.xyz = gl_Position.xyz * vec3(1, 1, 0.5) + vec3(0, 0, 0.5);
}

#elif RASTER_FRAG

layout(location = 0) in vec2 v_tex;
layout(location = 1) in vec3 v_nrm;
layout(location = 2) flat in uint v_rotation;

#include <voxels/pack_unpack.glsl>

void main() {
    vec3 p = gl_FragCoord.xyz;
    // vec3 vs_nrm = v_nrm;
    // switch (v_rotation) {
    // case 0: vs_nrm = vs_nrm.zyx; break;
    // case 1: vs_nrm = vs_nrm.xzy; break;
    // case 2: break;
    // }

    p.z = p.z * INPUT.size.z; // + sign(vs_nrm.z) * 0.1;
    switch (v_rotation) {
    case 0: p = p.zyx; break;
    case 1: p = p.xzy; break;
    case 2: break;
    }
    uvec3 vp = clamp(uvec3(floor(p)), uvec3(0), uvec3(INPUT.size - 1));
    uint o_index = vp.x + vp.y * INPUT.size.x + vp.z * INPUT.size.x * INPUT.size.y;

    vec4 tex0_col = texture(daxa_sampler2D(daxa_push_constant.texture_id, daxa_push_constant.texture_sampler), v_tex);
    uint r = uint(pow(tex0_col.r, 1.0 / 2.2) * 255);
    uint g = uint(pow(tex0_col.g, 1.0 / 2.2) * 255);
    uint b = uint(pow(tex0_col.b, 1.0 / 2.2) * 255);
    // uint r = uint(clamp(v_nrm.r * 0.5 + 0.5, 0.0, 1.0) * 255);
    // uint g = uint(clamp(v_nrm.g * 0.5 + 0.5, 0.0, 1.0) * 255);
    // uint b = uint(clamp(v_nrm.b * 0.5 + 0.5, 0.0, 1.0) * 255);

    Voxel voxel;
    voxel.material_type = 1;
    voxel.normal = v_nrm;
    voxel.roughness = 0.9;
    voxel.color = tex0_col.rgb;
    PackedVoxel packed_voxel = pack_voxel(voxel);

    // uint i = 255;
    // uint u32_voxel = 0;
    // u32_voxel = u32_voxel | (r << 0x00);
    // u32_voxel = u32_voxel | (g << 0x08);
    // u32_voxel = u32_voxel | (b << 0x10);
    // u32_voxel = u32_voxel | (i << 0x18);
    atomicExchange(VOXELS(o_index), packed_voxel.data);
}

#endif
