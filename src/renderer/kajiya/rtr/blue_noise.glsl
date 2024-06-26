#pragma once

// #include <renderer/kajiya/inc/quasi_random.glsl>
// #include <utilities/gpu/bindless_textures.glsl>

// The source texture is RGBA8, and the output here is quantized to [0.5/256 .. 255.5/256]
// vec4 blue_noise_for_pixel(uvec2 px, uint n) {
//     const uvec2 tex_dims = uvec2(256, 256);
//     const uvec2 offset = r2_sequence(n) * tex_dims;
//     return bindless_textures[BINDLESS_LUT_BLUE_NOISE_256_LDR_RGBA_0][(px + offset) % tex_dims] * 255.0 / 256.0 + 0.5 / 256.0;
// }

// ----
// https://crates.io/crates/blue-noise-sampler

float blue_noise_sampler(
    int pixel_i,
    int pixel_j,
    int sampleIndex,
    int sampleDimension,
    daxa_BufferPtr(int) ranking_tile_buf,
    daxa_BufferPtr(int) scambling_tile_buf,
    daxa_BufferPtr(int) sobol_buf) {
    // wrap arguments
    pixel_i = pixel_i & 127;
    pixel_j = pixel_j & 127;
    sampleIndex = sampleIndex & 255;
    sampleDimension = sampleDimension & 255;

    // xor index based on optimized ranking
    // jb: 1spp blue noise has all 0 in ranking_tile_buf so we can skip the load
    int rankedSampleIndex = sampleIndex ^ deref(advance(ranking_tile_buf, sampleDimension + (pixel_i + pixel_j * 128) * 8));

    // fetch value in sequence
    int value = deref(advance(sobol_buf, sampleDimension + rankedSampleIndex * 256));

    // If the dimension is optimized, xor sequence value based on optimized scrambling
    value = value ^ deref(advance(scambling_tile_buf, (sampleDimension % 8) + (pixel_i + pixel_j * 128) * 8));

    // convert to float and return
    float v = (0.5f + value) / 256.0f;
    return v;
}

// ----
