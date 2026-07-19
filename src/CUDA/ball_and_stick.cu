// ball_and_stick.cu
// ---------------------------------------------------------------------------
// Renders a molecule as shaded SPHERES (atoms) and TUBES (bonds).
//
// The stock renderers in 3d_points_lines.cu run one thread per primitive and
// stamp flat-colored discs and 1px lines, which read as stickers and hairlines.
// This kernel inverts the parallelism: one thread per PIXEL, walking every
// primitive and keeping the nearest one that covers it. That buys three things:
//
//   1. Shading. Knowing where inside a disc (or across a tube) we are gives us
//      the surface normal, so we can light it and it reads as a solid object.
//   2. Correct occlusion. The per-primitive kernels blend in whatever order the
//      threads finish, so overlaps are a coin flip. Here each pixel resolves its
//      own depth test, which is deterministic and right.
//   3. Spheres and tubes resolve against EACH OTHER in the same pass, so a bond
//      passing behind an atom is properly hidden.
//
// Molecules have a handful of atoms, so looping over all primitives per pixel is
// cheap - the same trade-off scattering.cu makes. May need to revisit if we ever
// move to proteins. But this can be tested first.
//
// A note on sizing: radii here are WORLD-space, in the same units as the atom
// coordinates (Angstroms) - unlike the flat point renderer, whose sizes are in
// screen pixels. That is what makes pulling the camera back behave like a real
// zoom: the atoms and bonds shrink along with the molecule, instead of the cage
// receding while the balls stay the same size on screen.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <math.h>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/ThreeDimensionStructs.h"
#include "color.cuh"
#include "common_graphics.cuh"

// Direction from a surface toward the key light, in camera space (x right, y up,
// z toward the viewer).
__device__ __constant__ float LIGHT[3] = {-0.35f, 0.50f, 0.79f};

// Light a unit normal and pack the result. Spheres and tubes share this: a tube's
// normal has exactly the same form as a sphere's, just measured from the bond
// axis instead of from a center point.
__device__ __forceinline__ int shade_surface(float nx, float ny, float nz, int col) {
    float base_r = ((col >> 16) & 0xff) / 255.0f;
    float base_g = ((col >>  8) & 0xff) / 255.0f;
    float base_b = ( col        & 0xff) / 255.0f;

    // Lambert diffuse.
    float diffuse = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2];
    if (diffuse < 0.0f) diffuse = 0.0f;

    // Blinn-Phong specular. The viewer looks down +z, so the half-vector between
    // the light and the eye is constant and can fold it in directly.
    const float HX = -0.185f, HY = 0.264f, HZ = 0.946f;
    float spec_dot = nx * HX + ny * HY + nz * HZ;
    if (spec_dot < 0.0f) spec_dot = 0.0f;
    float specular = powf(spec_dot, 40.0f) * 0.55f;

    // A little bit of of rim light at the silhouette. Against a black background this really helps
    float rim = powf(1.0f - nz, 3.0f) * 0.30f;

    float shade = 0.20f + 0.85f * diffuse + rim;   // 0.20 is ambient

    float out_r = Cuda::clamp(base_r * shade + specular, 0.0f, 1.0f);
    float out_g = Cuda::clamp(base_g * shade + specular, 0.0f, 1.0f);
    float out_b = Cuda::clamp(base_b * shade + specular, 0.0f, 1.0f);

    return 0xff000000
         | ((int)(out_r * 255.0f) << 16)
         | ((int)(out_g * 255.0f) <<  8)
         | ((int)(out_b * 255.0f));
}

__global__ void render_ball_and_stick_kernel(
    unsigned int* pixels, const Cuda::ivec2 wh,
    float geom_mean_size, float opacity, float radius_multiplier,
    const Cuda::Point* spheres, int num_spheres,
    const Cuda::Line*  tubes,   int num_tubes, float tube_size,
    Cuda::quat camera_direction, Cuda::vec3 camera_pos, float fov)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= wh.x || py >= wh.y) return;

    const float k = geom_mean_size * fov;   // pixels per unit at unit depth

    float best_depth   = 1e30f;
    int   best_color   = 0;
    float best_opacity = 0.0f;
    float nx = 0.0f, ny = 0.0f, nz = 0.0f;
    bool  hit = false;

    // ----- Atoms: ray against a screen-space disc, normal lifted onto a sphere -----
    for (int i = 0; i < num_spheres; i++) {
        Cuda::Point p = spheres[i];
        if (p.opacity <= 0.0f) continue;

        bool behind_camera = false;
        Cuda::vec3 proj;
        Cuda::d_coordinate_to_pixel(
            p.center, behind_camera,
            camera_direction, camera_pos, fov,
            geom_mean_size, wh, proj);
        if (behind_camera) continue;

        // World radius, projected to pixels at this atom's depth so it shrinks
        // with distance, both across the molecule and as the camera pulls back.
        float world_r = p.size * radius_multiplier;
        if (world_r <= 0.0f) continue;
        float r = world_r * k / proj.z;

        float dx = (float)px - proj.x;
        float dy = (float)py - proj.y;
        float d2 = dx * dx + dy * dy;
        float r2 = r * r;
        if (d2 >= r2) continue;                  // this pixel misses the sphere

        // Height of the sphere's front surface above the disc, as a fraction of
        // the radius. z-component of the unit normal.
        float this_nz = sqrtf(1.0f - d2 / r2);

        // Camera-space depth of that surface point, so primitives at different
        // distances occlude each other correctly.
        float depth = proj.z - this_nz * world_r;

        if (depth < best_depth) {
            best_depth   = depth;
            best_color   = p.color;
            best_opacity = p.opacity;
            nx =  dx / r;
            ny = -dy / r;      // screen y grows downward, world y grows upward
            nz =  this_nz;
            hit = true;
        }
    }

    // ----- Bonds: ray against a screen-space capsule around the projected axis -----
    for (int i = 0; i < num_tubes; i++) {
        Cuda::Line ln = tubes[i];
        if (ln.opacity <= 0.0f) continue;

        bool behind_a = false, behind_b = false;
        Cuda::vec3 a, b;
        Cuda::d_coordinate_to_pixel(ln.start, behind_a, camera_direction, camera_pos, fov, geom_mean_size, wh, a);
        Cuda::d_coordinate_to_pixel(ln.end,   behind_b, camera_direction, camera_pos, fov, geom_mean_size, wh, b);
        if (behind_a || behind_b) continue;

        float world_r = tube_size * radius_multiplier;
        if (world_r <= 0.0f) continue;

        // Foot of the perpendicular from this pixel onto the projected bond axis,
        // clamped to the segment (which rounds the ends into caps. harmless,
        // since the caps are not visible).
        float ax = b.x - a.x, ay = b.y - a.y;
        float len2 = ax * ax + ay * ay;
        float t = 0.0f;
        if (len2 > 1e-6f) {
            t = (((float)px - a.x) * ax + ((float)py - a.y) * ay) / len2;
            t = Cuda::clamp(t, 0.0f, 1.0f);
        }

        // Depth along the bond at that point, which sets how thick the tube looks
        // there - so a bond angled away from the camera tapers, as it should.
        float axis_depth = a.z + t * (b.z - a.z);
        float r = world_r * k / axis_depth;

        float dx = (float)px - (a.x + t * ax);
        float dy = (float)py - (a.y + t * ay);
        float d2 = dx * dx + dy * dy;
        float r2 = r * r;
        if (d2 >= r2) continue;                  // this pixel misses the tube

        // Same normal construction as the sphere, measured from the axis instead
        // of from a center: bright along the lit side, falling off to the edges,
        // which is exactly what makes it read as a cylinder.
        float this_nz = sqrtf(1.0f - d2 / r2);

        float depth = axis_depth - this_nz * world_r;

        if (depth < best_depth) {
            best_depth   = depth;
            best_color   = ln.color;
            best_opacity = ln.opacity;
            nx =  dx / r;
            ny = -dy / r;
            nz =  this_nz;
            hit = true;
        }
    }

    if (!hit) return;      // nothing here, leave the pixel alone

    // One thread owns this pixel, so no atomics are needed.
    overlay_pixel(Cuda::ivec2(px, py), shade_surface(nx, ny, nz, best_color),
                  opacity * best_opacity, pixels, wh);
}

extern "C" void render_ball_and_stick_on_gpu(
    unsigned int* d_pixels, const Cuda::ivec2& wh,
    float geom_mean_size, float opacity, float radius_multiplier,
    Cuda::Point* h_spheres, int num_spheres,
    Cuda::Line*  h_tubes,   int num_tubes, float tube_size,
    const Cuda::quat& camera_direction, const Cuda::vec3& camera_pos, float fov)
{
    if (num_spheres <= 0 && num_tubes <= 0) return;

    Cuda::Point* d_spheres = nullptr;
    Cuda::Line*  d_tubes   = nullptr;
    size_t sphere_bytes = (size_t)num_spheres * sizeof(Cuda::Point);
    size_t tube_bytes   = (size_t)num_tubes   * sizeof(Cuda::Line);

    if (num_spheres > 0) {
        cudaMalloc((void**)&d_spheres, sphere_bytes);
        cudaMemcpy(d_spheres, h_spheres, sphere_bytes, cudaMemcpyHostToDevice);
    }
    if (num_tubes > 0) {
        cudaMalloc((void**)&d_tubes, tube_bytes);
        cudaMemcpy(d_tubes, h_tubes, tube_bytes, cudaMemcpyHostToDevice);
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((wh.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (wh.y + threadsPerBlock.y - 1) / threadsPerBlock.y);
    render_ball_and_stick_kernel<<<numBlocks, threadsPerBlock>>>(
        d_pixels, wh,
        geom_mean_size, opacity, radius_multiplier,
        d_spheres, num_spheres,
        d_tubes,   num_tubes, tube_size,
        camera_direction, camera_pos, fov);
    cudaDeviceSynchronize();

    if (d_spheres) cudaFree(d_spheres);
    if (d_tubes)   cudaFree(d_tubes);
}
