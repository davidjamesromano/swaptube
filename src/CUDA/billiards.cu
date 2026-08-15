// billiards.cu
// ---------------------------------------------------------------------------
// A small antialiased 2D vector renderer: filled convex polygons, thick line
// segments, and round dots. Written for OuterBilliardsScene, which needs to lay
// down a table outline and then a few thousand orbit hops on top of it, but
// nothing here knows anything about billiards.
//
// Parallelism: one thread per PIXEL, walking every primitive - the same trade
// ball_and_stick.cu makes, and for the same two reasons:
//
//   1. Antialiasing and blending come out right. Each thread composites its own
//      pixel in array order and writes it once, so overlapping segments blend
//      deterministically instead of racing (the per-primitive kernels in
//      3d_points_lines.cu stamp with atomicCAS and let the winner be whoever
//      finishes first).
//   2. One launch draws the whole batch. An orbit is hundreds or thousands of
//      segments; a launch apiece would be all overhead.
//
// Each primitive is bounding-box rejected before the real distance test, so the
// per-pixel loop costs a couple of compares for the (many) primitives nowhere
// near it. Successive launches on the same stream stay ordered, so the caller
// controls layering by choosing the order it calls these in.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdint>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "../Host_Device_Shared/BilliardsStructs.h"
#include "color.cuh"

// How much of a pixel a shape covers, given the distance from the pixel to the
// shape's centerline and the shape's half-width. This one-pixel ramp across the
// edge is the whole antialiasing story.
__device__ __forceinline__ float edge_coverage(float distance, float half_width) {
    return Cuda::clamp(half_width + 0.5f - distance, 0.0f, 1.0f);
}

__device__ __forceinline__ float distance_to_segment(const Cuda::vec2& p, const Cuda::vec2& a, const Cuda::vec2& b) {
    const Cuda::vec2 ab = b - a;
    const Cuda::vec2 ap = p - a;
    const float len_sq = Cuda::dot(ab, ab);
    // A degenerate segment is just its own start point.
    const float t = (len_sq > 1e-12f) ? Cuda::clamp(Cuda::dot(ap, ab) / len_sq, 0.0f, 1.0f) : 0.0f;
    return Cuda::length(p - (a + ab * t));
}

// ---------------------------------------------------------------------------
// Convex polygon fill.
//
// For a polygon wound counterclockwise, a point is inside exactly when it lies
// to the left of every directed edge. Keeping the SMALLEST of those signed
// distances (rather than just the sign) gives the distance to the nearest edge,
// negative outside - which feeds straight into the same one-pixel coverage ramp
// the other two kernels use. The caller guarantees the winding.
// ---------------------------------------------------------------------------
__global__ void billiards_polygon_kernel(
    uint32_t* pixels, const Cuda::ivec2 wh,
    const Cuda::vec2* verts, const int n,
    const uint32_t color, const float opacity,
    const Cuda::ivec2 min_pos, const Cuda::ivec2 max_pos)
{
    const Cuda::ivec2 pos = min_pos + Cuda::ivec2(blockIdx.x * blockDim.x + threadIdx.x,
                                                  blockIdx.y * blockDim.y + threadIdx.y);
    if (pos.x >= max_pos.x || pos.y >= max_pos.y || pos.x >= wh.x || pos.y >= wh.y) return;

    const Cuda::vec2 p(pos.x, pos.y);

    float nearest = 1e30f;
    for (int i = 0; i < n; i++) {
        const Cuda::vec2 a = verts[i];
        const Cuda::vec2 b = verts[(i + 1) % n];
        const Cuda::vec2 ab = b - a;
        const float len = Cuda::length(ab);
        if (len < 1e-6f) continue;   // coincident vertices carry no edge
        const Cuda::vec2 ap = p - a;
        const float signed_distance = (ab.x * ap.y - ab.y * ap.x) / len;   // > 0 to the left
        if (signed_distance < nearest) nearest = signed_distance;
    }

    const float coverage = edge_coverage(-nearest, 0.0f);
    if (coverage <= 0.0f) return;
    overlay_pixel(pos, color, coverage * opacity, pixels, wh);
}

__global__ void billiards_segments_kernel(
    uint32_t* pixels, const Cuda::ivec2 wh,
    const Cuda::Segment2D* segments, const int n)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= wh.x || py >= wh.y) return;

    const Cuda::vec2 p(px, py);
    const int index = py * wh.x + px;
    uint32_t accumulated = pixels[index];
    bool touched = false;

    for (int i = 0; i < n; i++) {
        const Cuda::Segment2D s = segments[i];
        const float half = s.thickness * 0.5f;
        const float margin = half + 1.0f;
        // Cheap rejection first: the overwhelming majority of segments are
        // nowhere near this pixel.
        if (p.x < fminf(s.a.x, s.b.x) - margin || p.x > fmaxf(s.a.x, s.b.x) + margin ||
            p.y < fminf(s.a.y, s.b.y) - margin || p.y > fmaxf(s.a.y, s.b.y) + margin) continue;

        const float coverage = edge_coverage(distance_to_segment(p, s.a, s.b), half);
        if (coverage <= 0.0f) continue;
        accumulated = Cuda::color_combine(accumulated, s.color, Cuda::clamp(coverage * s.opacity, 0.0f, 1.0f));
        touched = true;
    }

    if (touched) pixels[index] = accumulated;
}

__global__ void billiards_dots_kernel(
    uint32_t* pixels, const Cuda::ivec2 wh,
    const Cuda::Dot2D* dots, const int n)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= wh.x || py >= wh.y) return;

    const Cuda::vec2 p(px, py);
    const int index = py * wh.x + px;
    uint32_t accumulated = pixels[index];
    bool touched = false;

    for (int i = 0; i < n; i++) {
        const Cuda::Dot2D d = dots[i];
        const float margin = d.radius + 1.0f;
        if (p.x < d.center.x - margin || p.x > d.center.x + margin ||
            p.y < d.center.y - margin || p.y > d.center.y + margin) continue;

        const float coverage = edge_coverage(Cuda::length(p - d.center), d.radius);
        if (coverage <= 0.0f) continue;
        accumulated = Cuda::color_combine(accumulated, d.color, Cuda::clamp(coverage * d.opacity, 0.0f, 1.0f));
        touched = true;
    }

    if (touched) pixels[index] = accumulated;
}

// ---------------------------------------------------------------------------
// Host-callable launchers. The batches are a few kilobytes at most, so each call
// simply ships its array to the GPU, launches, and frees again - the pattern
// scattering.cu uses.
// ---------------------------------------------------------------------------

extern "C" void billiards_fill_polygon(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    const Cuda::vec2* h_verts, int n,
    uint32_t color, float opacity)
{
    if (n < 3 || opacity <= 0.0f) return;

    // Only the polygon's own bounding box can be covered.
    float lx = h_verts[0].x, rx = h_verts[0].x, ty = h_verts[0].y, by = h_verts[0].y;
    for (int i = 1; i < n; i++) {
        lx = fminf(lx, h_verts[i].x); rx = fmaxf(rx, h_verts[i].x);
        ty = fminf(ty, h_verts[i].y); by = fmaxf(by, h_verts[i].y);
    }
    const int left = (int)floorf(lx) - 1, top    = (int)floorf(ty) - 1;
    const int right = (int)ceilf(rx) + 1, bottom = (int)ceilf(by)  + 1;
    const Cuda::ivec2 min_pos(left  > 0     ? left  : 0,     top    > 0     ? top    : 0);
    const Cuda::ivec2 max_pos(right < wh.x  ? right : wh.x,  bottom < wh.y  ? bottom : wh.y);
    const Cuda::ivec2 size = max_pos - min_pos;
    if (size.x <= 0 || size.y <= 0) return;

    Cuda::vec2* d_verts = nullptr;
    const size_t bytes = (size_t)n * sizeof(Cuda::vec2);
    cudaMalloc(&d_verts, bytes);
    cudaMemcpy(d_verts, h_verts, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y);
    billiards_polygon_kernel<<<grid, block>>>(d_pixels, wh, d_verts, n, color, opacity, min_pos, max_pos);
    cudaDeviceSynchronize();

    cudaFree(d_verts);
}

extern "C" void billiards_draw_segments(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    const Cuda::Segment2D* h_segments, int n)
{
    if (n <= 0) return;

    Cuda::Segment2D* d_segments = nullptr;
    const size_t bytes = (size_t)n * sizeof(Cuda::Segment2D);
    cudaMalloc(&d_segments, bytes);
    cudaMemcpy(d_segments, h_segments, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((wh.x + block.x - 1) / block.x, (wh.y + block.y - 1) / block.y);
    billiards_segments_kernel<<<grid, block>>>(d_pixels, wh, d_segments, n);
    cudaDeviceSynchronize();

    cudaFree(d_segments);
}

extern "C" void billiards_draw_dots(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    const Cuda::Dot2D* h_dots, int n)
{
    if (n <= 0) return;

    Cuda::Dot2D* d_dots = nullptr;
    const size_t bytes = (size_t)n * sizeof(Cuda::Dot2D);
    cudaMalloc(&d_dots, bytes);
    cudaMemcpy(d_dots, h_dots, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((wh.x + block.x - 1) / block.x, (wh.y + block.y - 1) / block.y);
    billiards_dots_kernel<<<grid, block>>>(d_pixels, wh, d_dots, n);
    cudaDeviceSynchronize();

    cudaFree(d_dots);
}
