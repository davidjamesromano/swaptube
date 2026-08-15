// outer_billiards_singularity.cu
// ---------------------------------------------------------------------------
// The singularity graph of an outer billiards table, and the periodic islands
// it leaves behind. One thread per pixel, one forward orbit per thread.
//
// Host_Device_Shared/OuterBilliardsShared.h has the mathematics and, in
// particular, the identity this whole file rests on:
//
//     dist(p, T^-k(R)) = dist(T^k(p), R)
//
// The k'th layer of the graph is the set of points that reach a singular ray in
// k hops. Nothing here ever constructs that set. A thread walks its own pixel
// forward instead, and at every step asks how far the CURRENT point is from the
// n rays - which, because each piece of T is a half-turn and rotations keep
// distances, is how far the ORIGINAL point was from that layer. Taking the best
// answer over all layers gives a signed-distance field for the whole fractal,
// evaluated in a loop with no memory traffic.
//
// Two things fall out of that for free. Lines come out antialiased, because a
// distance can be compared against a pixel's width instead of rounded to it. And
// they stay one pixel wide at any zoom, because that comparison is made in world
// units against the size a pixel actually covers - a rasterized version would
// thin out to nothing as you zoomed in on a preimage that fine.
//
// The distances are the CURVED ones, so the same argument carries the same
// picture into the hyperbolic plane: the map is an isometry there too, and the
// only thing the kernel has to do differently is convert a hyperbolic distance
// into a screen distance before comparing it to a line width.
//
// Both layers - web and islands - come out of the SAME walk, so drawing them
// together costs barely more than drawing either alone.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdint>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "../Host_Device_Shared/OuterBilliardsShared.h"
#include "color.cuh"

// Below this an opacity cannot change a pixel.
#define SINGULARITY_MIN_OPACITY BILLIARDS_MIN_OPACITY

// How much of this pixel the nearest line covers. `half_width` and `distance`
// are both in world units; the ramp is one pixel wide, so the edge softens over
// exactly the distance the screen cannot resolve.
__device__ __forceinline__ float line_coverage(float distance, float half_width, float world_per_pixel) {
    return Cuda::clamp((half_width + 0.5f * world_per_pixel - distance) / world_per_pixel, 0.0f, 1.0f);
}

__global__ void singularity_graph_kernel(
    uint32_t* pixels, const Cuda::ivec2 wh,
    const Cuda::SingularityGraphParams params,
    const int steps, const int web_steps, const int island_steps)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= wh.x || py >= wh.y) return;

    Cuda::vec2 start = Cuda::pixel_to_point_in_screen(
        Cuda::vec2(px, py), params.lx_ty, params.rx_by, Cuda::vec2(wh.x, wh.y));

    // See outer_billiards_flow.cu: a pixel is a Poincare-disk coordinate, so
    // swap in the Klein point of the same abstract place before anything below,
    // which is all still Klein-coordinate math.
    if (params.poincare_view != 0) {
        const float horizon = params.curvature < 0.0f ? 1.0f / sqrtf(-params.curvature) : 0.0f;
        start = Cuda::poincare_to_klein(start, horizon);
    }

    // Inside the table the map has nothing to say, and no orbit ever enters - so
    // this is the only step where the check is worth its second sweep. It also
    // catches pixels beyond the ideal boundary, which are not in the plane at all.
    const int pivot = Cuda::outer_billiards_pivot(params.verts, params.n, start, params.curvature);
    if (pivot < 0) return;

    const float wpp        = params.world_per_pixel;
    const float half_width = fmaxf(params.line_width, 0.0f) * 0.5f * wpp;
    const float halo       = fmaxf(4.0f * half_width, 1e-20f);
    const bool  want_glow  = params.glow > 0.001f;

    // Every distance below is measured in the curved plane; this is what a unit
    // of it is worth on screen, HERE. It depends only on where the pixel is, not
    // on where its orbit wanders, so it is computed once.
    const float to_screen = Cuda::curved_screen_scale(start, params.curvature);

    const bool want_islands = params.island_opacity > SINGULARITY_MIN_OPACITY && params.max_period > 1;
    const float start_norm = Cuda::curved_norm(start, params.curvature);

    // The brightest layer wins the pixel rather than the sum of them, so a dense
    // corner of the web reads as one crisp line instead of blowing out to white.
    float    web_intensity = 0.0f;
    uint32_t web_color     = params.line_color;

    // How near the whole graph passes this pixel, in screen units - which is what
    // says whether the pixel is in one of the gaps the graph leaves.
    float nearest = 1e30f;

    // Which hop brought the orbit back closest to where it set off, and how close.
    // In the Euclidean plane the closest return is exactly zero and this is the
    // period; in a curved one nothing returns exactly, but the return map on an
    // island is a rigid rotation, so the hop that comes back nearest is the same
    // hop for the WHOLE island - which is what makes it a usable flat color.
    float closest_return = 1e30f;
    int   return_hop     = 0;

    Cuda::vec2 p = start;
    for (int k = 0; k < steps; k++) {
        // The island boundary is taken from more layers than are drawn - see
        // island_depth. Past both, the walk continues only to find the closest
        // return, and the distance to the graph is no longer worth computing.
        if (k < web_steps || k < island_steps) {
            const float d = Cuda::outer_billiards_singular_distance(params.rays, params.ray_count, p,
                                                                    params.curvature) * to_screen;
            if (d < nearest) nearest = d;

            if (k < web_steps) {
                // Layer k fades in over the last unit of depth, so a ramping
                // `depth` grows the graph continuously instead of snapping a
                // layer at a time.
                float weight = Cuda::clamp(params.depth - (float)k, 0.0f, 1.0f);
                if (params.fade > 0.0f) weight *= 1.0f - params.fade * (float)k / fmaxf(params.depth, 1.0f);

                if (weight > 0.0f) {
                    float intensity = line_coverage(d, half_width, wpp);
                    if (want_glow) {
                        const float soft = params.glow * __expf(-d / halo);
                        intensity = 1.0f - (1.0f - intensity) * (1.0f - soft);
                    }
                    intensity *= weight;
                    if (intensity > web_intensity) {
                        web_intensity = intensity;
                        web_color = params.rainbow > 0.001f
                            ? Cuda::colorlerp(params.line_color,
                                              Cuda::rainbow((float)k / params.rainbow_period),
                                              params.rainbow)
                            : params.line_color;
                    }
                }
            }
        }

        p = Cuda::outer_billiards_hop(params.verts, params.n, p, params.curvature);

        if (want_islands && k < params.max_period) {
            // Compared, never reported, so the cheap monotone stand-in will do.
            const float back = Cuda::curved_closeness(start, p, start_norm, params.curvature);
            if (back < closest_return) { closest_return = back; return_hop = k + 1; }
        }
    }

    const int index = py * wh.x + px;
    uint32_t out = pixels[index];

    // Islands underneath, web on top - the graph is the boundary of the islands,
    // and burying it under a flat fill would be backwards.
    //
    // The fill is exactly what the web's own coverage leaves over, so it reaches
    // the line and stops, with the same one-pixel ramp holding the seam together.
    // The glow is deliberately left out of that: a halo is not a boundary, and
    // subtracting it would eat a ring out of every island.
    if (return_hop > 0) {
        const float fill = 1.0f - line_coverage(nearest, half_width, wpp);
        if (fill > 0.0f) {
            out = Cuda::color_combine(out, Cuda::rainbow(__log2f((float)return_hop) / params.period_octaves),
                                      fill * params.island_opacity);
        }
    }
    if (web_intensity > 0.0f) {
        out = Cuda::color_combine(out, web_color, Cuda::clamp(web_intensity * params.web_opacity, 0.0f, 1.0f));
    }
    pixels[index] = out;
}

// ---------------------------------------------------------------------------
// Host-callable launcher. The table travels inside the parameter block, so there
// is nothing to allocate, copy, or free around the launch.
// ---------------------------------------------------------------------------
extern "C" void outer_billiards_singularity_render(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    const Cuda::SingularityGraphParams& params)
{
    if (params.n < 3 || params.ray_count < 3 || wh.x <= 0 || wh.y <= 0) return;

    // Islands are the gaps in the graph, so without a graph there is nothing for
    // them to be the gaps in.
    const bool want_web = params.web_opacity > SINGULARITY_MIN_OPACITY && params.depth > 0.0f;
    const bool want_islands = params.island_opacity > SINGULARITY_MIN_OPACITY
                           && params.max_period > 1 && params.depth > 0.0f;
    if (!want_web && !want_islands) return;

    // Layer k needs k hops to reach, and layer k contributes only while depth > k
    // - so ceil(depth) layers are drawn. The islands walk further, both to place
    // their boundary against a deeper graph than the visible one and to find the
    // closest return.
    const int web_steps    = want_web ? (int)ceilf(params.depth) : 0;
    const int island_steps = want_islands ? params.island_depth : 0;
    int steps = web_steps;
    if (island_steps > steps) steps = island_steps;
    if (want_islands && params.max_period > steps) steps = params.max_period;
    if (steps <= 0) return;

    dim3 block(16, 16);
    dim3 grid((wh.x + block.x - 1) / block.x, (wh.y + block.y - 1) / block.y);
    singularity_graph_kernel<<<grid, block>>>(d_pixels, wh, params, steps, web_steps, island_steps);
    cudaDeviceSynchronize();
}
