// outer_billiards_flow.cu
// ---------------------------------------------------------------------------
// Where every point goes, as a color. One thread per pixel, one orbit per
// thread - see FlowFieldParams in Host_Device_Shared/OuterBilliardsShared.h for
// what the picture means.
//
// The plane is painted with a color wheel: hue from the direction, white at the
// middle, black at infinity. Each pixel is then given the color of the place its
// own orbit has reached after n hops. The pixel does not move; its color does.
//
// Reading it:
//
//   COHERENT PATCHES  a region where the iterate is a rigid motion - a periodic
//                     island. The wheel is carried around intact, so the patch
//                     keeps its shape and its color no matter how far n runs.
//   NOISE             a region where nearby points separate. Pixels that began
//                     as neighbours end up unrelated, and the coloring shreds.
//   DARKENING         an orbit on its way to infinity, running off the bright end
//                     of the wheel.
//
// Because the whole point is that adjacent pixels disagree, this does NOT
// antialias by default: averaging chaos gives grey, which is a different (and
// also interesting) picture, so `samples` is left at 1 unless a caller asks.
//
// `iterations` is real, and the fractional part is spent TURNING - see
// outer_billiards_turn. That matters more than it sounds: it is what makes this
// safe to animate at all. The map is a half turn about a vertex, so a
// half-finished hop is a quarter turn, and the coloring moves continuously
// instead of jumping between whole iterates.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdint>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "../Host_Device_Shared/Color.h"
#include "../Host_Device_Shared/OuterBilliardsShared.h"
#include "color.cuh"

#define FLOW_MIN_OPACITY BILLIARDS_MIN_OPACITY

// The color wheel the plane is painted with, as a function of position.
//
// Distance is read as latitude on a sphere whose pole is the center and whose
// other pole is infinity: white at the middle, saturating to a full hue at
// `scale`, then darkening away to black. So a point that has run off carries no
// color at all, which is exactly what should happen to an orbit that escapes.
__device__ __forceinline__ uint32_t flow_wheel(const Cuda::vec2& q, const Cuda::vec2& center,
                                               float scale, float curvature, int shade_by_distance) {
    const Cuda::vec2 offset = q - center;

    // Direction is the Euclidean angle even in a curved plane: geodesics through
    // the center are straight in these coordinates, so for a wheel centered on
    // the table this is the geodesic bearing.
    float hue = atan2f(offset.y, offset.x) * 0.15915494f;   // / 2 pi
    if (hue < 0.0f) hue += 1.0f;

    if (!shade_by_distance) return Cuda::HSVtoRGB(hue, 1.0f, 1.0f);

    // Distance in the plane's own metric, so the wheel does not stretch when the
    // curvature is animated.
    const float radius = Cuda::curved_distance(center, q, curvature);
    const float latitude = 0.63661977f * atanf(radius / fmaxf(scale, 1e-6f));   // 2/pi * atan, in [0,1)

    float saturation, value;
    if (latitude < 0.5f) { saturation = 2.0f * latitude; value = 1.0f; }
    else                 { saturation = 1.0f;            value = 2.0f * (1.0f - latitude); }
    return Cuda::HSVtoRGB(hue, saturation, value);
}

__global__ void flow_field_kernel(
    uint32_t* pixels, const Cuda::ivec2 wh,
    const Cuda::FlowFieldParams params,
    const int whole, const float fraction)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= wh.x || py >= wh.y) return;

    const Cuda::vec2 wh_f(wh.x, wh.y);
    const int side = params.samples < 1 ? 1 : params.samples;
    const float step = 1.0f / (float)side;

    float red = 0.0f, green = 0.0f, blue = 0.0f;
    int taken = 0;

    for (int sy = 0; sy < side; sy++) {
        for (int sx = 0; sx < side; sx++) {
            const Cuda::vec2 sample((float)px + ((float)sx + 0.5f) * step,
                                    (float)py + ((float)sy + 0.5f) * step);
            Cuda::vec2 p = Cuda::pixel_to_point_in_screen(sample, params.lx_ty, params.rx_by, wh_f);

            // A pixel is a Poincare-disk coordinate instead of a Klein one, so
            // swap in the Klein point of the same abstract place before any of
            // the math below - which stays entirely in Klein coordinates, same
            // as always.
            if (params.poincare_view != 0) {
                const float horizon = params.curvature < 0.0f ? 1.0f / sqrtf(-params.curvature) : 0.0f;
                p = Cuda::poincare_to_klein(p, horizon);
            }

            // Inside the table, or outside the plane, there is no orbit to follow.
            if (Cuda::outer_billiards_pivot(params.verts, params.n, p, params.curvature) < 0) continue;

            // An unbounded orbit (see outer_billiards_singularity.cu's kite, or
            // any table in a curved plane whose points climb toward the ideal
            // boundary) grows every hop, and nothing bounds how far - a fixed
            // iteration count never used to run long enough for that to matter,
            // but flow_auto_depth can now ask for tens of thousands of hops. Once
            // a point is this far out it is already reading as escaped (flow_wheel
            // fades it toward black by distance), so stopping here changes
            // nothing about the picture - it only keeps the float32 arithmetic in
            // outer_billiards_reflect from compounding past what it can represent.
            // Past that point `curved_norm` can hit `curvature * inf`, which is
            // NaN even for curvature == 0, and one NaN pixel is a visible fleck
            // no amount of extra depth would ever fix.
            const float ESCAPE_RADIUS_SQ = 1e12f;
            for (int k = 0; k < whole; k++) {
                const Cuda::vec2 next = Cuda::outer_billiards_hop(params.verts, params.n, p, params.curvature);
                if (!(Cuda::dot(next, next) < ESCAPE_RADIUS_SQ)) break;   // also catches nan/inf: comparison is false
                p = next;
            }

            // The fraction either carries the destination partway through the hop
            // it is in the middle of - so the coloring flows continuously - or is
            // ignored, holding each whole iterate until the next one lands. Both
            // agree exactly on the whole numbers; only the way between differs.
            //
            // Partway through a hop means partway through the TURN. A hop is a
            // half turn about the pivot, so half a hop is a quarter turn about
            // it, and every point of the wedge is still somewhere different.
            // Sliding along the chord from p to T(p) instead would send the whole
            // wedge through its own pivot at the halfway mark - see
            // outer_billiards_turn.
            Cuda::vec2 destination = p;
            if (params.smooth != 0 && fraction > 1e-4f) {
                const int pivot = Cuda::outer_billiards_tangent_vertex(params.verts, params.n, p);
                destination = Cuda::outer_billiards_turn(params.verts[pivot], p, params.curvature,
                                                         fraction * 3.14159265f);
            }

            const uint32_t color = flow_wheel(destination, params.center, params.scale, params.curvature,
                                              params.shade_by_distance);
            red   += Cuda::getr(color);
            green += Cuda::getg(color);
            blue  += Cuda::getb(color);
            taken++;
        }
    }

    if (taken == 0) return;   // every sample landed on the table; leave the pixel alone

    const float share = 1.0f / (float)taken;
    const uint32_t color = Cuda::argb(255, (int)(red * share), (int)(green * share), (int)(blue * share));

    // Samples that missed the plane do not darken the ones that hit it; they just
    // do not vote, so the edge of the table stays clean.
    const float coverage = (float)taken / (float)(side * side);
    overlay_pixel(Cuda::ivec2(px, py), color, params.opacity * coverage, pixels, wh);
}

extern "C" void outer_billiards_flow_render(
    uint32_t* d_pixels, const Cuda::ivec2& wh,
    const Cuda::FlowFieldParams& params)
{
    if (params.n < 3 || wh.x <= 0 || wh.y <= 0) return;
    if (params.opacity <= FLOW_MIN_OPACITY) return;

    const float count = params.iterations > 0.0f ? params.iterations : 0.0f;
    const int whole = (int)floorf(count);
    const float fraction = count - (float)whole;

    dim3 block(16, 16);
    dim3 grid((wh.x + block.x - 1) / block.x, (wh.y + block.y - 1) / block.y);
    flow_field_kernel<<<grid, block>>>(d_pixels, wh, params, whole, fraction);
    cudaDeviceSynchronize();
}
