#pragma once

#include <cstdint>
#include "vec.h"
#include "shared_precompiler_directives.h"

// Below this, a layer is visually indistinguishable from off - shared so the
// CPU scene and both GPU kernels agree on when a layer is worth paying for.
#define BILLIARDS_MIN_OPACITY 0.004f

// ---------------------------------------------------------------------------
// The outer billiards map, written once so the CPU (which traces orbits as
// polylines, in DataObjects/OuterBilliards.h) and the GPU (which renders the
// singularity graph per pixel, in src/CUDA/outer_billiards_singularity.cu)
// can never disagree about where a hop lands.
//
// THE MAP. From a point p outside a convex table there are two tangent lines;
// take the one that leaves the table on your left as seen from p, and reflect
// p through the vertex v it touches: T(p) = 2v - p. That vertex is the
// CLOCKWISE-MOST one as seen from p, found in one sweep by
// outer_billiards_tangent_vertex.
//
// THE SINGULAR SET. Which vertex T pivots about changes exactly when two
// vertices line up with p, so the exterior splits into n wedges - one per
// vertex - cut by the n rays R_i = { v_i + t(v_i - v_{i+1}) : t >= 0 }
// (vertices wound counterclockwise). On those rays T has no answer, and the
// singularity graph is where every point eventually LANDS on one:
//
//     S_D = union over k = 0..D of T^-k(R)
//
// T is an isometry on each wedge, so dist(p, T^-k(R)) = dist(T^k(p), R) -
// which turns the fractal into a distance field one forward orbit deep, with
// no line clipping or subdivision. That identity is why the renderer is a
// per-pixel kernel and not a pile of segments.
//
// CURVATURE. Every function below takes curvature K <= 0 (0 = Euclidean).
// K < 0 draws the hyperbolic plane of that curvature in the BELTRAMI-KLEIN
// model: the plane compressed into a disk of radius R = 1/sqrt(-K), where
// geodesics are straight chords - so the table, the singular rays and every
// orbit hop stay line segments exactly as in the flat case, and the tangency
// search (an incidence question) is untouched.
//
// What DOES change is the reflection and the metric. Lift q to (q,1) in R^3
// and use the bilinear form <u,v> = K(u_x v_x + u_y v_y) + u_z v_z (Minkowski
// for K < 0, degenerate but not singular at K = 0). With N(q) = <(q,1),(q,1)>
// = 1 + K|q|^2, rotation by pi about v works out to
//
//     T_v(q) = (A v - q) / (A - 1),    A = 2 (K q.v + 1) / (K |v|^2 + 1)
//
// which is exactly A = 2, T_v(q) = 2v - q at K = 0. Every formula below is
// arranged this way - one expression, no branch on curvature, Euclidean
// recovered exactly in the limit - so a scene can ANIMATE curvature and watch
// the plane curl into a disk.
// ---------------------------------------------------------------------------

SHARED_FILE_PREFIX

// The table rides inside the kernel's parameter block rather than a buffer of
// its own - costs a copy, saves a cudaMalloc every frame.
#define MAX_BILLIARD_VERTICES 32

// ---------------------------------------------------------------------------
// Curvature primitives
// ---------------------------------------------------------------------------

// N(q) = 1 + K|q|^2: always 1 in the Euclidean plane; in the hyperbolic plane
// it runs from 1 at the center to 0 at the ideal boundary and negative past it.
HOST_DEVICE inline float curved_norm(const vec2& q, float curvature) {
    return 1.0f + curvature * dot(q, q);
}

HOST_DEVICE inline bool curved_in_plane(const vec2& q, float curvature) {
    return curved_norm(q, curvature) > 1e-7f;
}

// asinh(x sqrt(-K)) / sqrt(-K) - just x at K = 0.
HOST_DEVICE inline float curved_arcsinh(float x, float curvature) {
    const float k = -curvature;
    if (k < 1e-9f) return x;
    const float bend = sqrtf(k);
    return asinhf(bend * x) / bend;
}

// How much shorter a hyperbolic length looks on screen. The Klein metric is
// anisotropic (1/N radially, 1/sqrt(N) across), so this is their geometric
// mean - exact at K = 0, and only ever used to set line thickness, never
// where a line goes.
HOST_DEVICE inline float curved_screen_scale(const vec2& q, float curvature) {
    const float n = curved_norm(q, curvature);
    if (n <= 0.0f) return 0.0f;
    return powf(n, 0.75f);
}

// (cosh(d sqrt(-K)) - 1) / (-2K) - a strictly increasing function of geodesic
// distance, cheaper than curved_distance (no sqrt), so anything that only
// COMPARES distances should call this instead. Written through
// cosh(d)-1 = 2 sinh^2(d/2) rather than arccosh, which keeps its precision
// when the two points are close - the common case for a fractal made of
// near-misses. `a_norm` is curved_norm(a), passed in so callers sweeping many
// b against a fixed a don't recompute it.
HOST_DEVICE inline float curved_closeness(const vec2& a, const vec2& b, float a_norm, float curvature) {
    const vec2 delta = a - b;
    const float flat = dot(delta, delta);
    if (curvature == 0.0f) return flat * 0.25f;

    const float nb = curved_norm(b, curvature);
    if (a_norm <= 0.0f || nb <= 0.0f) return 1e30f;

    const float cross = a.x * b.y - a.y * b.x;
    const float inner = 1.0f + curvature * dot(a, b);
    const float geo   = sqrtf(a_norm * nb);
    const float denom = 2.0f * geo * (inner + geo);
    if (denom <= 1e-30f) return 1e30f;
    return (flat + curvature * cross * cross) / denom;
}

// Geodesic distance from a to b.
HOST_DEVICE inline float curved_distance(const vec2& a, const vec2& b, float curvature) {
    const float half_sq = curved_closeness(a, b, curved_norm(a, curvature), curvature);
    if (half_sq >= 1e29f) return 1e30f;
    return 2.0f * curved_arcsinh(sqrtf(half_sq > 0.0f ? half_sq : 0.0f), curvature);
}

// ---------------------------------------------------------------------------
// KLEIN <-> POINCARE. Same hyperbolic plane, same disk of radius `horizon`,
// but Klein draws geodesics as straight chords and Poincare as arcs bowing
// toward the center. The map, the metric and every distance above are all
// defined in KLEIN coordinates; converting only decides where a point gets
// DRAWN. `horizon <= 0` is the identity both ways - there is no disk to
// normalize against. Standard unit-disk formulas, exact inverses of each other.
// ---------------------------------------------------------------------------
HOST_DEVICE inline vec2 klein_to_poincare(const vec2& q, float horizon) {
    if (!(horizon > 1e-6f)) return q;
    const vec2 u = q * (1.0f / horizon);
    const float len_sq = dot(u, u);
    if (len_sq >= 1.0f) return q;   // already at (or past) the shared boundary
    const float denom = 1.0f + sqrtf(1.0f - len_sq);
    return u * (horizon / denom);
}

HOST_DEVICE inline vec2 poincare_to_klein(const vec2& p, float horizon) {
    if (!(horizon > 1e-6f)) return p;
    const vec2 u = p * (1.0f / horizon);
    const float len_sq = dot(u, u);
    if (len_sq >= 1.0f) return p;
    return u * (2.0f * horizon / (1.0f + len_sq));
}

// Positive when b is counterclockwise of a.
HOST_DEVICE inline float billiards_cross(const vec2& a, const vec2& b) { return a.x * b.y - a.y * b.x; }

// ---------------------------------------------------------------------------
// The map
// ---------------------------------------------------------------------------

// The vertex the tangent line from p touches: from a point OUTSIDE the table
// every vertex lies within half a turn, so "counterclockwise of" is a genuine
// ordering and one sweep finds the clockwise-most vertex. Winding- and
// curvature-agnostic - it only looks at directions from p.
//
// Exactly collinear means p sits on an edge's line, where the map is
// discontinuous anyway; take the farther vertex so the choice is at least
// consistent.
HOST_DEVICE inline int outer_billiards_tangent_vertex(const vec2* verts, int n, const vec2& p) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        const vec2 a = verts[best] - p;
        const vec2 b = verts[i] - p;
        const float turn = billiards_cross(a, b);
        if (turn < 0.0f) best = i;
        else if (turn == 0.0f && dot(b, b) > dot(a, a)) best = i;
    }
    return best;
}

// That sweep assumed p was outside: check it by requiring every vertex lie
// counterclockwise of the winner. The threshold has to be RELATIVE - a cross
// product from p grows like distance^2, and so does its rounding error, so an
// absolute threshold gets swamped in the far field and reports plainly-outside
// points as inside at random. Comparing against |a||b| instead makes this an
// angular test, which is what it always meant.
HOST_DEVICE inline bool outer_billiards_is_outside(const vec2* verts, int n, const vec2& p, int tangent) {
    const float SLACK = 1e-5f;   // radians, near enough
    const vec2 a = verts[tangent] - p;
    const float a_sq = dot(a, a);
    for (int i = 0; i < n; i++) {
        const vec2 b = verts[i] - p;
        const float turn = billiards_cross(a, b);
        // Squared, to keep a square root out of a per-pixel path.
        if (turn < 0.0f && turn * turn > SLACK * SLACK * a_sq * dot(b, b)) return false;
    }
    return true;
}

// The vertex the map pivots about, or -1 where it is undefined (inside the
// table, or outside the plane). Costs two sweeps; an orbit already known to be
// outside can call outer_billiards_tangent_vertex directly and skip the second.
HOST_DEVICE inline int outer_billiards_pivot(const vec2* verts, int n, const vec2& p, float curvature) {
    if (n < 2 || !curved_in_plane(p, curvature)) return -1;
    const int best = outer_billiards_tangent_vertex(verts, n, p);
    return outer_billiards_is_outside(verts, n, p, best) ? best : -1;
}

// One hop: reflect p through the pivot, 2*pivot - p. In a curved plane a
// "reflection through a point" is a half turn about it instead, which needs
// the general curved formula below - but it still reduces to the plain
// midpoint reflection whenever curvature is exactly 0, so that is its own
// branch rather than something to trust the algebra for.
HOST_DEVICE inline vec2 outer_billiards_reflect(const vec2& pivot, const vec2& p, float curvature) {
    if (curvature == 0.0f) return pivot * 2.0f - p;

    const float pivot_norm = curved_norm(pivot, curvature);
    if (pivot_norm <= 1e-9f) return p;   // the pivot is not in the plane; nothing sensible to do
    const float scale = 2.0f * (curvature * dot(p, pivot) + 1.0f) / pivot_norm;
    const float denom = scale - 1.0f;
    // Only reachable outside the plane, where an isometry has nowhere to send p.
    if (denom > -1e-9f && denom < 1e-9f) return p;
    return (pivot * scale - p) / denom;
}

// A full hop from a point already known to be outside.
HOST_DEVICE inline vec2 outer_billiards_hop(const vec2* verts, int n, const vec2& p, float curvature) {
    return outer_billiards_reflect(verts[outer_billiards_tangent_vertex(verts, n, p)], p, curvature);
}

// PART of a hop: rotate p by `angle` about the pivot, so at angle = pi this is
// outer_billiards_reflect exactly. Answers "where has this point truly reached
// partway through a hop" - which is what recoloring a pixel by its orbit's
// progress needs (see outer_billiards_flow.cu). It is NOT how a drawn orbit's
// leading segment is extended: that is a deliberate straight lerp toward the
// hop's endpoint instead (see OuterBilliardsScene::draw_orbit), so the segment
// reads as a line being traced out. Using this rotation there instead would
// swing the whole in-progress segment around the pivot, which is not that -
// and sliding along the segment from p to T(p) would be worse still, since the
// midpoint of p and 2*pivot-p is the pivot itself for EVERY p, so a lerp
// there collapses an entire wedge onto its own pivot at the halfway mark.
HOST_DEVICE inline vec2 outer_billiards_turn(const vec2& pivot, const vec2& p,
                                             float curvature, float angle) {
    const float ca = cosf(angle), sa = sinf(angle);

    if (curvature == 0.0f) {
        const vec2 offset = p - pivot;
        return pivot + vec2(ca * offset.x - sa * offset.y, sa * offset.x + ca * offset.y);
    }

    // Same idea, in the curved plane: lift p to (p,1), split it into the part
    // along the pivot's own lift (pivot,1) and a remainder, and rotate just the
    // remainder by `angle` within the plane tangent to the hyperboloid there
    // (found as the cross product of the two lifts, which is orthogonal to both
    // under the curved bilinear form). Reduces to the flat case above at K = 0.
    const float pivot_norm = curved_norm(pivot, curvature);
    if (pivot_norm <= 1e-9f) return p;   // the pivot is not in the plane; nothing to turn about

    const float along_pivot = (curvature * dot(p, pivot) + 1.0f) / pivot_norm;
    const vec3 pivot_lift(pivot.x, pivot.y, 1.0f);
    const vec3 remainder(p.x - along_pivot * pivot.x, p.y - along_pivot * pivot.y, 1.0f - along_pivot);

    const vec3 turn_axis = cross(pivot_lift, remainder);
    const float inv_len = 1.0f / sqrtf(pivot_norm);
    const vec3 quarter_turn(turn_axis.x * inv_len, turn_axis.y * inv_len, curvature * turn_axis.z * inv_len);

    const vec3 turned(along_pivot * pivot.x + ca * remainder.x + sa * quarter_turn.x,
                      along_pivot * pivot.y + ca * remainder.y + sa * quarter_turn.y,
                      along_pivot           + ca * remainder.z + sa * quarter_turn.z);

    // Only reachable at the ideal boundary, where there is nowhere on screen to put it.
    if (turned.z > -1e-9f && turned.z < 1e-9f) return p;
    return vec2(turned.x / turned.z, turned.y / turned.z);
}

// ---------------------------------------------------------------------------
// The singular set
// ---------------------------------------------------------------------------

// Twice the signed area, positive when the vertices are wound counterclockwise.
HOST_DEVICE inline float billiards_double_signed_area(const vec2* verts, int n) {
    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        const vec2 a = verts[i];
        const vec2 b = verts[(i + 1) % n];
        total += a.x * b.y - b.x * a.y;
    }
    return total;
}

// Where the i'th singular ray starts, and which way it heads: away from the
// table, along the side it extends. NOT normalized. Vertices must be
// counterclockwise.
HOST_DEVICE inline vec2 outer_billiards_ray_origin(const vec2* verts, int n, int i) { return verts[i]; }
HOST_DEVICE inline vec2 outer_billiards_ray_step(const vec2* verts, int n, int i) {
    return verts[i] - verts[(i + 1) % n];
}

// A side of zero length has no direction to extend and contributes no ray -
// the ordinary way a table loses a corner when two vertices are animated onto
// each other.
HOST_DEVICE inline bool outer_billiards_side_is_degenerate(const vec2* verts, int n, int i) {
    const vec2 step = verts[i] - verts[(i + 1) % n];
    return dot(step, step) < 1e-18f;
}

// One singular ray, boiled down to what the distance test needs. Built once
// per frame on the host, then read many times per pixel.
struct SingularRay {
    // The covector of the ray's full geodesic, scaled so that
    // |line . (q,1)| / sqrt(N(q)) is the curved sine of the distance to it.
    vec3 line;
    // The covector of the perpendicular geodesic at the ray's origin. Positive
    // means the foot of the perpendicular lands ON the ray, not behind its start.
    vec3 cap;
    vec2 origin;
};

HOST_DEVICE inline SingularRay outer_billiards_build_ray(const vec2* verts, int n, int i, float curvature) {
    SingularRay ray;
    const vec2 start = outer_billiards_ray_origin(verts, n, i);   // where the ray starts
    const vec2 far_end = verts[(i + 1) % n];                      // the far end of the side it extends
    ray.origin = start;

    // The line through start and far_end, homogeneously: line . (q,1) = 0.
    const vec3 line(start.y - far_end.y, far_end.x - start.x, start.x * far_end.y - start.y * far_end.x);

    // Normalize against the curved form, not the Euclidean one, so the
    // covector reports curved distance directly (a plain point-to-line
    // formula at K = 0).
    float scale = line.x * line.x + line.y * line.y + curvature * line.z * line.z;
    scale = (scale > 1e-20f) ? 1.0f / sqrtf(scale) : 0.0f;
    ray.line = vec3(line.x * scale, line.y * scale, line.z * scale);

    // The perpendicular at start: the geodesic whose normal is the ray's own
    // tangent there. At K=0 this reduces to (start-far_end).(q-start), i.e.
    // "has q got past the start yet".
    const vec3 curved_normal(line.x, line.y, curvature * line.z);
    ray.cap = vec3(start.y * curved_normal.z - curved_normal.y,
                   curved_normal.x - start.x * curved_normal.z,
                   start.x * curved_normal.y - start.y * curved_normal.x);
    return ray;
}

// Geodesic distance from q to the ray. The cap decides whether the nearest
// point is the perpendicular foot or the ray's own start.
HOST_DEVICE inline float outer_billiards_ray_distance(const SingularRay& ray, const vec2& q,
                                                      float norm_q, float curvature) {
    if (ray.cap.x * q.x + ray.cap.y * q.y + ray.cap.z >= 0.0f) {
        const float height = ray.line.x * q.x + ray.line.y * q.y + ray.line.z;
        return curved_arcsinh(fabsf(height) / sqrtf(norm_q), curvature);
    }
    return curved_distance(q, ray.origin, curvature);
}

// `count` is however many rays the table actually has - its number of
// non-degenerate sides, not necessarily its number of vertices.
HOST_DEVICE inline float outer_billiards_singular_distance(const SingularRay* rays, int count, const vec2& q,
                                                           float curvature) {
    const float norm_q = curved_norm(q, curvature);
    if (norm_q <= 0.0f) return 1e30f;
    float best = 1e30f;
    for (int i = 0; i < count; i++) {
        const float d = outer_billiards_ray_distance(rays[i], q, norm_q, curvature);
        if (d < best) best = d;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Everything the singularity kernel needs, in one parameter block. Filled in
// by OuterBilliardsScene::draw() and passed to the launcher by value. No
// default member initializers: this crosses into a kernel parameter, so it
// stays a plain aggregate and the scene sets every field.
// ---------------------------------------------------------------------------
struct SingularityGraphParams {
    vec2        verts[MAX_BILLIARD_VERTICES];   // counterclockwise, for the tangency search
    SingularRay rays[MAX_BILLIARD_VERTICES];    // the same table, precooked for the distance test
    int         n;           // vertices
    int         ray_count;   // non-degenerate sides, which is not always the same thing
    float       curvature;   // 0 Euclidean, negative hyperbolic
    int         poincare_view;   // 1 draws the Poincare-disk picture instead of Klein's

    vec2  lx_ty, rx_by;      // the visible window, world units
    float world_per_pixel;   // one pixel's width out there - sets every line's real thickness

    // --- the web: the singularity graph itself ---------------------------
    float    web_opacity;
    float    depth;           // how many preimages. REAL: layer k weighs clamp(depth-k, 0, 1),
                              // so ramping this grows the web smoothly instead of layer by layer.
    float    line_width;      // pixels
    float    glow;            // 0..1 peak of a soft halo four line-widths wide
    float    rainbow;         // blend line_color toward a hue that advances with depth
    float    rainbow_period;  // layers per full trip around the color wheel
    uint32_t line_color;

    // --- the islands: the gaps the graph leaves behind --------------------
    // An island is filled because the graph does not reach it, NOT because its
    // orbit is periodic - in a curved plane the return map on an island is a
    // small ROTATION rather than the identity, so nothing ever exactly closes.
    // Asking for an exact return instead would carve a tolerance-sized disk out
    // of each island rather than the island itself. So the fill is the
    // complement of the web (which bounds it exactly), and the orbit is only
    // asked which hop brought it back CLOSEST, to pick the color.
    float    island_opacity;
    int      max_period;      // how far to look for that closest return
    // Layers of the graph the fill is bounded by - deliberately more than are
    // DRAWN, so the gaps between hairlines in the chaotic region (gaps only
    // because the drawing stopped) close up instead of filling with color,
    // while a true island - open at any depth - still reads as one.
    int      island_depth;
    // Octaves per trip around the color wheel, applied to the log of the
    // return time so neighbouring return times get neighbouring colors.
    float    period_octaves;
};

// ---------------------------------------------------------------------------
// A different question about the same map: not WHERE THE MAP BREAKS but WHERE
// EVERY POINT GOES. Paint the plane with a color wheel (hue from direction,
// white at the middle, fading to black far out), then recolor each pixel with
// the color of wherever its orbit has reached after n hops - nothing moves,
// only the coloring does.
//
// On a periodic island the n'th iterate is a rigid motion, so the coloring is
// the original wheel merely turned - a coherent patch that stays coherent
// forever. In a chaotic region neighbouring points separate, so the coloring
// dissolves into noise. If shade_by_distance is on, an escaping orbit fades to
// black; off, the wheel is hue alone at full strength, so only ORDER and CHAOS
// read.
// ---------------------------------------------------------------------------
struct FlowFieldParams {
    vec2  verts[MAX_BILLIARD_VERTICES];
    int   n;
    float curvature;
    int   poincare_view;     // 1 draws the Poincare-disk picture instead of Klein's

    vec2  lx_ty, rx_by;      // the visible window, world units

    float iterations;        // real; the fractional part turns partway through the next hop

    vec2  center;            // what the color wheel is centered on
    float scale;             // the radius at which the hue reaches full strength -
                             // meaningless when shade_by_distance is off
    int   shade_by_distance; // 1 whitens near center and fades to black far out
                             // (see scale); 0 leaves every pixel at full hue and
                             // full strength, so only direction is ever read
    float opacity;
};

SHARED_FILE_SUFFIX
