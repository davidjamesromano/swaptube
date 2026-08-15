#pragma once

#include <cstdint>
#include "vec.h"
#include "shared_precompiler_directives.h"

// Below this, a layer is visually indistinguishable from off - shared so the
// CPU scene and both GPU kernels agree on when a layer is worth paying for.
#define BILLIARDS_MIN_OPACITY 0.004f

// ---------------------------------------------------------------------------
// The outer billiards map and its singular set, written once so that the CPU -
// which traces individual orbits as polylines - and the GPU - which renders the
// singularity graph one pixel at a time - can never disagree about where a hop
// lands. DataObjects/OuterBilliards.h wraps the host side of this; the kernel in
// src/CUDA/outer_billiards_singularity.cu calls it directly.
//
// THE MAP. From a point p outside a convex table there are two tangent lines.
// Take the one that leaves the table on your left as you look from p toward the
// vertex v it touches, and reflect p through that vertex: T(p) = 2v - p. Seen
// from p, that tangency vertex is simply the CLOCKWISE-MOST of them all, which
// is what outer_billiards_tangent_vertex finds in one sweep.
//
// THE SINGULAR SET. The tangency vertex changes as p moves, and it changes
// exactly when two vertices line up with p - so the exterior is cut into n
// wedges, one per vertex, and T is a different reflection on each. The cuts are
// the n rays
//
//     R_i = { v_i + t (v_i - v_{i+1}) : t >= 0 }
//
// for vertices wound COUNTERCLOCKWISE: each side of the table extended past one
// of its endpoints. On those rays the map has no answer, and the singularity
// graph is what you get by asking which points eventually land on one:
//
//     S_D = union over k = 0..D of T^-k(R)
//
// Nothing here has to invert anything to draw that. T is an isometry on each
// wedge, so distance survives it, and
//
//     dist(p, T^-k(R)) = dist(T^k(p), R)
//
// which turns the whole fractal into a distance field one forward orbit deep -
// no line clipping, no subdivision, and antialiased at any zoom. That identity
// is the reason the renderer is a per-pixel kernel and not a pile of segments.
//
// ---------------------------------------------------------------------------
// CURVATURE
// ---------------------------------------------------------------------------
// Every function below takes a `curvature` K <= 0. K = 0 is the Euclidean plane.
// K < 0 is the hyperbolic plane of curvature K, drawn in the BELTRAMI-KLEIN
// model: the whole plane compressed into a disk of radius R = 1/sqrt(-K), in
// which geodesics are straight chords. Klein is the right model here precisely
// because straightness survives it - the table's sides, the singular rays, and
// every hop of an orbit are line segments in these coordinates exactly as they
// were in the Euclidean picture, so the vector overlay needs no curved-line
// machinery and the tangency search below is untouched (which vertex is
// clockwise-most from p is a question about incidence, and projective models
// preserve incidence).
//
// What DOES change is the reflection and the metric. Lift a Klein point q to
// (q, 1) in R^3 and use the bilinear form
//
//     <u,v> = K (u_x v_x + u_y v_y) + u_z v_z
//
// which is Minkowski for K < 0 and merely degenerate - never singular - at K = 0.
// Writing N(q) = <(q,1),(q,1)> = 1 + K|q|^2 for the normalizer, rotation by pi
// about v (which is what "reflect through the vertex" means in any curvature)
// works out to
//
//     T_v(q) = (A v - q) / (A - 1),    A = 2 (K q.v + 1) / (K |v|^2 + 1)
//
// and at K = 0 that is A = 2 and T_v(q) = 2v - q on the nose. Every formula in
// this file is arranged that way: one expression, no branch on the curvature,
// Euclidean recovered exactly in the limit. That is what lets a scene ANIMATE
// the curvature and watch the plane curl into a disk.
// ---------------------------------------------------------------------------

SHARED_FILE_PREFIX

// The table rides to the GPU inside the kernel's parameter block rather than in
// a buffer of its own, which costs a copy but saves a cudaMalloc every frame.
// Kernel parameters have a few kilobytes to spend and this is 1.3KB of it.
#define MAX_BILLIARD_VERTICES 32

// ---------------------------------------------------------------------------
// Curvature primitives
// ---------------------------------------------------------------------------

// N(q) = 1 + K|q|^2. One inside the Euclidean plane always; inside the hyperbolic
// plane it runs from 1 at the center down to 0 at the ideal boundary, and turns
// negative past it, where there is no geometry to speak of.
HOST_DEVICE inline float curved_norm(const vec2& q, float curvature) {
    return 1.0f + curvature * dot(q, q);
}

HOST_DEVICE inline bool curved_in_plane(const vec2& q, float curvature) {
    return curved_norm(q, curvature) > 1e-7f;
}

// asinh(x sqrt(-K)) / sqrt(-K), the inverse of the sine of the curved plane -
// and just x when K is zero, which is why every distance below is written
// through it.
HOST_DEVICE inline float curved_arcsinh(float x, float curvature) {
    const float k = -curvature;
    if (k < 1e-9f) return x;
    const float bend = sqrtf(k);
    return asinhf(bend * x) / bend;
}

// How much shorter a hyperbolic length looks on screen, here. The Klein metric
// is anisotropic - it stretches by 1/N along the radius and by 1/sqrt(N) across
// it - so no single number is right for every direction. This is their geometric
// mean, which is exact in the Euclidean limit and errs symmetrically elsewhere;
// it only ever sets how thick a line is drawn, never where it goes.
HOST_DEVICE inline float curved_screen_scale(const vec2& q, float curvature) {
    const float n = curved_norm(q, curvature);
    if (n <= 0.0f) return 0.0f;
    return powf(n, 0.75f);
}

// (cosh(d sqrt(-K)) - 1) / (-2K), which is |a-b|^2 / 4 when K is zero. This is a
// strictly increasing function of the geodesic distance and costs one square
// root, so anything that only needs to COMPARE distances - which hop came back
// nearest, say - should ask for this and skip the transcendental below.
//
// Written this way, rather than through arccosh, because cosh(d) - 1 = 2 sinh^2(d/2)
// keeps its digits when the two points are close, and the direct form loses every
// one of them - which, for a fractal made of near-misses, is most of the time.
// `a_norm` is curved_norm(a), passed in because callers sweeping b against a
// fixed a already have it.
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
// KLEIN <-> POINCARE
// ---------------------------------------------------------------------------
// Both models compress the same hyperbolic plane into the same disk of radius
// `horizon` - the ideal boundary is one shared circle - but Klein draws
// geodesics as straight chords while Poincare draws them as arcs bowing toward
// the center. Nothing above this needs to know which one is on screen: the
// map, the metric and every distance are all defined in KLEIN coordinates, so
// converting is purely a question of where a given abstract point gets drawn,
// never of recomputing what happens to it. `horizon <= 0` (the flat plane, or
// no plane at all) is the identity in both directions - there is no disk to
// normalize against.
//
// Both are the standard unit-disk projective/conformal formulas, applied after
// scaling into the unit disk and scaled back out afterward, and are exact
// inverses of one another.
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

// The vertex the tangent line from p touches. Seen from a point OUTSIDE the
// table every vertex lies inside a wedge narrower than half a turn, so "is
// counterclockwise of" is a genuine ordering there and one sweep finds the
// clockwise-most vertex. Winding-agnostic, and curvature-agnostic: it never
// looks at two vertices' adjacency or at any distance, only at their directions
// from p, and the Klein model keeps those honest.
//
// Exactly collinear means p sits on the line through an edge, where the map is
// discontinuous anyway. Take the farther vertex, so the choice is at least
// consistent - and so a vertex parked on top of another, or in the middle of an
// edge, never wins.
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

// That sweep assumed p was outside, so check the assumption: every vertex has to
// lie counterclockwise of the winner. If one does not, the vertices wrap more
// than half a turn around p, which means p is inside the table and there is no
// tangent line to be had.
//
// The slack has to be RELATIVE. A cross product of two vectors from p to the
// table grows like the square of how far away p is, and so does the rounding
// error in it - roughly |a||b| * 1e-7 in float. An absolute threshold that is
// generous next to the table is therefore swamped out in the far field, where
// nearly-aligned vertices produce a cross product that is small, noisy, and
// occasionally negative enough to trip. Points that are plainly outside then
// report themselves inside, at random, one pixel in three: the far corners of a
// wide shot come out unshaded and speckled, and no amount of care further down
// helps, because the orbit is abandoned before it starts. Comparing against
// |a||b| instead makes this an angular test, which is what it always meant.
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

// The vertex the map pivots about, or -1 where it is undefined - inside the
// table, or outside the plane itself. Costs two sweeps instead of one; an orbit
// already known to be outside can skip the second by calling
// outer_billiards_tangent_vertex directly, which is what the fractal kernel does
// after its first step.
HOST_DEVICE inline int outer_billiards_pivot(const vec2* verts, int n, const vec2& p, float curvature) {
    if (n < 2 || !curved_in_plane(p, curvature)) return -1;
    const int best = outer_billiards_tangent_vertex(verts, n, p);
    return outer_billiards_is_outside(verts, n, p, best) ? best : -1;
}

// One hop: rotate p by half a turn about the vertex it turned about. In the
// Euclidean plane that is the midpoint reflection 2v - p; the general form is
// the same statement in a curved plane, and reduces to it at K = 0.
HOST_DEVICE inline vec2 outer_billiards_reflect(const vec2& pivot, const vec2& p, float curvature) {
    const float nv = curved_norm(pivot, curvature);
    if (nv <= 1e-9f) return p;   // the pivot is not in the plane; nothing sensible to do
    const float a = 2.0f * (curvature * dot(p, pivot) + 1.0f) / nv;
    const float denom = a - 1.0f;
    // Only reachable outside the plane, where an isometry has nowhere to send p.
    if (denom > -1e-9f && denom < 1e-9f) return p;
    return (pivot * a - p) / denom;
}

// A full hop from a point already known to be outside: find the vertex it
// pivots about, then reflect through it. The CPU and GPU call sites that
// don't need the pivot index separately want this, not the two calls spelled
// out - see the header comment above for why the two can never be allowed to
// disagree.
HOST_DEVICE inline vec2 outer_billiards_hop(const vec2* verts, int n, const vec2& p, float curvature) {
    return outer_billiards_reflect(verts[outer_billiards_tangent_vertex(verts, n, p)], p, curvature);
}

// PART of a hop. T is a rotation by half a turn about the pivot, so the honest
// way to get halfway there is to turn half as far: this is rotation by `angle`
// about that same vertex, and at angle = pi it is outer_billiards_reflect above,
// exactly.
//
// Sliding along the segment from p to T(p) instead would be the obvious thing to
// write, and it is a disaster. The midpoint of p and 2v - p is v - for EVERY p.
// So at half a hop an entire wedge collapses onto its own pivot, the picture
// flattens to one flat color per wedge, and it does that TWICE per hop. Anything
// animating a fraction of a hop wants this function, not a lerp.
//
// Lift q to Q = (q,1) and use the same bilinear form as the rest of this file,
// <u,w> = K(u_x w_x + u_y w_y) + u_z w_z. Split Q into its component along the
// pivot's lift V and the remainder, and turn the remainder:
//
//     Q(t) = c V + cos t . X + sin t . J(X),   X = Q - c V,   c = <Q,V> / <V,V>
//
// where J is the quarter turn in the plane tangent to the hyperboloid at V.
// Writing it as J(X) = D (V x X) / sqrt(N(v)), with D = diag(1,1,K) and x the
// ordinary cross product, gives J^2 = -1 on that tangent plane for every K - and
// at K = 0, where the form goes degenerate, D quietly deletes the third component
// and what is left is the plain 2D quarter turn (-d_y, d_x). K-uniform like
// everything else here, no branch.
HOST_DEVICE inline vec2 outer_billiards_turn(const vec2& pivot, const vec2& p,
                                             float curvature, float angle) {
    const float nv = curved_norm(pivot, curvature);
    if (nv <= 1e-9f) return p;   // the pivot is not in the plane; nothing to turn about

    const float c = (curvature * dot(p, pivot) + 1.0f) / nv;   // <Q,V> / <V,V>
    const vec3 V(pivot.x, pivot.y, 1.0f);
    const vec3 X(p.x - c * pivot.x, p.y - c * pivot.y, 1.0f - c);

    const vec3 cr = cross(V, X);
    const float inv = 1.0f / sqrtf(nv);
    const vec3 jx(cr.x * inv, cr.y * inv, curvature * cr.z * inv);

    const float ca = cosf(angle), sa = sinf(angle);
    const vec3 turned(c * pivot.x + ca * X.x + sa * jx.x,
                      c * pivot.y + ca * X.y + sa * jx.y,
                      c           + ca * X.z + sa * jx.z);

    // Only reachable at the ideal boundary, where the turn has taken the point
    // out of the plane and there is nowhere on screen to put it.
    if (turned.z > -1e-9f && turned.z < 1e-9f) return p;
    return vec2(turned.x / turned.z, turned.y / turned.z);
}

// ---------------------------------------------------------------------------
// The singular set
// ---------------------------------------------------------------------------

// Twice the signed area, positive when the vertices are wound counterclockwise.
// The singular rays care about the winding (the pivot search does not), so the
// caller settles it once with this before building anything below.
HOST_DEVICE inline float billiards_double_signed_area(const vec2* verts, int n) {
    float total = 0.0f;
    for (int i = 0; i < n; i++) {
        const vec2 a = verts[i];
        const vec2 b = verts[(i + 1) % n];
        total += a.x * b.y - b.x * a.y;
    }
    return total;
}

// Where the i'th singular ray starts, and which way it heads - away from the
// table, along the side it extends. NOT normalized, and unaffected by curvature:
// a geodesic ray is a straight ray in these coordinates. Vertices must be
// counterclockwise.
HOST_DEVICE inline vec2 outer_billiards_ray_origin(const vec2* verts, int n, int i) { return verts[i]; }
HOST_DEVICE inline vec2 outer_billiards_ray_step(const vec2* verts, int n, int i) {
    return verts[i] - verts[(i + 1) % n];
}

// A side of zero length has no direction to extend, so it contributes no ray.
// This is not a pathological case to guard against but the ordinary way a table
// loses a corner: vertices are animated one to one, so a pentagon becomes a
// triangle by sliding two of its corners onto their neighbors, and the two sides
// that collapse have to stop generating rays when they do.
HOST_DEVICE inline bool outer_billiards_side_is_degenerate(const vec2* verts, int n, int i) {
    const vec2 step = verts[i] - verts[(i + 1) % n];
    return dot(step, step) < 1e-18f;
}

// One singular ray, boiled down to what the distance test needs. Built once per
// frame on the host by outer_billiards_build_ray, then read a few hundred times
// per pixel - which is the whole reason it is precomputed.
struct SingularRay {
    // The covector of the ray's full geodesic, scaled so that
    // |line . (q,1)| / sqrt(N(q)) is the curved sine of the distance to it.
    vec3 line;
    // The covector of the perpendicular geodesic at the ray's origin. Positive
    // means the foot of the perpendicular lands ON the ray rather than behind
    // its start, where the nearest point is the origin instead.
    vec3 cap;
    vec2 origin;
};

HOST_DEVICE inline SingularRay outer_billiards_build_ray(const vec2* verts, int n, int i, float curvature) {
    SingularRay ray;
    const vec2 v = outer_billiards_ray_origin(verts, n, i);   // where the ray starts
    const vec2 u = verts[(i + 1) % n];                        // the far end of the side it extends
    ray.origin = v;

    // The line through v and u, homogeneously: m . (q,1) = 0.
    const vec3 m(v.y - u.y, u.x - v.x, v.x * u.y - v.y * u.x);

    // Normalize against the curved form rather than the Euclidean one, so the
    // covector reports curved distance directly. At K = 0 this is a plain divide
    // by the length of the normal, and the whole thing collapses to the familiar
    // point-to-line formula.
    float scale = m.x * m.x + m.y * m.y + curvature * m.z * m.z;
    scale = (scale > 1e-20f) ? 1.0f / sqrtf(scale) : 0.0f;
    ray.line = vec3(m.x * scale, m.y * scale, m.z * scale);

    // The perpendicular at v is the geodesic whose normal is the ray's own
    // tangent there; in the lift that is (v,1) x (m_x, m_y, K m_z). At K = 0 it
    // reduces to (v-u) . (q-v), which is exactly "has q got past the start yet".
    const vec3 w(m.x, m.y, curvature * m.z);
    ray.cap = vec3(v.y * w.z - w.y,
                   w.x - v.x * w.z,
                   v.x * w.y - v.y * w.x);
    return ray;
}

// Distance from q to the ray - geodesic distance, which is what the preimage
// identity at the top of this file needs. The cap decides whether the nearest
// point is the perpendicular foot or the ray's own start.
HOST_DEVICE inline float outer_billiards_ray_distance(const SingularRay& ray, const vec2& q,
                                                      float norm_q, float curvature) {
    if (ray.cap.x * q.x + ray.cap.y * q.y + ray.cap.z >= 0.0f) {
        const float height = ray.line.x * q.x + ray.line.y * q.y + ray.line.z;
        return curved_arcsinh(fabsf(height) / sqrtf(norm_q), curvature);
    }
    return curved_distance(q, ray.origin, curvature);
}

// `count` is however many rays the table actually has, which is its number of
// non-degenerate sides - not necessarily its number of vertices.
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
// Everything the singularity kernel needs, in one parameter block. Filled in by
// OuterBilliardsScene::draw() and passed to the launcher by value.
//
// No default member initializers: this crosses into a kernel parameter, so it
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
    float    fade;            // 1 dims the deepest layer all the way out
    float    rainbow;         // blend line_color toward a hue that advances with depth
    float    rainbow_period;  // layers per full trip around the color wheel
    uint32_t line_color;

    // --- the islands: the gaps the graph leaves behind ---------------------
    //
    // An island is filled because the graph does not reach it, NOT because its
    // orbit is periodic. Those are the same statement in the Euclidean plane and
    // very much not in a curved one: there the return map on an island is a small
    // ROTATION about an elliptic fixed point rather than the identity, so points
    // come back near themselves in proportion to how far out they sit and none of
    // them ever closes. Asking for an exact return therefore carved out a disk
    // whose radius was set by the tolerance - a dot at the middle of each island,
    // and a smeared band near the ideal boundary - instead of the island. So the
    // fill is now the complement of the web, which bounds it exactly, and the
    // orbit is only asked which hop brought it back CLOSEST, to pick the color.
    float    island_opacity;
    int      max_period;      // how far to look for that closest return
    // How many layers of the graph the fill is bounded by - deliberately more
    // than are DRAWN. The gaps between the hairlines of the chaotic region are
    // gaps only because the drawing stopped; deeper layers close them, while a
    // true island stays open however deep you look. Bounding the fill by the
    // visible web alone floods that region with color and loses the distinction
    // the islands exist to make.
    int      island_depth;
    // Octaves per full trip around the color wheel. The hue follows the log of
    // the return time, not the time itself, so that neighbouring values get
    // neighbouring colors: under a morph it drifts by ones and twos, and a linear
    // hue would strobe every time it did.
    float    period_octaves;
};

// ---------------------------------------------------------------------------
// A different question about the same map, and a different picture.
//
// The singularity graph above asks WHERE THE MAP BREAKS. This asks WHERE EVERY
// POINT GOES. Paint the plane with a color wheel - hue from the direction, white
// at the middle, fading to black far out - and then, instead of moving the
// points, recolor each one with the color of the place its orbit has reached
// after n hops. Nothing on screen moves; only the coloring does.
//
// What that shows is the difference between order and chaos, directly. On a
// periodic island the n'th iterate is a rigid motion, so the coloring there is
// the original wheel, merely turned: a coherent patch of color that stays
// coherent forever. In a chaotic region neighbouring points separate, so pixels
// that started alike end up far apart and the coloring dissolves into noise. And,
// if shade_by_distance is on, a point whose orbit runs away to infinity fades to
// black, because that is what the far end of the color wheel is - off, the wheel
// is hue alone, full strength everywhere, and only ORDER and CHAOS still read;
// there is no far end left to fade towards.
// ---------------------------------------------------------------------------
struct FlowFieldParams {
    vec2  verts[MAX_BILLIARD_VERTICES];
    int   n;
    float curvature;
    int   poincare_view;     // 1 draws the Poincare-disk picture instead of Klein's

    vec2  lx_ty, rx_by;      // the visible window, world units

    float iterations;        // real; `smooth` decides what the fraction means
    int   smooth;            // 1 slides along the hop, 0 holds each whole iterate
    int   samples;           // n x n per pixel; 1 leaves the noise raw

    vec2  center;            // what the color wheel is centered on
    float scale;             // the radius at which the hue reaches full strength -
                             // meaningless when shade_by_distance is off
    int   shade_by_distance; // 1 whitens near center and fades to black far out
                             // (see scale); 0 leaves every pixel at full hue and
                             // full strength, so only direction is ever read
    float opacity;
};

SHARED_FILE_SUFFIX
