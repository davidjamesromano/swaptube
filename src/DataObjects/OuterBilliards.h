#pragma once

#include <vector>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/OuterBilliardsShared.h"

// ---------------------------------------------------------------------------
// The OUTER billiards map (also called the dual billiards map) about a convex
// table.
//
// Inner billiards sends a ball bouncing around the INSIDE of a table. Outer
// billiards is its mirror image: the point lives outside, and instead of
// bouncing off the boundary it pivots about it. From a point p outside the
// table there are two tangent lines; take the one that leaves the whole table on
// your left as you look from p toward the touching vertex v, and reflect p
// through that vertex:
//
//     T(p) = 2v - p
//
// The hop is a straight segment whose MIDPOINT is the vertex it turned about, so
// an orbit drawn as a polyline shows its own construction. Because reflection
// preserves the distance to v, the image is exactly as far out as the source was,
// and the point circles the table forever without ever landing on it.
//
// Whether those orbits stay bounded is the interesting question. A square (or
// any lattice polygon) makes every orbit periodic - closed necklaces of hops. A
// regular pentagon does not: its orbits are quasiperiodic and fill an intricate
// self-similar web. Whether some polygon has an orbit that escapes to infinity
// was open for decades.
//
// The map is not defined everywhere. Which vertex the tangent line touches
// changes as p moves, and it changes exactly along n rays - the sides of the
// table extended past their endpoints. singular_rays() hands those back, and
// the set of points that eventually LAND on one of them is the singularity
// graph, the fractal OuterBilliardsScene renders on the GPU.
//
// CURVATURE. Set `curvature` negative and the same table lives in the hyperbolic
// plane instead, drawn in the Beltrami-Klein model - so the coordinates here
// still mean what they meant, geodesics are still straight, and only the
// reflection and the distances change. Zero is the Euclidean plane, exactly.
// See Host_Device_Shared/OuterBilliardsShared.h for how that is arranged.
//
// This class is plain host-side math with no rendering and no state hookup; the
// arithmetic itself lives in that shared header so that the GPU renderer agrees
// with it hop for hop. OuterBilliardsScene rebuilds one of these from its
// animated vertices every frame and asks it for orbits.
// ---------------------------------------------------------------------------
class OuterBilliards {
public:
    std::vector<vec2> vertices;

    // 0 is the Euclidean plane. Negative is the hyperbolic plane of that
    // curvature, whose ideal boundary is the circle of radius 1/sqrt(-curvature).
    float curvature = 0.0f;

    OuterBilliards() {}
    explicit OuterBilliards(const std::vector<vec2>& verts, float curvature = 0.0f)
        : vertices(verts), curvature(curvature) {}

    // Vertices counterclockwise on a circle of the given radius. `rotation` is
    // where the first vertex sits, in radians. Rotations about the origin are
    // the same map in Klein coordinates as in the plane, so a polygon built this
    // way about the origin is regular in whatever curvature it is placed in.
    static std::vector<vec2> regular_polygon(int sides, float radius = 1.0f,
                                             float rotation = 0.0f, const vec2& center = vec2(0, 0));

    int size() const { return (int)vertices.size(); }

    // The radius of the ideal boundary, or 0 in the Euclidean plane, where there
    // is none.
    float horizon() const;

    // Index of the vertex the tangent line from p touches, or -1 where the map is
    // undefined (p inside the table, outside the plane, or fewer than two
    // vertices).
    int pivot_index(const vec2& p) const;

    // One hop. Returns p unchanged where the map is undefined; pass pivot_out to
    // learn which vertex it turned about (-1 if it did not move).
    vec2 step(const vec2& p, int* pivot_out = nullptr) const;

    // `steps` hops. The returned path starts with `start` itself, so it holds
    // steps+1 points - unless the orbit ran into a point where the map is
    // undefined, in which case it stops early. `pivots_out`, if given, receives
    // one vertex index per hop taken.
    std::vector<vec2> orbit(const vec2& start, int steps, std::vector<int>* pivots_out = nullptr) const;

    // How many hops until the orbit from `start` closes, or 0 if it has not
    // closed within `max_period`. `tolerance` is how near a return counts as a
    // return - the map is a piecewise isometry, so an orbit that closes closes
    // exactly, and the tolerance is only there to absorb rounding.
    int period(const vec2& start, int max_period, float tolerance = 1e-4f) const;

    // Is p inside the table (where the map has nothing to say)? Answered by the
    // tangent search itself, so it needs no winding or convexity assumption.
    bool is_inside(const vec2& p) const { return pivot_index(p) < 0; }

    // --- the singular set ------------------------------------------------
    // The same vertices, guaranteed wound counterclockwise. The rays below - and
    // the GPU renderer - depend on the winding; nothing above does.
    std::vector<vec2> counterclockwise() const;

    // The n rays along which the map is undefined, as (origin, unit direction,
    // length) triples: ray i leaves vertex i heading away from the table along
    // the side it extends. Geodesic rays are straight in these coordinates, so
    // these are line segments - `length` is however far the ray runs before it
    // leaves `reach`, or reaches the ideal boundary in a curved plane.
    struct Ray { vec2 origin, direction; float length; };
    std::vector<Ray> singular_rays(float reach) const;

    // The same rays as the renderer sees them, ready for
    // outer_billiards_singular_distance. One per vertex, counterclockwise.
    std::vector<SingularRay> singular_ray_data() const;

    // Distance from p to the nearest singular ray, in the plane's own metric.
    float singular_distance(const vec2& p) const;

    vec2 centroid() const;

    // Distance from the centroid to the farthest vertex - a handy scale for
    // framing a shot or for placing a starting point safely outside. Measured on
    // screen, not in the curved metric, which is what framing wants.
    float circumradius() const;
};
