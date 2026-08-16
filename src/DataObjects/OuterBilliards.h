#pragma once

#include <vector>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/OuterBilliardsShared.h"

// ---------------------------------------------------------------------------
// The OUTER (dual) billiards map about a convex table: from a point p outside
// the table there are two tangent lines; take the one that leaves the whole
// table on your left, and reflect p through the vertex v it touches:
//
//     T(p) = 2v - p
//
// The hop's MIDPOINT is the vertex it turned about, so an orbit drawn as a
// polyline shows its own construction, and the point circles the table
// forever without ever landing on it. A square (or any lattice polygon) makes
// every orbit periodic; a regular pentagon's orbits are quasiperiodic and fill
// a self-similar web. Whether some polygon has an orbit that escapes to
// infinity was open for decades.
//
// Which vertex the tangent line touches changes along n rays - the table's
// sides extended past their endpoints (singular_rays()) - and everything that
// eventually LANDS on one of them is the singularity graph, the fractal
// OuterBilliardsScene renders on the GPU.
//
// `curvature` negative puts the same table in the hyperbolic plane instead,
// drawn in the Beltrami-Klein model: geodesics stay straight, only the
// reflection and the distances change. See Host_Device_Shared/OuterBilliardsShared.h,
// which holds the actual arithmetic so the GPU renderer agrees with this class
// hop for hop. This class itself is plain host-side math with no rendering;
// OuterBilliardsScene rebuilds one from its animated vertices every frame.
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

    vec2 centroid() const;

    // Distance from the centroid to the farthest vertex - a handy scale for
    // framing a shot or for placing a starting point safely outside. Measured on
    // screen, not in the curved metric, which is what framing wants.
    float circumradius() const;
};
