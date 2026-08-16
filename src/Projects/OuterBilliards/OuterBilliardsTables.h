#pragma once

#include "../../Host_Device_Shared/vec.h"
#include <vector>
#include <cmath>

// Shared between OrderAndChaos.cpp and OrderAndChaosHyperbolic.cpp, which run
// the same four tables through the flat and curved planes respectively.

// The kite K(a) Schwartz works with: vertices (-1,0), (0,1), (a,0), (0,-1).
// Any irrational a in (0,1) has unbounded orbits in the flat plane.
static const float KITE_A = 0.41421356f;   // sqrt(2) - 1

static std::vector<vec2> kite(float a) {
    return {vec2(-1, 0), vec2(0, -1), vec2(a, 0), vec2(0, 1)};
}

// A quadrilateral with nothing going for it: no particular angles or radii, so
// it is neither a lattice polygon nor a regular one nor an affine image of either.
static std::vector<vec2> generic_quad() {
    const float angle[4]  = {0.00f, 1.51f, 2.97f, 4.44f};
    const float radius[4] = {1.00f, 0.93f, 1.07f, 0.97f};
    std::vector<vec2> v;
    for (int i = 0; i < 4; i++) v.push_back(vec2(radius[i] * cosf(angle[i]), radius[i] * sinf(angle[i])));
    return v;
}
