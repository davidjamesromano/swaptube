#pragma once

#include <cstdint>
#include "vec.h"
#include "shared_precompiler_directives.h"

// The two primitives the 2D vector renderer in src/CUDA/billiards.cu understands.
// Both live in PIXEL space: the scene converts its world coordinates through
// CoordinateScene::point_to_pixel before filling these in, so the GPU never needs
// to know about the coordinate window.

SHARED_FILE_PREFIX

struct Segment2D {
    vec2 a, b;
    uint32_t color;
    float opacity;
    float thickness;   // full width, in pixels
    HOST_DEVICE Segment2D() {}
    HOST_DEVICE Segment2D(const vec2& start, const vec2& end, uint32_t clr, float opa, float thick)
        : a(start), b(end), color(clr), opacity(opa), thickness(thick) {}
};

struct Dot2D {
    vec2 center;
    float radius;      // in pixels
    uint32_t color;
    float opacity;
    HOST_DEVICE Dot2D() {}
    HOST_DEVICE Dot2D(const vec2& c, float r, uint32_t clr, float opa)
        : center(c), radius(r), color(clr), opacity(opa) {}
};

SHARED_FILE_SUFFIX
