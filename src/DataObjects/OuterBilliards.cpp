#include "OuterBilliards.h"
#include <algorithm>
#include <cmath>

std::vector<vec2> OuterBilliards::regular_polygon(int sides, float radius, float rotation, const vec2& center) {
    std::vector<vec2> verts;
    if (sides < 3) return verts;
    const float step = 6.283185307179586f / (float)sides;
    for (int i = 0; i < sides; i++) {
        const float angle = rotation + step * (float)i;
        verts.push_back(center + vec2(std::cos(angle), std::sin(angle)) * radius);
    }
    return verts;
}

float OuterBilliards::horizon() const {
    return curvature < 0.0f ? 1.0f / std::sqrt(-curvature) : 0.0f;
}

int OuterBilliards::pivot_index(const vec2& p) const {
    if (vertices.empty()) return -1;
    return outer_billiards_pivot(vertices.data(), size(), p, curvature);
}

vec2 OuterBilliards::step(const vec2& p, int* pivot_out) const {
    const int pivot = pivot_index(p);
    if (pivot_out) *pivot_out = pivot;
    if (pivot < 0) return p;
    return outer_billiards_reflect(vertices[pivot], p, curvature);
}

std::vector<vec2> OuterBilliards::orbit(const vec2& start, int steps, std::vector<int>* pivots_out) const {
    std::vector<vec2> path;
    if (pivots_out) pivots_out->clear();
    path.push_back(start);
    if (steps <= 0) return path;

    path.reserve(steps + 1);
    vec2 p = start;
    for (int i = 0; i < steps; i++) {
        int pivot = -1;
        const vec2 next = step(p, &pivot);
        if (pivot < 0) break;   // the orbit reached a point the map cannot continue from
        if (pivots_out) pivots_out->push_back(pivot);
        path.push_back(next);
        p = next;
    }
    return path;
}

std::vector<vec2> OuterBilliards::counterclockwise() const {
    std::vector<vec2> wound = vertices;
    if (wound.size() >= 3 && billiards_double_signed_area(wound.data(), (int)wound.size()) < 0.0f) {
        std::reverse(wound.begin(), wound.end());
    }
    return wound;
}

std::vector<SingularRay> OuterBilliards::singular_ray_data() const {
    std::vector<SingularRay> rays;
    const std::vector<vec2> wound = counterclockwise();
    const int n = (int)wound.size();
    if (n < 3) return rays;

    rays.reserve(n);
    for (int i = 0; i < n; i++) {
        if (outer_billiards_side_is_degenerate(wound.data(), n, i)) continue;
        rays.push_back(outer_billiards_build_ray(wound.data(), n, i, curvature));
    }
    return rays;
}

std::vector<OuterBilliards::Ray> OuterBilliards::singular_rays(float reach) const {
    std::vector<Ray> rays;
    const std::vector<vec2> wound = counterclockwise();
    const int n = (int)wound.size();
    if (n < 3) return rays;

    // In a curved plane the ray runs out of plane before it runs out of `reach`,
    // and past the ideal boundary there is nothing to draw.
    const float edge = horizon();

    rays.reserve(n);
    for (int i = 0; i < n; i++) {
        const vec2 origin = outer_billiards_ray_origin(wound.data(), n, i);
        const vec2 step = outer_billiards_ray_step(wound.data(), n, i);
        const float len = length(step);
        if (len < 1e-9f) continue;   // coincident vertices carry no ray
        const vec2 direction = step / len;

        float span = reach;
        if (edge > 0.0f) {
            // Where the ray crosses |q| = edge. The origin is a vertex of a table
            // inside the plane, so there is always exactly one crossing ahead.
            const float b = dot(origin, direction);
            const float c = dot(origin, origin) - edge * edge;
            const float discriminant = b * b - c;
            span = (discriminant > 0.0f) ? std::fmax(-b + std::sqrt(discriminant), 0.0f) : 0.0f;
            span = std::fmin(span, reach);
        }
        rays.push_back({origin, direction, span});
    }
    return rays;
}

vec2 OuterBilliards::centroid() const {
    if (vertices.empty()) return vec2(0, 0);
    vec2 sum(0, 0);
    for (const vec2& v : vertices) sum += v;
    return sum / (float)vertices.size();
}

float OuterBilliards::circumradius() const {
    const vec2 center = centroid();
    float radius = 0.0f;
    for (const vec2& v : vertices) radius = std::fmax(radius, length(v - center));
    return radius;
}
