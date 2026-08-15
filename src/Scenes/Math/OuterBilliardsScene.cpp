#include "OuterBilliardsScene.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// The 2D vector renderer, implemented in src/CUDA/billiards.cu. Declared here
// with plain, non-namespaced types; the linker connects it to the definition.
extern "C" void billiards_fill_polygon(uint32_t* d_pixels, const ivec2& wh,
                                       const vec2* h_verts, int n,
                                       uint32_t color, float opacity);
extern "C" void billiards_draw_segments(uint32_t* d_pixels, const ivec2& wh,
                                        const Segment2D* h_segments, int n);
extern "C" void billiards_draw_dots(uint32_t* d_pixels, const ivec2& wh,
                                    const Dot2D* h_dots, int n);

// The singularity graph, implemented in src/CUDA/outer_billiards_singularity.cu.
extern "C" void outer_billiards_singularity_render(uint32_t* d_pixels, const ivec2& wh,
                                                   const SingularityGraphParams& params);

// Where every point goes, as a color, from src/CUDA/outer_billiards_flow.cu.
extern "C" void outer_billiards_flow_render(uint32_t* d_pixels, const ivec2& wh,
                                            const FlowFieldParams& params);

// Paints the whole panel one color, from src/CUDA/pixels_manip.cu.
extern "C" void cuda_fill_pixels(uint32_t* d_pixels, const ivec2& wh, uint32_t color);

// Below this an opacity is not worth a copy to the GPU, and below this a radius
// or a thickness is not worth drawing.
static const float MIN_OPACITY = BILLIARDS_MIN_OPACITY;
static const float MIN_SIZE    = 0.35f;

// A hop's midpoint marker and the dot an orbit sets off from are drawn bigger
// than an ordinary orbit point, so they read as distinct.
static const float EMPHASIS = 1.7f;

// How finely push_geodesic() subdivides a Klein-straight segment before bowing
// it into a Poincare arc. Fine enough that the facets do not show at the zooms
// this scene invites, and cheap enough not to matter - see draw_horizon()'s
// SIDES for the same tradeoff made for the boundary circle.
static const int GEODESIC_SEGMENTS = 32;

OuterBilliardsScene::OuterBilliardsScene(const vec2& dimensions)
    : CoordinateScene(dimensions) {
    manager.set({
        // How many hops to draw. Fractional: the whole part is completed hops and
        // the fraction extends the next one partway.
        {"iterations", "0"},

        // The table. shape_opacity is the master for the whole thing; the other
        // two are relative to it.
        {"shape_opacity",      "1"},
        {"shape_fill_opacity", "0.22"},
        {"vertex_dot_size",    "1"},

        // The orbits.
        {"orbit_opacity",  "1"},
        {"dot_size",       "1"},     // scales every dot
        {"line_thickness", "1"},     // scales every line
        {"orbit_fade",     "0"},     // 1 dims the oldest hop all the way out
        {"pivot_opacity",  "0"},     // mark the vertex each hop turns about
        {"rainbow",        "0"},     // blend hop color toward a hue that advances with age
        {"rainbow_period", "12"},    // hops per full trip around the color wheel

        // The n rays where the map is undefined, drawn as plain lines.
        {"ray_opacity", "0"},

        // Where every point goes, as a color. Under everything else, because it
        // fills the plane. Ramping flow_iterations is the whole show.
        {"flow_opacity",    "0"},
        {"flow_iterations", "0"},
        {"flow_scale",      "0"},   // 0 sizes the color wheel from the table
        {"flow_continuous", "1"},   // 1 turns through each hop, 0 holds every whole iterate
        {"flow_samples",    "1"},   // n x n per pixel; 1 leaves chaos as raw noise
        {"flow_shade_by_distance", "1"},   // 0 drops white-middle/fade-to-black, leaving hue alone
        {"flow_auto_depth", "0"},   // 1 adds hops as the view pans/zooms past what flow_iterations was tuned for

        // Which plane all of this lives in. 0 is Euclidean; negative bends it
        // into the hyperbolic plane of that curvature, whose ideal boundary is
        // the circle of radius 1/sqrt(-curvature). Animate it and the plane
        // curls up while the table, the orbits and the whole fractal follow.
        {"curvature",        "0"},
        {"horizon_opacity",  "0"},   // draw the ideal boundary, when there is one
        {"poincare_view",    "0"},   // 1 bows geodesics into arcs (Poincare disk) instead of Klein's straight chords

        // The singularity graph. Off by default: it is the expensive layer, and
        // a scene that only wants orbits should not pay for it.
        {"singularity_opacity",       "0"},
        {"singularity_depth",         "0"},    // preimages to draw; real, so it can be ramped
        {"singularity_width",         "1.2"},  // pixels, held constant as you zoom
        {"singularity_glow",          "0"},    // 0..1 peak of a soft halo around each line
        {"singularity_fade",          "0"},    // 1 dims the deepest layer all the way out
        {"singularity_rainbow",       "0"},    // tint lines by how deep a preimage they are
        {"singularity_rainbow_period","24"},   // layers per full trip around the color wheel

        // The periodic islands - the regions the graph never reaches.
        // The gaps the graph leaves. Filled up to the graph itself, and colored
        // by which hop brought the orbit back nearest to where it set off.
        {"island_opacity",      "0"},
        {"island_max_period",   "0"},    // 0 works it out from the shot; cost is linear in it
        {"island_period_scale", "3"},    // OCTAVES of return time per trip around the color wheel
    });
}

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

std::string OuterBilliardsScene::vertex_var(int index, const std::string& axis) {
    return "v" + std::to_string(index) + "." + axis;
}

void OuterBilliardsScene::set_shape(const std::vector<vec2>& verts) {
    // The fractal renderer carries the table inside its kernel parameter block,
    // which is why there is a ceiling at all.
    if ((int)verts.size() > MAX_BILLIARD_VERTICES) {
        throw std::runtime_error(
            "OuterBilliardsScene::set_shape: " + std::to_string(verts.size()) +
            " vertices, but the singularity renderer ships the table to the GPU inside its "
            "kernel parameters and tops out at " + std::to_string(MAX_BILLIARD_VERTICES) +
            ". Raise MAX_BILLIARD_VERTICES in Host_Device_Shared/OuterBilliardsShared.h if you "
            "really need more.");
    }

    // Retire the variables belonging to vertices the new shape does not have.
    for (int i = (int)verts.size(); i < (int)shape_target.size(); i++) {
        manager.remove(vertex_var(i, "x"));
        manager.remove(vertex_var(i, "y"));
    }

    shape_target = verts;

    StateSet equations;
    for (int i = 0; i < (int)verts.size(); i++) {
        equations[vertex_var(i, "x")] = std::to_string(verts[i].x);
        equations[vertex_var(i, "y")] = std::to_string(verts[i].y);
    }
    manager.set(equations);
}

void OuterBilliardsScene::set_regular_polygon(int sides, float radius, float rotation, const vec2& center) {
    set_shape(OuterBilliards::regular_polygon(sides, radius, rotation, center));
}

void OuterBilliardsScene::move_vertex(const TransitionType tt, int index, const vec2& to, bool smooth) {
    if (index < 0 || index >= (int)shape_target.size()) {
        throw std::runtime_error(
            "OuterBilliardsScene::move_vertex: no vertex " + std::to_string(index) +
            "; the table has " + std::to_string(shape_target.size()) + ".");
    }
    shape_target[index] = to;
    manager.transition(tt, {
        {vertex_var(index, "x"), std::to_string(to.x)},
        {vertex_var(index, "y"), std::to_string(to.y)},
    }, smooth);
}

void OuterBilliardsScene::morph_shape(const TransitionType tt, const std::vector<vec2>& verts, bool smooth) {
    if (verts.size() != shape_target.size()) {
        throw std::runtime_error(
            "OuterBilliardsScene::morph_shape: cannot morph a " + std::to_string(shape_target.size()) +
            "-vertex table into a " + std::to_string(verts.size()) + "-vertex one. Vertices are animated "
            "one to one, so the counts have to match - use set_shape() to swap in a different polygon.");
    }
    StateSet equations;
    for (int i = 0; i < (int)verts.size(); i++) {
        equations[vertex_var(i, "x")] = std::to_string(verts[i].x);
        equations[vertex_var(i, "y")] = std::to_string(verts[i].y);
    }
    shape_target = verts;
    manager.transition(tt, equations, smooth);
}

// ---------------------------------------------------------------------------
// Orbits
// ---------------------------------------------------------------------------

std::string OuterBilliardsScene::orbit_var(const std::string& name, const std::string& axis) {
    return name + "." + axis;
}

int OuterBilliardsScene::find_orbit(const std::string& name) const {
    for (int i = 0; i < (int)orbits.size(); i++) if (orbits[i].name == name) return i;
    return -1;
}

const std::string& OuterBilliardsScene::require_orbit(const std::string& name, const std::string& caller) const {
    if (find_orbit(name) < 0) {
        throw std::runtime_error("OuterBilliardsScene::" + caller + ": no orbit named '" + name +
                                 "'. Add it with add_orbit() first.");
    }
    return name;
}

bool OuterBilliardsScene::has_orbit(const std::string& name) const { return find_orbit(name) >= 0; }

void OuterBilliardsScene::add_orbit(const std::string& name, const vec2& start, uint32_t color) {
    if (find_orbit(name) >= 0) {
        throw std::runtime_error("OuterBilliardsScene::add_orbit: there is already an orbit named '" + name + "'.");
    }
    orbits.push_back({name, color, start});
    manager.set({
        {orbit_var(name, "x"),       std::to_string(start.x)},
        {orbit_var(name, "y"),       std::to_string(start.y)},
        {orbit_var(name, "opacity"), "1"},
        // Follow the scene-wide hop count until someone overrides this one.
        {orbit_var(name, "iterations"), "<iterations>"},
    });
}

void OuterBilliardsScene::remove_orbit(const std::string& name) {
    const int index = find_orbit(name);
    if (index < 0) return;
    orbits.erase(orbits.begin() + index);
    manager.remove({orbit_var(name, "x"), orbit_var(name, "y"),
                    orbit_var(name, "opacity"), orbit_var(name, "iterations")});
}

void OuterBilliardsScene::move_start(const TransitionType tt, const std::string& name, const vec2& to, bool smooth) {
    require_orbit(name, "move_start");
    orbits[find_orbit(name)].start_target = to;
    manager.transition(tt, {
        {orbit_var(name, "x"), std::to_string(to.x)},
        {orbit_var(name, "y"), std::to_string(to.y)},
    }, smooth);
}

void OuterBilliardsScene::move_start(const TransitionType tt, const vec2& to, bool smooth) {
    if (orbits.empty()) {
        throw std::runtime_error("OuterBilliardsScene::move_start: this scene has no orbits yet. "
                                 "Add one with add_orbit() first.");
    }
    move_start(tt, orbits.front().name, to, smooth);
}

vec2 OuterBilliardsScene::start_of(const std::string& name) const {
    const int index = find_orbit(name);
    if (index < 0) {
        throw std::runtime_error("OuterBilliardsScene::start_of: no orbit named '" + name + "'.");
    }
    return orbits[index].start_target;
}

void OuterBilliardsScene::fade_orbit(const TransitionType tt, const std::string& name, double opacity) {
    require_orbit(name, "fade_orbit");
    manager.transition(tt, orbit_var(name, "opacity"), std::to_string(opacity));
}

void OuterBilliardsScene::set_orbit_color(const std::string& name, uint32_t color) {
    require_orbit(name, "set_orbit_color");
    orbits[find_orbit(name)].color = color;
}

// ---------------------------------------------------------------------------
// How many hops
// ---------------------------------------------------------------------------

void OuterBilliardsScene::set_iterations(double count) {
    manager.set("iterations", std::to_string(count));
}

void OuterBilliardsScene::iterate_to(const TransitionType tt, double count, bool smooth) {
    manager.transition(tt, "iterations", std::to_string(count), smooth);
}

void OuterBilliardsScene::iterate_orbit_to(const TransitionType tt, const std::string& name, double count, bool smooth) {
    require_orbit(name, "iterate_orbit_to");
    manager.transition(tt, orbit_var(name, "iterations"), std::to_string(count), smooth);
}

void OuterBilliardsScene::follow_global_iterations(const TransitionType tt, const std::string& name) {
    require_orbit(name, "follow_global_iterations");
    manager.transition(tt, orbit_var(name, "iterations"), "<iterations>");
}

// ---------------------------------------------------------------------------
// The singularity graph
// ---------------------------------------------------------------------------

void OuterBilliardsScene::fade_rays(const TransitionType tt, double opacity, bool smooth) {
    manager.transition(tt, "ray_opacity", std::to_string(opacity), smooth);
}

void OuterBilliardsScene::set_singularity_depth(double depth) {
    manager.set("singularity_depth", std::to_string(depth));
}

void OuterBilliardsScene::grow_singularities(const TransitionType tt, double depth, bool smooth) {
    // Growing the web with the layer still switched off would unfold the whole
    // thing invisibly and then pop it on, which is never what the caller meant.
    manager.transition(tt, {
        {"singularity_depth",   std::to_string(depth)},
        {"singularity_opacity", "1"},
    }, smooth);
}

void OuterBilliardsScene::fade_singularities(const TransitionType tt, double opacity, bool smooth) {
    manager.transition(tt, "singularity_opacity", std::to_string(opacity), smooth);
}

void OuterBilliardsScene::fade_islands(const TransitionType tt, double opacity, bool smooth) {
    manager.transition(tt, "island_opacity", std::to_string(opacity), smooth);
}

void OuterBilliardsScene::set_island_max_period(int max_period) {
    manager.set("island_max_period", std::to_string(max_period));
}

// ---------------------------------------------------------------------------
// Where every point goes
// ---------------------------------------------------------------------------

void OuterBilliardsScene::fade_flow(const TransitionType tt, double opacity, bool smooth) {
    manager.transition(tt, "flow_opacity", std::to_string(opacity), smooth);
}

void OuterBilliardsScene::set_flow_iterations(double count) {
    manager.set("flow_iterations", std::to_string(count));
}

void OuterBilliardsScene::flow_to(const TransitionType tt, double count, bool smooth) {
    // Linear, not eased: the point of this layer is to watch the coloring evolve
    // at a steady rate, and an eased ramp would make it look like the dynamics
    // were speeding up and slowing down.
    manager.transition(tt, "flow_iterations", std::to_string(count), smooth);
}

void OuterBilliardsScene::set_flow_continuous(bool continuous) {
    manager.set("flow_continuous", continuous ? "1" : "0");
}

void OuterBilliardsScene::set_flow_auto_depth(bool on) {
    manager.set("flow_auto_depth", on ? "1" : "0");
}

// ---------------------------------------------------------------------------
// Curvature
// ---------------------------------------------------------------------------

void OuterBilliardsScene::set_curvature(double curvature) {
    manager.set("curvature", std::to_string(curvature));
}

void OuterBilliardsScene::bend_space(const TransitionType tt, double curvature, bool smooth) {
    // The ideal boundary is the one landmark that says which plane you are in, so
    // it comes along automatically - and goes away again in the flat limit, where
    // it is infinitely far off and would be a lie to draw.
    manager.transition(tt, {
        {"curvature",       std::to_string(curvature)},
        {"horizon_opacity", curvature < 0.0 ? "1" : "0"},
    }, smooth);
}

double OuterBilliardsScene::horizon_for(double curvature) {
    return curvature < 0.0 ? 1.0 / std::sqrt(-curvature) : 0.0;
}

double OuterBilliardsScene::curvature_for_horizon(double horizon) {
    if (!(horizon > 1e-6)) return 0.0;
    return -1.0 / (horizon * horizon);
}

// ---------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------

// CoordinateScene stores the viewport as window_height = 0.2*exp(zoom), which it
// then INVERTS to place the edges - so the visible half-height in world units is
// 2.5*exp(-zoom), and bigger zoom means closer in. Solve that for zoom.
static std::string zoom_for_half_height(float half_height) {
    if (!(half_height > 1e-5f)) half_height = 1e-5f;
    return std::to_string(std::log(2.5f / half_height));
}

void OuterBilliardsScene::frame_view(const vec2& center, float half_height) {
    manager.set({
        {"center_x", std::to_string(center.x)},
        {"center_y", std::to_string(center.y)},
        {"zoom",     zoom_for_half_height(half_height)},
    });
}

void OuterBilliardsScene::frame_view(const TransitionType tt, const vec2& center, float half_height, bool smooth) {
    manager.transition(tt, {
        {"center_x", std::to_string(center.x)},
        {"center_y", std::to_string(center.y)},
        {"zoom",     zoom_for_half_height(half_height)},
    }, smooth);
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

std::vector<vec2> OuterBilliardsScene::current_shape() const {
    std::vector<vec2> verts;
    verts.reserve(shape_target.size());
    for (int i = 0; i < (int)shape_target.size(); i++) {
        verts.push_back(vec2(state[vertex_var(i, "x")], state[vertex_var(i, "y")]));
    }
    return verts;
}

// How many hops to allow before giving up on an orbit closing.
//
// A closed orbit's period grows with how far out it starts. Two hops compose to a
// translation by twice a side of the table, so getting once around takes roughly
// pi * distance / side hops, and closing up takes several trips - which means a
// FIXED ceiling does not cut off some uniform fringe, it cuts off exactly the
// corners of the shot, the part of the view farthest from the table. That is
// what used to leave the outer regions unshaded. Scale it with how far the view
// actually reaches instead, measured in the plane's own metric so that a curved
// plane - where the far field is much farther away than it looks - gets its share.
// The constant is empirical, and generous on purpose: the cells farthest out are
// the ones whose periods are hardest to predict, and an orbit that is one hop
// short of closing shades as though it never closes at all.
static const float PERIODS_PER_RADIUS = 90.0f;
static const int   MIN_AUTO_PERIOD = 24;
static const int   MAX_AUTO_PERIOD = 1600;   // cost is linear in this; somewhere it has to stop

// How much deeper than the drawn web the islands look when deciding where their
// own edges are. See where it is used.
static const float ISLAND_BOUNDARY_DEPTH = 3.0f;
static const int   MAX_ISLAND_DEPTH = 2000;

float OuterBilliardsScene::view_reach(const OuterBilliards& table) {
    const vec2 center = table.centroid();
    const float lx = (float)state["left_x"],   rx = (float)state["right_x"];
    const float ty = (float)state["top_y"],    by = (float)state["bottom_y"];
    const vec2 corners[4] = {vec2(lx, ty), vec2(rx, ty), vec2(lx, by), vec2(rx, by)};

    // In a curved plane the view can extend past the ideal boundary, where there
    // is no distance to measure; pull those corners just inside it.
    const float edge = table.horizon();

    float reach = 0.0f;
    for (int i = 0; i < 4; i++) {
        vec2 corner = corners[i];
        if (edge > 0.0f) {
            const float len = length(corner);
            if (len > edge * 0.999f) corner = corner * (edge * 0.999f / len);
        }
        reach = std::fmax(reach, curved_distance(center, corner, table.curvature));
    }
    return reach;
}

int OuterBilliardsScene::auto_island_period(const OuterBilliards& table) {
    const float radius = table.circumradius();
    if (!(radius > 1e-6f)) return MIN_AUTO_PERIOD;

    const int wanted = (int)(PERIODS_PER_RADIUS * view_reach(table) / radius);
    return std::max(MIN_AUTO_PERIOD, std::min(MAX_AUTO_PERIOD, wanted));
}

// How many extra hops flow_auto_depth adds per radius the view reaches past the
// table, and per doubling of on-screen resolution past the reference below -
// mirrors auto_island_period's reasoning: a hop is a bounded step, so a point
// twice as far out, or a boundary examined at twice the pixel density, takes
// roughly twice as many hops before its long-run color can be trusted. Both
// constants are empirical, and deliberately conservative: at the reference
// reach and resolution below they contribute nothing, so a shot that never
// turns flow_auto_depth on - which is every scripted shot in the codebase
// today - sees no change at all.
static const float FLOW_ITERATIONS_PER_RADIUS         = 40.0f;
static const float FLOW_REFERENCE_PIXELS_PER_RADIUS   = 200.0f;
static const float FLOW_ITERATIONS_PER_DOUBLING       = 600.0f;
static const double MAX_AUTO_FLOW_ITERATIONS          = 60000.0;   // cost is linear; somewhere it has to stop

double OuterBilliardsScene::auto_flow_iterations(const OuterBilliards& table) {
    const double base = std::fmax((double)state["flow_iterations"], 0.0);

    const float radius = table.circumradius();
    if (!(radius > 1e-6f)) return base;

    // Panning: how much farther the view reaches than the table itself.
    const float reach = view_reach(table);
    const float pan_extra = std::fmax(0.0f, FLOW_ITERATIONS_PER_RADIUS * (reach / radius - 1.0f));

    // Zooming: how much finer the view resolves the table than the reference
    // resolution flow_iterations is assumed tuned for. Additive per doubling,
    // not multiplicative per pixel, because it is the chaotic separation between
    // neighboring orbits - roughly exponential in hop count - that needs
    // resolving, not the pixel count itself.
    const float wpp = world_per_pixel();
    const float pixels_per_radius = (wpp > 1e-9f) ? radius / wpp : 0.0f;
    const float doublings = (pixels_per_radius > FLOW_REFERENCE_PIXELS_PER_RADIUS)
        ? std::log2(pixels_per_radius / FLOW_REFERENCE_PIXELS_PER_RADIUS) : 0.0f;
    const float zoom_extra = FLOW_ITERATIONS_PER_DOUBLING * doublings;

    return std::min(MAX_AUTO_FLOW_ITERATIONS, base + (double)pan_extra + (double)zoom_extra);
}

float OuterBilliardsScene::world_per_pixel() {
    const int height = get_height();
    if (height <= 0) return 1.0f;
    // CoordinateScene keeps the aspect square, so the vertical span alone settles
    // what a pixel is worth.
    return (float)(state["bottom_y"] - state["top_y"]) / (float)height;
}

vec2 OuterBilliardsScene::to_pixel_warped(const vec2& klein_point, const OuterBilliards& table) {
    if (state["poincare_view"] > 0.5) {
        return point_to_pixel(klein_to_poincare(klein_point, table.horizon()));
    }
    return point_to_pixel(klein_point);
}

void OuterBilliardsScene::push_geodesic(const vec2& a, const vec2& b, const OuterBilliards& table,
                                        uint32_t color, float opacity, float thickness) {
    if (state["poincare_view"] <= 0.5) {
        push_segment(point_to_pixel(a), point_to_pixel(b), color, opacity, thickness);
        return;
    }

    // A Klein geodesic between a and b is exactly their straight interpolation -
    // that is the whole point of the Klein model - so subdividing that segment
    // and bowing each piece traces the true Poincare arc between the same two
    // points.
    vec2 previous = to_pixel_warped(a, table);
    for (int i = 1; i <= GEODESIC_SEGMENTS; i++) {
        const float t = (float)i / (float)GEODESIC_SEGMENTS;
        const vec2 next = to_pixel_warped(veclerp(a, b, t), table);
        push_segment(previous, next, color, opacity, thickness);
        previous = next;
    }
}

// Where every point goes, painted as a color wheel carried along by the map.
// This fills the plane, so it goes down before anything else.
void OuterBilliardsScene::draw_flow_field(const OuterBilliards& table) {
    const float opacity = (float)state["flow_opacity"];
    if (opacity < MIN_OPACITY) return;

    const int n = table.size();
    if (n < 3 || n > MAX_BILLIARD_VERTICES) return;

    FlowFieldParams params;
    for (int i = 0; i < n; i++) params.verts[i] = table.vertices[i];
    params.n = n;
    params.curvature = table.curvature;
    params.poincare_view = (state["poincare_view"] > 0.5) ? 1 : 0;
    params.lx_ty = vec2(state["left_x"], state["top_y"]);
    params.rx_by = vec2(state["right_x"], state["bottom_y"]);

    const bool auto_depth = state["flow_auto_depth"] > 0.5;
    params.iterations = std::fmax(
        (float)(auto_depth ? auto_flow_iterations(table) : (double)state["flow_iterations"]), 0.0f);
    params.smooth     = (state["flow_continuous"] > 0.5) ? 1 : 0;
    params.samples    = std::max(1, std::min(4, (int)state["flow_samples"]));

    params.center  = table.centroid();
    // The wheel wants to be saturated across the band the orbits actually live
    // in, which is a few table-radii out; zero asks for that rather than a number.
    const float requested = (float)state["flow_scale"];
    params.scale   = (requested > 1e-6f) ? requested : 3.0f * table.circumradius();
    params.shade_by_distance = (state["flow_shade_by_distance"] > 0.5) ? 1 : 0;
    params.opacity = opacity;

    outer_billiards_flow_render(gpu_pix.get_ptr(), panel_wh, params);
}

// ---------------------------------------------------------------------------
// The fractal. One GPU launch draws both the islands and the web, because they
// come out of the same forward orbit per pixel - see the kernel for why that
// walk gives the preimages without ever constructing them.
// ---------------------------------------------------------------------------
void OuterBilliardsScene::draw_singularity_graph(const OuterBilliards& table) {
    const float web_opacity    = (float)state["singularity_opacity"];
    const float island_opacity = (float)state["island_opacity"];
    const float depth          = (float)state["singularity_depth"];

    // Zero means "work it out from the shot", which is almost always better than
    // a number picked by hand - see auto_island_period.
    const int requested = (int)state["island_max_period"];
    const int max_period = (requested > 0) ? requested : auto_island_period(table);

    const bool wants_web     = web_opacity    > MIN_OPACITY && depth > 0.0f;
    const bool wants_islands = island_opacity > MIN_OPACITY && max_period > 1 && depth > 0.0f;
    if (!wants_web && !wants_islands) return;

    // Which side of a vertex its ray leaves from depends on the winding, so the
    // renderer wants the vertices counterclockwise. Nothing else here does.
    const std::vector<vec2> wound = table.counterclockwise();
    const std::vector<SingularRay> rays = table.singular_ray_data();
    const int n = (int)wound.size();
    // A table mid-collapse has fewer sides than vertices, so the two counts are
    // tracked separately - fewer than three rays and there is no graph to draw.
    const int ray_count = (int)rays.size();
    if (n < 3 || n > MAX_BILLIARD_VERTICES || ray_count < 3) return;

    const float wpp = world_per_pixel();

    SingularityGraphParams params;
    for (int i = 0; i < n; i++) params.verts[i] = wound[i];
    for (int i = 0; i < ray_count; i++) params.rays[i] = rays[i];
    params.n = n;
    params.ray_count = ray_count;
    params.curvature = table.curvature;
    params.poincare_view = (state["poincare_view"] > 0.5) ? 1 : 0;
    params.lx_ty = vec2(state["left_x"], state["top_y"]);
    params.rx_by = vec2(state["right_x"], state["bottom_y"]);
    params.world_per_pixel = wpp;

    params.web_opacity    = wants_web ? web_opacity : 0.0f;
    params.depth          = depth;
    params.line_width     = (float)state["singularity_width"];
    params.glow           = (float)state["singularity_glow"];
    params.fade           = (float)state["singularity_fade"];
    params.rainbow        = (float)state["singularity_rainbow"];
    params.rainbow_period = std::fmax((float)state["singularity_rainbow_period"], 1e-3f);
    params.line_color     = singularity_color;

    params.island_opacity = wants_islands ? island_opacity : 0.0f;
    params.max_period     = max_period;
    // The fill is bounded by a deeper graph than the one on screen, so that the
    // gaps between the hairlines of the chaotic region - which are gaps only
    // because the drawing stopped - close up instead of filling with color.
    // Three times over is enough to separate them from the real islands, which
    // stay open at any depth.
    params.island_depth   = std::min((int)std::ceil(depth * ISLAND_BOUNDARY_DEPTH), MAX_ISLAND_DEPTH);
    params.period_octaves = std::fmax((float)state["island_period_scale"], 1e-3f);

    outer_billiards_singularity_render(gpu_pix.get_ptr(), panel_wh, params);
}

// The bare singular rays, as vectors. This is layer zero of the web, drawn
// crisply and on its own so a shot can point at it before the fractal grows.
// Geodesics are straight in these coordinates, so this is the same code in
// either plane - the rays just stop at the ideal boundary in a curved one.
void OuterBilliardsScene::draw_singular_rays(const OuterBilliards& table, float thickness) {
    const float opacity = (float)state["ray_opacity"];
    if (opacity < MIN_OPACITY) return;

    // A ray is infinite; this reaches the far corner of the panel from anywhere
    // the table could plausibly be, and push_segment culls what misses.
    const float reach = (float)(state["right_x"] - state["left_x"])
                      + (float)(state["bottom_y"] - state["top_y"])
                      + 4.0f * table.circumradius();

    for (const OuterBilliards::Ray& ray : table.singular_rays(reach)) {
        push_geodesic(ray.origin, ray.origin + ray.direction * ray.length, table,
                      singularity_color, opacity, thickness);
    }
}

// The ideal boundary: the circle a hyperbolic plane compresses into, infinitely
// far away in its own metric and drawn here because otherwise nothing on screen
// says which plane you are looking at. There is none in the Euclidean case, so
// this quietly draws nothing there.
void OuterBilliardsScene::draw_horizon(const OuterBilliards& table, float thickness) {
    const float opacity = (float)state["horizon_opacity"];
    const float radius = table.horizon();
    if (opacity < MIN_OPACITY || radius <= 0.0f) return;

    // Fine enough that the seams do not show at the zooms this scene invites,
    // and cheap enough not to matter: the segments are culled per pixel anyway.
    const int SIDES = 512;
    vec2 previous = point_to_pixel(vec2(radius, 0));
    for (int i = 1; i <= SIDES; i++) {
        const float angle = 6.283185307179586f * (float)i / (float)SIDES;
        const vec2 next = point_to_pixel(vec2(std::cos(angle), std::sin(angle)) * radius);
        push_segment(previous, next, horizon_color, opacity, thickness);
        previous = next;
    }
}

void OuterBilliardsScene::push_segment(const vec2& a, const vec2& b, uint32_t color, float opacity, float thickness) {
    if (opacity < MIN_OPACITY || thickness < MIN_SIZE) return;
    if (hasnan(a) || hasnan(b)) return;
    const float margin = thickness * 0.5f + 1.0f;
    // Both ends past the same edge of the panel means the whole segment is.
    if ((a.x < -margin              && b.x < -margin) ||
        (a.x > panel_wh.x + margin  && b.x > panel_wh.x + margin) ||
        (a.y < -margin              && b.y < -margin) ||
        (a.y > panel_wh.y + margin  && b.y > panel_wh.y + margin)) return;
    segments.push_back(Segment2D(a, b, color, opacity, thickness));
}

void OuterBilliardsScene::push_dot(const vec2& center, float radius, uint32_t color, float opacity) {
    if (opacity < MIN_OPACITY || radius < MIN_SIZE) return;
    if (hasnan(center)) return;
    const float margin = radius + 1.0f;
    if (center.x < -margin || center.x > panel_wh.x + margin ||
        center.y < -margin || center.y > panel_wh.y + margin) return;
    dots.push_back(Dot2D(center, radius, color, opacity));
}

void OuterBilliardsScene::draw_orbit(const OrbitSpec& orbit, const OuterBilliards& table,
                                     float thickness, float dot_radius) {
    const float opacity = (float)state["orbit_opacity"] * (float)state[orbit_var(orbit.name, "opacity")];
    if (opacity < MIN_OPACITY) return;

    const vec2 start(state[orbit_var(orbit.name, "x")], state[orbit_var(orbit.name, "y")]);

    // The whole part of the hop count is how many hops have landed; the fraction
    // is how far along the next one the moving head has travelled.
    const double count = state[orbit_var(orbit.name, "iterations")];
    const int whole = (count > 0.0) ? (int)std::floor(count) : 0;
    const float partial = (count > 0.0) ? (float)(count - whole) : 0.0f;
    const bool wants_partial = partial > 1e-4f;

    std::vector<int> pivots;
    const std::vector<vec2> path = table.orbit(start, whole + (wants_partial ? 1 : 0), &pivots);
    const int hops = (int)path.size() - 1;

    // The head is only mid-flight if the orbit actually got that far - it stops
    // early if it ever reaches a point the map cannot continue from.
    const bool head_in_flight = wants_partial && (int)path.size() == whole + 2;

    const uint32_t color = orbit.color ? orbit.color : orbit_color;
    const float fade           = (float)state["orbit_fade"];
    const float rainbow_mix    = (float)state["rainbow"];
    const float rainbow_period = std::fmax((float)state["rainbow_period"], 1e-3f);

    // The hops themselves. Each one is a straight segment whose midpoint is the
    // table vertex it turned about.
    for (int k = 0; k < hops; k++) {
        const vec2 from = path[k];
        vec2 to = path[k + 1];
        if (head_in_flight && k == hops - 1) to = veclerp(from, to, partial);

        // age runs 0 at the oldest hop to 1 at the newest, so orbit_fade dims the
        // trail behind the head without ever touching the head itself.
        const float age = (hops > 1) ? (float)(k + 1) / (float)hops : 1.0f;
        uint32_t hop_color = color;
        if (rainbow_mix > 0.001f) hop_color = colorlerp(color, rainbow((float)k / rainbow_period), rainbow_mix);
        push_geodesic(from, to, table, hop_color, opacity * (1.0f - fade * (1.0f - age)), thickness);
    }

    // The vertex each hop turned about, if asked for.
    const float pivot_opacity = opacity * (float)state["pivot_opacity"];
    for (int k = 0; k < (int)pivots.size(); k++) {
        push_dot(to_pixel_warped(table.vertices[pivots[k]], table), dot_radius * EMPHASIS, pivot_color, pivot_opacity);
    }

    // A dot everywhere the orbit has landed, then the moving head, then the point
    // it all started from - drawn last and biggest, so it stays legible however
    // dense the orbit gets around it.
    const int landed = head_in_flight ? (int)path.size() - 2 : (int)path.size() - 1;
    for (int k = 1; k <= landed; k++) push_dot(to_pixel_warped(path[k], table), dot_radius, color, opacity);
    if (head_in_flight) {
        push_dot(to_pixel_warped(veclerp(path[landed], path[landed + 1], partial), table), dot_radius, color, opacity);
    }
    push_dot(to_pixel_warped(path[0], table), dot_radius * EMPHASIS, start_color, opacity);
}

void OuterBilliardsScene::draw() {
    panel_wh = get_width_height();

    // The panel arrives zeroized to fully transparent; lay the background down
    // before anything else goes on it.
    if (background_color != 0) cuda_fill_pixels(gpu_pix.get_ptr(), panel_wh, background_color);

    // Rebuild the table from the vertex variables as they stand this frame, and
    // with it every orbit and the whole fractal - which is what makes all of them
    // follow along when the table deforms, or when the plane it sits in does.
    const OuterBilliards table(current_shape(), (float)state["curvature"]);
    const int n = table.size();

    // The flow fills the plane, so it is the floor; the fractal is the backdrop
    // everything else is read against, so it comes next - and both go under the
    // coordinate grid, which is there to be read.
    draw_flow_field(table);
    draw_singularity_graph(table);

    // The coordinate grid, if ticks_opacity was turned up.
    CoordinateScene::draw();

    const float gm = get_geom_mean_size();
    const float thickness  = gm / 320.0f * (float)state["line_thickness"];
    const float dot_radius = gm / 210.0f * (float)state["dot_size"];
    const float shape_opacity = (float)state["shape_opacity"];

    // Used for the fill (below, always straight-edged - see push_geodesic's
    // header comment) and the vertex dots (below, correctly warped either way).
    pixel_verts.clear();
    for (const vec2& v : table.vertices) pixel_verts.push_back(to_pixel_warped(v, table));

    // The table's interior, beneath everything.
    const float fill_opacity = shape_opacity * (float)state["shape_fill_opacity"];
    if (n >= 3 && fill_opacity > MIN_OPACITY) {
        // The fill kernel wants the interior on the left of every directed edge.
        // Pixel space flips y, and the caller's winding is their business, so
        // settle it here from the signed area.
        std::vector<vec2> wound = pixel_verts;
        const float double_area = billiards_double_signed_area(wound.data(), n);
        if (double_area < 0.0f) std::reverse(wound.begin(), wound.end());
        billiards_fill_polygon(gpu_pix.get_ptr(), panel_wh, wound.data(), n, table_fill_color, fill_opacity);
    }

    segments.clear();
    dots.clear();

    // Horizon under rays under orbits under the table outline - so the thing
    // being explained is never hidden by the thing explaining it.
    draw_horizon(table, thickness * 0.8f);
    draw_singular_rays(table, thickness);
    for (const OrbitSpec& orbit : orbits) draw_orbit(orbit, table, thickness, dot_radius);

    if (n >= 2 && shape_opacity > MIN_OPACITY) {
        for (int i = 0; i < n; i++) {
            push_geodesic(table.vertices[i], table.vertices[(i + 1) % n], table,
                         table_color, shape_opacity, thickness * 1.25f);
        }
        const float vertex_radius = dot_radius * (float)state["vertex_dot_size"];
        for (int i = 0; i < n; i++) push_dot(pixel_verts[i], vertex_radius, table_color, shape_opacity);
    }

    // Two launches, in order: lines, then every dot on top of them.
    billiards_draw_segments(gpu_pix.get_ptr(), panel_wh, segments.data(), (int)segments.size());
    billiards_draw_dots(gpu_pix.get_ptr(), panel_wh, dots.data(), (int)dots.size());
}
