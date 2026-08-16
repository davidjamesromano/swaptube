#pragma once

#include "../Common/CoordinateScene.h"
#include "../../DataObjects/OuterBilliards.h"
#include "../../Host_Device_Shared/BilliardsStructs.h"
#include "../../Host_Device_Shared/OuterBilliardsShared.h"
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Outer billiards, drawn on a coordinate plane. See DataObjects/OuterBilliards.h
// for what the map actually is.
//
// Five layers, back to front, each independently animatable and all off by
// default except the table:
//
//   THE FLOW    the plane painted with a color wheel, each pixel recolored
//               with the color of wherever its own orbit has reached - see
//               fade_flow(). How grainy it gets is decided by the table and
//               the framing, not the hop count (see OrderAndChaos.cpp).
//   THE ISLANDS a flat color in each region the singularity graph never
//               reaches, hued by which hop brings the orbit back nearest to
//               where it set off.
//   THE WEB     the singularity graph: every point that lands on a ray where
//               the map is undefined, within some number of hops. Rendered
//               per-pixel on the GPU.
//   THE RAYS    just the n rays themselves, drawn crisply as vectors.
//   THE TABLE   the polygon, and orbits hopping around it.
//
// Everything that moves is a state variable, animatable with
// manager.transition(); the knobs central to the design have wrappers:
// set_shape()/set_regular_polygon() (the table), add_orbit()/move_start()
// (where an orbit sets off from), iterate_to() ("iterations", how many hops -
// real, so the fraction draws the next hop partway), grow_singularities()
// ("singularity_depth", same real-valued unfolding), and bend_space()
// ("curvature": 0 Euclidean, negative hyperbolic in the Beltrami-Klein model -
// see OuterBilliardsShared.h).
//
// Other state variables, all with sensible defaults: shape_opacity,
// shape_fill_opacity, vertex_dot_size, orbit_opacity, dot_size, line_thickness,
// orbit_fade (dim the older hops), pivot_opacity (mark the vertex each hop
// turns about), rainbow/rainbow_period (tint hops by age), ray_opacity,
// horizon_opacity, singularity_opacity, singularity_width, singularity_glow,
// singularity_rainbow/singularity_rainbow_period, island_opacity,
// island_max_period (0 sizes it from the shot), poincare_view (draw the
// Poincare disk instead of Klein's straight chords), flow_scale,
// flow_shade_by_distance, flow_auto_depth. Panning and zooming come from
// CoordinateScene: center_x, center_y, zoom - or use frame_view(), in world
// units.
// ---------------------------------------------------------------------------
class OuterBilliardsScene : public CoordinateScene {
public:
    OuterBilliardsScene(const vec2& dimensions = vec2(1, 1));

    // --- the table -------------------------------------------------------
    // Installs a polygon, replacing whatever was there. Do not call this while
    // the vertices are mid-transition; between microblocks is the right time.
    void set_shape(const std::vector<vec2>& verts);
    void set_regular_polygon(int sides, float radius = 1.0f, float rotation = 0.0f, const vec2& center = vec2(0, 0));

    // Animate one corner, or all of them. morph_shape needs a polygon with the
    // same number of vertices as the current one - to change the count, fade the
    // scene out and set_shape() a new one.
    void move_vertex(const TransitionType tt, int index, const vec2& to, bool smooth = true);
    void morph_shape(const TransitionType tt, const std::vector<vec2>& verts, bool smooth = true);

    // Where the vertices are headed. During a transition the drawn table is
    // somewhere between this and where it started; this is the destination.
    const std::vector<vec2>& shape() const { return shape_target; }
    int num_vertices() const { return (int)shape_target.size(); }
    static std::string vertex_var(int index, const std::string& axis);

    // --- orbits ----------------------------------------------------------
    // color 0 means "use orbit_color".
    void add_orbit(const std::string& name, const vec2& start, uint32_t color = 0);
    void remove_orbit(const std::string& name);

    void move_start(const TransitionType tt, const std::string& name, const vec2& to, bool smooth = true);
    void move_start(const TransitionType tt, const vec2& to, bool smooth = true);   // the default orbit

    void fade_orbit(const TransitionType tt, const std::string& name, double opacity);

    static std::string orbit_var(const std::string& name, const std::string& axis);

    // --- how many hops ---------------------------------------------------
    // "iterations": whole part is completed hops, fraction draws the next one
    // partway - every orbit shares this count.
    void set_iterations(double count);
    void iterate_to(const TransitionType tt, double count, bool smooth = true);

    // --- the singularity graph -------------------------------------------
    // Show the n rays where the map itself is undefined - the seed the whole
    // fractal is grown from.
    void fade_rays(const TransitionType tt, double opacity, bool smooth = true);

    // Set the depth outright, or unfold the web to it. Depth is real, same as
    // iterations. grow_singularities() also turns the layer on if it is off,
    // so it is usually the only call a shot needs.
    void set_singularity_depth(double depth);
    void grow_singularities(const TransitionType tt, double depth, bool smooth = true);
    void fade_singularities(const TransitionType tt, double opacity, bool smooth = true);

    // The regions the web never reaches, filled by period. "island_max_period"
    // (0 by default, sizing it from the shot) is a plain state variable, not
    // wrapped here - cost is linear in it, so raise it by hand only if the
    // auto-sized search is visibly leaving corners unshaded.
    void fade_islands(const TransitionType tt, double opacity, bool smooth = true);

    // --- where every point goes ------------------------------------------
    // Paints the plane with a color wheel, then recolors each pixel with the
    // color of wherever ITS orbit has got to - order shows as patches that
    // keep their color, chaos as noise, an escaping orbit as a fade to black
    // (flow_shade_by_distance, on by default). See FlowFieldParams in
    // Host_Device_Shared/OuterBilliardsShared.h.
    //
    // flow_to() is the whole show: ramp it LINEARLY (never eased) and watch
    // the plane sort itself out. A hop is a half turn, so the coloring is
    // completely rearranged once per unit of flow_iterations - faster than
    // about four or five a second and consecutive frames stop looking related.
    // The picture converges early (on a pentagon, 100 hops and 3200 hops agree
    // to about a part in a hundred), so there is little to gain from going deep.
    void fade_flow(const TransitionType tt, double opacity, bool smooth = true);
    void set_flow_iterations(double count);
    void flow_to(const TransitionType tt, double count, bool smooth = false);

    // Off by default. On, the iteration count sent to the GPU is
    // flow_iterations PLUS however much more the CURRENT view needs -
    // panning past the table or zooming in past the resolution
    // flow_iterations was tuned for both call for more hops before the
    // coloring is trustworthy (see auto_flow_iterations()). Meant for
    // interactive exploration (open_ui()), not scripted shots that already
    // pick flow_iterations by hand.
    void set_flow_auto_depth(bool on);

    // --- which plane all of this lives in --------------------------------
    // "curvature": 0 Euclidean, negative hyperbolic (Beltrami-Klein model -
    // geodesics stay straight, every layer above keeps working unchanged).
    // bend_space() animates it and brings the ideal boundary along.
    //
    // "poincare_view" (plain state variable, off by default) redraws the same
    // plane in the Poincare disk model instead - geodesics bow into arcs -
    // without touching the map, the metric, or the ideal boundary.
    void set_curvature(double curvature);
    void bend_space(const TransitionType tt, double curvature, bool smooth = true);

    // The ideal boundary's radius for a given curvature - the inverse of
    // bend_space()'s input, handy for sizing a shot ("fit the whole plane in a
    // disk of radius 3").
    static double curvature_for_horizon(double horizon);

    // --- framing ---------------------------------------------------------
    // half_height is how far above the center of the panel you want to see, in
    // world units. (CoordinateScene's zoom is logarithmic and inverted; this is
    // the same thing said in units the shape is measured in.)
    void frame_view(const vec2& center, float half_height);
    void frame_view(const TransitionType tt, const vec2& center, float half_height, bool smooth = true);

    // --- appearance (not animatable; opacities are, above) ---------------
    uint32_t table_color      = 0xffffc040;   // outline and vertices
    uint32_t table_fill_color = 0xff704818;
    uint32_t orbit_color      = 0xff40d8ff;   // default for orbits added without one
    uint32_t start_color      = 0xffffffff;   // the dot an orbit sets off from
    uint32_t pivot_color      = 0xffff7040;   // the vertex each hop turns about
    uint32_t singularity_color = 0xffff9840;  // the web, and the bare rays
    uint32_t horizon_color     = 0xff5878a0;  // the ideal boundary of a curved plane

    void draw() override;

private:
    struct OrbitSpec {
        std::string name;
        uint32_t color;
        vec2 start_target;
    };

    std::vector<vec2>      shape_target;   // where the vertices are headed
    std::vector<OrbitSpec> orbits;

    // Rebuilt from scratch every frame, but kept around so the vectors hold on to
    // their capacity instead of reallocating for every frame of a deep orbit.
    std::vector<vec2>      pixel_verts;
    std::vector<Segment2D> segments;
    std::vector<Dot2D>     dots;
    ivec2                  panel_wh = ivec2(0, 0);   // refreshed at the top of draw()

    // The vertices as the state has them RIGHT NOW - mid-transition, that is
    // partway between where they were and shape_target. Only meaningful inside
    // draw(), where `state` is fresh.
    std::vector<vec2> current_shape() const;

    // How much of the world one pixel covers, this frame. The fractal sizes its
    // lines with this, so they stay a fixed width on screen at any zoom.
    float world_per_pixel();

    // Where a Klein-coordinate point lands on screen - point_to_pixel() as
    // normal, unless poincare_view is on, in which case it is first swapped for
    // its Poincare-disk position. See OuterBilliardsShared.h's KLEIN <-> POINCARE
    // section for what that means and why the map itself never has to change.
    vec2 to_pixel_warped(const vec2& klein_point, const OuterBilliards& table);

    // A geodesic between two Klein-coordinate points. Klein geodesics are
    // straight, so with poincare_view off this is just push_segment(); on, the
    // straight Klein segment is subdivided and each piece run through
    // to_pixel_warped(), tracing the bowed Poincare arc.
    void push_geodesic(const vec2& a, const vec2& b, const OuterBilliards& table,
                       uint32_t color, float opacity, float thickness);

    // How deep to hunt for a closed orbit, when island_max_period is left at 0.
    int auto_island_period(const OuterBilliards& table);

    // How far the current view reaches from the table's center, in the plane's
    // own curved metric - the shared distance measure behind both
    // auto_island_period and auto_flow_iterations. Corners past the ideal
    // boundary are pulled just inside it first, since there is no distance to
    // measure beyond it.
    float view_reach(const OuterBilliards& table);

    // What flow_iterations effectively is when flow_auto_depth is on: the
    // requested count, plus more for however far the view reaches past the
    // table and however far it has zoomed in past the resolution that count was
    // tuned for. See set_flow_auto_depth().
    double auto_flow_iterations(const OuterBilliards& table);

    void draw_flow_field(const OuterBilliards& table);
    void draw_singularity_graph(const OuterBilliards& table);
    void draw_singular_rays(const OuterBilliards& table, float thickness);
    void draw_horizon(const OuterBilliards& table, float thickness);

    void draw_orbit(const OrbitSpec& orbit, const OuterBilliards& table,
                    float thickness, float dot_radius);

    // Primitives far outside the panel cost the GPU almost nothing but still cost
    // a copy across the bus, so drop them here. These also swallow anything
    // non-finite, so one bad coordinate cannot take out a frame.
    void push_segment(const vec2& a, const vec2& b, uint32_t color, float opacity, float thickness);
    void push_dot(const vec2& center, float radius, uint32_t color, float opacity);

    int find_orbit(const std::string& name) const;
    const std::string& require_orbit(const std::string& name, const std::string& caller) const;
};
