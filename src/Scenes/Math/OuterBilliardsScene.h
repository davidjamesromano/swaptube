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
// The scene draws up to five layers, back to front, each one independently
// animatable and all of them off by default except the table:
//
//   THE FLOW       the plane painted with a color wheel, each pixel recolored
//                  with the color of wherever its own orbit has reached. Order,
//                  chaos and escape all read off it directly; see fade_flow().
//                  How grainy it gets is decided by the TABLE and the FRAMING,
//                  not by the hop count - a regular polygon is almost all large
//                  periodic islands whatever you do, and a generic one is almost
//                  all mosaic. Both were measured; see OrderAndChaos.cpp.
//
//   THE ISLANDS    a flat color in each region the singularity graph never
//                  reaches, hued by which hop brings the orbit back nearest to
//                  where it set off. Filled right up to the graph, which bounds
//                  them - islands are defined as its gaps, not by asking whether
//                  an orbit is periodic, because in a curved plane none of them
//                  quite is (see OuterBilliardsShared.h).
//   THE WEB        the singularity graph: every point that lands on a ray where
//                  the map is undefined, within some number of hops. This is the
//                  fractal, and it is rendered per-pixel on the GPU.
//   THE RAYS       just the n rays themselves - the first layer of the web, drawn
//                  crisply as vectors, for pointing at.
//   THE TABLE      the polygon, and orbits hopping around it.
//
// Everything that moves is a state variable, so any of it can be animated with
// manager.transition() - and the ones the framework is built around have
// wrappers:
//
//   THE TABLE      set_shape() / set_regular_polygon() install a polygon and
//                  publish "v0.x", "v0.y", "v1.x", ... one pair per vertex.
//                  move_vertex() animates one corner, morph_shape() animates all
//                  of them at once into another polygon with the same number of
//                  corners. Everything else is recomputed from the vertices every
//                  frame, so the orbits AND the fractal reshape live as the table
//                  deforms.
//
//   THE START      add_orbit() publishes "<name>.x", "<name>.y" for a point to
//                  set off from, plus "<name>.opacity" and "<name>.iterations".
//                  move_start() animates where it sets off from. Any number of
//                  orbits can share the table; they all read the same vertices.
//
//   THE COUNT      "iterations" is how many hops to draw, and it is a REAL
//                  number: the whole part is how many hops are complete and the
//                  fraction draws the next one partway, so ramping it looks like
//                  the orbit being traced out rather than snapping hop to hop.
//                  Each orbit's "<name>.iterations" defaults to following it.
//
//   THE DEPTH      "singularity_depth" is how many preimages of the rays to
//                  draw, and it is real in the same way: layer k fades in as the
//                  depth crosses k, so grow_singularities() unfolds the fractal
//                  smoothly instead of a layer at a time. This is the expensive
//                  knob - cost is linear in it - but 200 layers at 1080p is
//                  still a few milliseconds.
//
//   THE PLANE      "curvature" is 0 for the Euclidean plane and negative for the
//                  hyperbolic plane of that curvature, drawn in the Beltrami-
//                  Klein model - where geodesics are straight chords, so every
//                  layer above keeps working unchanged and only the reflection
//                  and the distances differ. Animate it with bend_space() and
//                  the plane curls into a disk while the fractal reorganizes
//                  into a hyperbolic tiling.
//
// Other state variables, all with sensible defaults: shape_opacity,
// shape_fill_opacity, vertex_dot_size, orbit_opacity, dot_size, line_thickness,
// orbit_fade (dim the older hops), pivot_opacity (mark the vertex each hop turns
// about), rainbow and rainbow_period (tint hops by age), ray_opacity,
// horizon_opacity, singularity_opacity, singularity_width, singularity_glow,
// singularity_fade,
// singularity_rainbow, singularity_rainbow_period, island_opacity,
// island_max_period, island_period_scale. Panning and zooming come from
// CoordinateScene: center_x, center_y, zoom - or use frame_view(), which works in
// world units.
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
    bool has_orbit(const std::string& name) const;

    void move_start(const TransitionType tt, const std::string& name, const vec2& to, bool smooth = true);
    void move_start(const TransitionType tt, const vec2& to, bool smooth = true);   // the default orbit
    vec2 start_of(const std::string& name) const;                                   // where it is headed

    void fade_orbit(const TransitionType tt, const std::string& name, double opacity);
    void set_orbit_color(const std::string& name, uint32_t color);

    static std::string orbit_var(const std::string& name, const std::string& axis);

    // --- how many hops ---------------------------------------------------
    void set_iterations(double count);
    void iterate_to(const TransitionType tt, double count, bool smooth = true);
    // Give one orbit its own hop count, or hand it back to the global one.
    void iterate_orbit_to(const TransitionType tt, const std::string& name, double count, bool smooth = true);
    void follow_global_iterations(const TransitionType tt, const std::string& name);

    // --- the singularity graph -------------------------------------------
    // Show the n rays where the map itself is undefined - the seed the whole
    // fractal is grown from.
    void fade_rays(const TransitionType tt, double opacity, bool smooth = true);

    // Set the depth outright, or unfold the web to it. Depth is real; see the
    // header comment. grow_singularities() also turns the layer on if it is off,
    // so it is usually the only call a shot needs.
    void set_singularity_depth(double depth);
    void grow_singularities(const TransitionType tt, double depth, bool smooth = true);
    void fade_singularities(const TransitionType tt, double opacity, bool smooth = true);

    // The regions the web never reaches, filled by period. Cost is linear in
    // island_max_period, which is why that is a separate knob from the depth -
    // but 0 (the default) sizes it from the shot, which is what you want unless
    // you are deliberately trading completeness for speed.
    void fade_islands(const TransitionType tt, double opacity, bool smooth = true);
    void set_island_max_period(int max_period);

    // --- where every point goes ------------------------------------------
    // Paints the plane with a color wheel and then recolors each pixel with the
    // color of wherever ITS orbit has got to. Nothing moves; only the coloring
    // does. Order shows as patches that keep their color, chaos as noise, and an
    // escaping orbit as a fade to black - the last of which is the state variable
    // flow_shade_by_distance (default on); turn it off and the wheel is hue
    // alone, full strength everywhere, which is order and chaos with nothing else
    // competing for the eye. See FlowFieldParams in
    // Host_Device_Shared/OuterBilliardsShared.h.
    //
    // flow_to() is the whole show: ramp it linearly and watch the plane sort
    // itself out. Cost is linear in the count, as everywhere else here.
    //
    // RATE MATTERS. A hop is a half turn, so the coloring is completely
    // rearranged once per unit of flow_iterations. Ramp it faster than about
    // four or five per second and consecutive frames stop being related to each
    // other; ramp it slower and the whole thing reads as one continuous motion.
    // Very little is lost by going slowly, because the picture converges early:
    // measured on a pentagon, 100 hops and 3200 hops differ by about one part in
    // a hundred. Deep is cheap to ask for and buys almost nothing.
    void fade_flow(const TransitionType tt, double opacity, bool smooth = true);
    void set_flow_iterations(double count);
    void flow_to(const TransitionType tt, double count, bool smooth = false);
    // True turns continuously through each hop, so the coloring flows; false
    // holds every whole iterate until the next one lands, so the hops are
    // countable but the picture jumps. They agree exactly on the whole numbers.
    // Continuous is the default and is what you want for anything but a slow
    // count-the-hops beat - see outer_billiards_turn for what "partway through a
    // hop" has to mean for this to be smooth at all.
    void set_flow_continuous(bool continuous);

    // Off by default, for backward compatibility with shots that pick
    // flow_iterations by hand for pacing (see OrderAndChaos.cpp). On, the
    // iteration count actually sent to the GPU is flow_iterations PLUS however
    // much more the CURRENT view needs on top of it: panning out toward the
    // corners of the shot, or zooming in past the resolution flow_iterations was
    // tuned for, both call for more hops before the coloring is trustworthy at
    // that depth - see auto_flow_iterations(). Meant for interactive exploration
    // (open_ui()), where there is no fixed shot to have tuned flow_iterations
    // for in the first place.
    void set_flow_auto_depth(bool on);

    // --- which plane all of this lives in --------------------------------
    // 0 is Euclidean. Negative is the hyperbolic plane of that curvature, drawn
    // in the Beltrami-Klein model - so geodesics stay straight and the table, the
    // orbits and the fractal all keep their meaning. bend_space() animates it,
    // and brings the ideal boundary along with it.
    //
    // "poincare_view" (a plain state variable, default off) redraws the same
    // plane in the Poincare disk model instead - geodesics bow into arcs, which
    // is the more familiar hyperbolic picture - without touching the map, the
    // metric or the ideal boundary, which stays the same shared circle either
    // way. See OuterBilliardsShared.h.
    void set_curvature(double curvature);
    void bend_space(const TransitionType tt, double curvature, bool smooth = true);

    // The two ways of naming the same thing: the ideal boundary's radius, and the
    // curvature that puts it there. Handy for sizing a shot - bend_space() takes
    // a curvature, but "I want the whole plane to fit in a disk of radius 3" is
    // usually the thought.
    static double horizon_for(double curvature);
    static double curvature_for_horizon(double horizon);

    // --- framing ---------------------------------------------------------
    // half_height is how far above the center of the panel you want to see, in
    // world units. (CoordinateScene's zoom is logarithmic and inverted; this is
    // the same thing said in units the shape is measured in.)
    void frame_view(const vec2& center, float half_height);
    void frame_view(const TransitionType tt, const vec2& center, float half_height, bool smooth = true);

    // --- appearance (not animatable; opacities are, above) ---------------
    uint32_t background_color = OPAQUE_BLACK;
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
