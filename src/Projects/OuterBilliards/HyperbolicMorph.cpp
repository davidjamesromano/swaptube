#include "../Scenes/Math/OuterBilliardsScene.h"
#include "../IO/LivePlayer.h"
#include "../Core/State/StateTester.h"
#include "../Core/Smoketest.h"

// ---------------------------------------------------------------------------
// Just the shape: a triangle morphing into a square, entirely inside the
// hyperbolic plane, colored by FlowGraph.cpp's map instead of the singularity
// graph. No flat-plane act first - curvature is set once, up front, and never
// touched again.
//
// poincare_view is turned on, so the geodesics bow into arcs (the Poincare
// disk picture) instead of staying Klein's straight chords - the map itself
// is still computed in Klein coordinates underneath; only where things get
// drawn changes. See OuterBilliardsShared.h's KLEIN <-> POINCARE section.
//
//     .\go.ps1 HyperbolicMorph 1920 1080 30 -n
//
// The triangle is built as a 4-vertex polygon with one pair of coincident
// corners (v0 = v1), so morph_shape() has a real quadrilateral - the same
// vertex count as the square it grows into - to interpolate toward.
//
// Curvature and framing are pinned to match the point in FlowGraph.cpp where
// it bends hyperbolic (its step 3): the same ideal-boundary-radius-3.2 bend
// and the same 3.7 half-height framing, rather than the tighter
// curvature_for_horizon(2.0)/2.4 framing FlowGraph only reaches afterward.
// flow_shade_by_distance is turned off, unlike FlowGraph, so the wheel is hue
// alone - no white middle, no fade to black - leaving the coloring reading
// only ORDER and CHAOS with nothing about distance from the table competing
// for the eye.
// ---------------------------------------------------------------------------

static const double FLOW_ITERATIONS = 20000;

// How many hops of the single tracked orbit to draw. Tune this and re-render
// to taste - rainbow is off by default, so every hop stays the same solid
// orbit_color regardless of how many are drawn.
static const double ORBIT_ITERATIONS = 200;

// Same disk as FlowGraph.cpp/SingularityGraph.cpp: an ideal boundary of
// radius 3.2, leaving the unit table comfortably inside it.
static const double BENT = -1.0 / (3.2 * 3.2);

void render_video() {
    OuterBilliardsScene bs;

    bs.manager.set("shape_fill_opacity", "0");
    bs.table_color = 0xff000000;

    const std::vector<vec2> corners = OuterBilliards::regular_polygon(3, 1.0f, M_PI / 10);
    std::vector<vec2> triangle(4);
    triangle[0] = triangle[1] = corners[1];
    triangle[2] = corners[2];
    triangle[3] = corners[0];
    bs.set_shape(triangle);

    bs.set_curvature(BENT);
    bs.manager.set("horizon_opacity", "1");
    bs.manager.set("poincare_view", "1");   // bow the geodesics into arcs instead of Klein's straight chords
    bs.frame_view(vec2(0, 0), 3.7f);

    bs.set_flow_iterations(FLOW_ITERATIONS);
    bs.manager.set("flow_shade_by_distance", "0");

    // A single tracked point, drawn as one solid-colored path so its shape can
    // be read at a glance as the table morphs underneath it. The map is
    // recomputed from the table's current vertices every frame, so this orbit
    // reshapes live right along with the triangle-to-square transition.
    bs.add_orbit("tracked", vec2(1, 1));
    bs.set_iterations(ORBIT_ITERATIONS);

    // FLOW_ITERATIONS is tuned for the framing set below, not for wherever
    // open_ui() ends up panning/zooming to - so let the view itself add depth
    // on top of it once it reaches or resolves past that framing. See
    // set_flow_auto_depth().
    bs.set_flow_auto_depth(true);

    stage_macroblock(SilenceBlock(0.2), 1);
    bs.fade_flow(MICRO, 1.0);
    bs.render_microblock();

    // Padding: sit on the triangle before anything moves.
    stage_macroblock(SilenceBlock(0.2), 1);
    bs.render_microblock();

    // Drop into the interactive state tester right here, so the morph itself
    // can be poked at and restarted from any state before committing to the
    // scripted clip below.
    if (!is_smoketest()) open_ui(bs);

    stage_macroblock(SilenceBlock(30), 1);
    bs.morph_shape(MICRO, OuterBilliards::regular_polygon(4, 1.0f, M_PI / 2));
    bs.render_microblock();

    // Padding: sit on the square at the end.
    stage_macroblock(SilenceBlock(1), 1);
    bs.render_microblock();
}
