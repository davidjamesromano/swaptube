#include "../Scenes/Math/OuterBilliardsScene.h"

// ---------------------------------------------------------------------------
// OrderAndChaos.cpp, run entirely in the hyperbolic plane instead of the flat
// one - same four tables, same color-wheel-carried-by-the-map idea, but drawn
// in the Beltrami-Klein model throughout. See OuterBilliardsShared.h for what
// changes under curvature (the reflection and the distances) and what does
// not (everything stays a straight line).
//
//     .\go.ps1 OrderAndChaosHyperbolic 1920 1080 30 -n
//
// ONE THING THAT DOES NOT PORT OVER: in the flat film, "pull the camera back"
// is what reveals finer mosaic, because further out literally is further out.
// Here the whole plane - all of it, no matter how far - already sits inside
// the horizon disk at a fixed screen radius; pulling back past that shows
// nothing new, there is nothing outside the circle to see. The lever that
// does the same job is flow_scale: it is measured in the plane's own curved
// metric (see flow_wheel in outer_billiards_flow.cu), not in screen pixels,
// so growing it sweeps the color wheel's white middle out toward the horizon
// exactly the way pulling the camera back swept it outward in the flat film -
// except the frame never has to move, because everything was always in it.
// And however large scale gets, curved_distance still blows up approaching
// the boundary, so there is always a thin black ring right at the true edge:
// the one landmark the flat picture never had.
//
// THE KITE. Schwartz's unbounded-orbit construction is a fact about the flat
// plane specifically; nothing here claims it survives into a curved one, so
// the kite appears as what it visibly is - an off-lattice, non-regular
// quadrilateral - without asserting anything about where its orbits end up.
// ---------------------------------------------------------------------------

static const float KITE_A = 0.41421356f;   // sqrt(2) - 1

static std::vector<vec2> kite(float a) {
    return {vec2(-1, 0), vec2(0, -1), vec2(a, 0), vec2(0, 1)};
}

static std::vector<vec2> generic_quad() {
    const float angle[4]  = {0.00f, 1.51f, 2.97f, 4.44f};
    const float radius[4] = {1.00f, 0.93f, 1.07f, 0.97f};
    std::vector<vec2> v;
    for (int i = 0; i < 4; i++) v.push_back(vec2(radius[i] * cosf(angle[i]), radius[i] * sinf(angle[i])));
    return v;
}

void render_video() {
    OuterBilliardsScene bs;

    // Curved from the first frame, and stays that way for the whole piece - a
    // horizon of 3.4 leaves every table below comfortably inside it, with room
    // for the wheel to still be sweeping when it reaches the edge.
    const double CURVATURE = OuterBilliardsScene::curvature_for_horizon(3.4);
    bs.set_curvature(CURVATURE);
    bs.manager.set("horizon_opacity", "1");

    // Fixed for the whole video: the horizon disk is everything there is to
    // see, and it never needs to move to reveal more of it.
    bs.frame_view(vec2(0, 0), 3.7f);

    bs.manager.set("shape_fill_opacity", "0");
    bs.table_color = 0xff000000;

    // -----------------------------------------------------------------------
    // 1. THE WHEEL ITSELF, and a square. Off the origin a rotation is the same
    //    isometry in Klein coordinates whatever the curvature, so a lattice
    //    polygon is still a lattice polygon and every orbit still closes.
    // -----------------------------------------------------------------------
    bs.set_regular_polygon(4, 1.0f, M_PI / 4);
    bs.manager.set("flow_scale", "1.4");

    stage_macroblock(SilenceBlock(4), 1);
    bs.fade_flow(MICRO, 1.0);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(12), 1);
    bs.flow_to(MICRO, 18);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(10), 1);
    bs.flow_to(MICRO, 48);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(3), 1);
    bs.render_microblock();

    // -----------------------------------------------------------------------
    // 2. A PENTAGON, still regular. Order at every scale: giant flat islands
    //    packing the disk, with a fractal gasket - the singularity graph -
    //    threaded between them, this time thinning out as it approaches the
    //    ideal boundary instead of running off to infinity in a straight line.
    // -----------------------------------------------------------------------
    stage_macroblock(SilenceBlock(2), 1);
    bs.fade_flow(MICRO, 0.0);
    bs.render_microblock();

    bs.set_shape(OuterBilliards::regular_polygon(5, 1.0f, M_PI / 2));
    bs.set_flow_iterations(0);
    bs.manager.set("flow_scale", "1.6");

    stage_macroblock(SilenceBlock(3), 1);
    bs.fade_flow(MICRO, 1.0);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(24), 1);
    bs.flow_to(MICRO, 72);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(4), 1);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(6), 1);
    bs.set_singularity_depth(200);
    bs.manager.set("singularity_glow", "0.3");
    bs.fade_singularities(MICRO, 0.9);
    bs.fade_flow(MICRO, 0.45);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(4), 1);
    bs.fade_singularities(MICRO, 0.0);
    bs.fade_flow(MICRO, 1.0);
    bs.render_microblock();

    // -----------------------------------------------------------------------
    // 3. A GENERIC QUADRILATERAL. No islands to speak of; the plane is mosaic
    //    all the way down. The frame never moves - instead flow_scale sweeps
    //    the wheel's white middle outward, all the way toward the horizon, to
    //    do the job the flat film did by pulling the camera back.
    // -----------------------------------------------------------------------
    stage_macroblock(SilenceBlock(2), 1);
    bs.fade_flow(MICRO, 0.0);
    bs.render_microblock();

    bs.set_shape(generic_quad());
    bs.set_flow_iterations(0);
    bs.manager.set("flow_scale", "0.8");

    stage_macroblock(SilenceBlock(3), 1);
    bs.fade_flow(MICRO, 1.0);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(24), 1);
    bs.flow_to(MICRO, 72);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(12), 1);
    bs.manager.transition(MICRO, "flow_scale", "5.0");
    bs.render_microblock();

    stage_macroblock(SilenceBlock(4), 1);
    bs.render_microblock();

    // -----------------------------------------------------------------------
    // 4. A KITE - off-lattice, off-regular, nothing symmetric about it. Same
    //    outward sweep as the quadrilateral, smaller, so there is still some
    //    darkness left in frame near the horizon for the fine structure there
    //    to show up against.
    // -----------------------------------------------------------------------
    stage_macroblock(SilenceBlock(2), 1);
    bs.fade_flow(MICRO, 0.0);
    bs.render_microblock();

    bs.set_shape(kite(KITE_A));
    bs.set_flow_iterations(0);
    bs.manager.set("flow_scale", "0.8");

    stage_macroblock(SilenceBlock(3), 1);
    bs.fade_flow(MICRO, 1.0);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(16), 1);
    bs.flow_to(MICRO, 48);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(10), 1);
    bs.manager.transition(MICRO, "flow_scale", "2.6");
    bs.render_microblock();

    stage_macroblock(SilenceBlock(5), 1);
    bs.render_microblock();
}
