#include "../Scenes/Math/OuterBilliardsScene.h"
#include "OuterBilliards/OuterBilliardsTables.h"

// ---------------------------------------------------------------------------
// OrderAndChaos.cpp, run entirely in the hyperbolic plane instead of the flat
// one - same four tables, same color-wheel-carried-by-the-map idea, but drawn
// in the Beltrami-Klein model throughout (see OuterBilliardsShared.h for what
// changes under curvature and what does not).
//
//     .\go.ps1 OrderAndChaosHyperbolic 1920 1080 30 -n
//
// ONE THING THAT DOES NOT PORT OVER: in the flat film, pulling the camera back
// reveals finer mosaic. Here the whole plane already sits inside the horizon
// disk at a fixed screen radius, so pulling back shows nothing new. flow_scale
// does the same job instead - measured in the plane's own curved metric, not
// screen pixels, so growing it sweeps the color wheel's white middle out
// toward the horizon without the frame ever moving. However large scale gets,
// curved_distance still blows up approaching the boundary, so there is always
// a thin black ring at the true edge - a landmark the flat picture never had.
//
// THE KITE appears as what it visibly is - an off-lattice, non-regular
// quadrilateral - without claiming Schwartz's flat-plane unbounded-orbit
// result survives into a curved one.
// ---------------------------------------------------------------------------

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
