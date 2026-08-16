#include "../Scenes/Math/OuterBilliardsScene.h"
#include "OuterBilliards/OuterBilliardsTables.h"

// ---------------------------------------------------------------------------
// Order, chaos, and escape in outer billiards - by coloring the plane with where
// every point GOES rather than by drawing any one orbit.
//
//     .\go.ps1 OrderAndChaos 1920 1080 30 -n
//
// Paint the exterior with a color wheel: hue from direction, white at the
// middle, fading to black far out. Then leave every pixel exactly where it is
// and give it the color of the place its own orbit has reached after n hops.
//
//   ORDER    a periodic island's n'th iterate is a rigid motion, so the wheel
//            is carried around intact and the patch stays coherent forever.
//   CHAOS    where singular rays keep cutting a region finer, neighbouring
//            pieces are carried off unrelated, and the coloring turns to grain.
//   ESCAPE   an orbit headed to infinity runs off the bright end of the wheel
//            and fades to black.
//
// TWO MEASURED FACTS decide how this file is written:
//
// 1. THE HOP COUNT BARELY MATTERS past a couple hundred hops - outer billiards
//    is a piecewise ISOMETRY, so neighbouring points only separate when a
//    singular ray passes between them, and the coloring converges rather than
//    degenerating. Running deeper costs time linearly and buys little.
// 2. THE TABLE, then the FRAMING, are what decide how broken-up it looks. A
//    regular polygon is quasi-rational - almost every point is periodic, in
//    enormous islands - so it cannot look like noise at any depth; take the
//    table off the lattice and the islands go away, and pulling the camera
//    back turns the remaining mosaic to grain once it gets finer than a pixel.
//
// RATE: a hop is a half turn, so one unit of flow_iterations completely
// rearranges the coloring. Everything here stays at 3 hops/second or below,
// or consecutive frames stop looking related.
// ---------------------------------------------------------------------------

void render_video() {
    OuterBilliardsScene bs;

    // The table itself is drawn flat black over the coloring - enough to see what
    // the orbits are turning about, not enough to fight with it.
    bs.manager.set("shape_fill_opacity", "0");
    bs.table_color = 0xff000000;

    // -----------------------------------------------------------------------
    // 1. THE WHEEL ITSELF, and a square. A lattice polygon: every orbit closes,
    //    4 hops near in and 8 further out, so the coloring shuffles through a
    //    cycle and comes back to itself forever.
    // -----------------------------------------------------------------------
    bs.frame_view(vec2(0, 0), 6.0f);
    bs.set_regular_polygon(4, 1.0f, M_PI / 4);

    stage_macroblock(SilenceBlock(4), 1);
    bs.fade_flow(MICRO, 1.0);
    bs.render_microblock();

    // Deliberately slower than the rest - one and a half hops a second - so that
    // the individual half turns are watchable rather than merely smooth.
    stage_macroblock(SilenceBlock(12), 1);
    bs.flow_to(MICRO, 18);
    bs.render_microblock();

    // Up to the working rate, and let it run. It never degrades: this is what
    // perfect order looks like under this coloring.
    stage_macroblock(SilenceBlock(10), 1);
    bs.flow_to(MICRO, 48);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(3), 1);
    bs.render_microblock();

    // -----------------------------------------------------------------------
    // 2. A PENTAGON. Off the lattice, but still regular - so still almost all
    //    order, arranged at every scale at once: a packing of huge flat islands
    //    with a fractal gasket threaded between them.
    // -----------------------------------------------------------------------
    stage_macroblock(SilenceBlock(2), 1);
    bs.fade_flow(MICRO, 0.0);
    bs.render_microblock();

    bs.set_shape(OuterBilliards::regular_polygon(5, 1.0f, M_PI / 2));
    bs.set_flow_iterations(0);

    stage_macroblock(SilenceBlock(3), 1);
    bs.fade_flow(MICRO, 1.0);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(24), 1);
    bs.flow_to(MICRO, 72);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(4), 1);
    bs.render_microblock();

    // And here is why the islands are islands: they are exactly the regions the
    // singularity graph never reaches. The gasket between them is exactly where
    // it is dense. (That web is what SingularityGraph.cpp is about.)
    //
    // The coloring is dimmed to make room for it. At full strength the web is
    // orange hairlines over a plane that is already every color at once, and the
    // one thing worth seeing - that the lines stop dead at every island edge and
    // never once cross one - is lost in it.
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
    // 3. A GENERIC QUADRILATERAL. Nothing regular about it, and the big islands
    //    simply are not there: the plane is mosaic all the way down. Then pull
    //    the camera back, which is the other half of it - the mosaic is finite
    //    everywhere, so making the pixels bigger than the tiles is what turns it
    //    into grain.
    // -----------------------------------------------------------------------
    stage_macroblock(SilenceBlock(2), 1);
    bs.fade_flow(MICRO, 0.0);
    bs.render_microblock();

    bs.set_shape(generic_quad());
    bs.set_flow_iterations(0);
    bs.frame_view(vec2(0, 0), 6.0f);
    bs.manager.set("flow_scale", "3");   // pin it, so the pull-back can animate it

    stage_macroblock(SilenceBlock(3), 1);
    bs.fade_flow(MICRO, 1.0);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(24), 1);
    bs.flow_to(MICRO, 72);
    bs.render_microblock();

    // The wheel opens up along with the shot. Hue still means direction and
    // brightness still means distance; this only moves the radius at which the
    // brightness runs out, which otherwise sits just off the middle of the wide
    // frame and leaves the whole outside black.
    stage_macroblock(SilenceBlock(12), 1);
    bs.frame_view(MICRO, vec2(0, 0), 25.0f);
    bs.manager.transition(MICRO, "flow_scale", "9");
    bs.render_microblock();

    stage_macroblock(SilenceBlock(4), 1);
    bs.render_microblock();

    // -----------------------------------------------------------------------
    // 4. A KITE, whose orbits are not all bounded - the case Schwartz settled.
    //    Same pull-back, and this time the plane darkens as it goes, because the
    //    orbits that are leaving have run off the bright end of the wheel.
    // -----------------------------------------------------------------------
    stage_macroblock(SilenceBlock(2), 1);
    bs.fade_flow(MICRO, 0.0);
    bs.render_microblock();

    bs.set_shape(kite(KITE_A));
    bs.set_flow_iterations(0);
    bs.frame_view(vec2(0, 0), 6.0f);
    bs.manager.set("flow_scale", "3");

    stage_macroblock(SilenceBlock(3), 1);
    bs.fade_flow(MICRO, 1.0);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(16), 1);
    bs.flow_to(MICRO, 48);
    bs.render_microblock();

    // Worth knowing how to read this one: a pixel is dark because its ORBIT
    // ended up far out, not because the pixel is. So the ring of darkness that
    // arrives with the wide shot is mostly just distant points staying distant -
    // the actual escapes are the dark specks scattered through the bright middle,
    // points that started next to the table and did not stay there.
    // Opened up less than the last act's, on purpose: this one wants some
    // falloff left in the frame, or the escapes have no dark to show up as.
    stage_macroblock(SilenceBlock(10), 1);
    bs.frame_view(MICRO, vec2(0, 0), 25.0f);
    bs.manager.transition(MICRO, "flow_scale", "6");
    bs.render_microblock();

    stage_macroblock(SilenceBlock(5), 1);
    bs.render_microblock();
}
