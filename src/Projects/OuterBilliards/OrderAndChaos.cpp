#include "../Scenes/Math/OuterBilliardsScene.h"

// ---------------------------------------------------------------------------
// Order, chaos, and escape in outer billiards - by coloring the plane with where
// every point GOES rather than by drawing any one orbit.
//
//     .\go.ps1 OrderAndChaos 1920 1080 30 -n
//
// Paint the exterior with a color wheel: hue from the direction, white at the
// middle, fading to black far out. Then leave every pixel exactly where it is
// and give it the color of the place its own orbit has reached after n hops.
// Nothing on screen moves. Only the coloring does.
//
//   ORDER    On a periodic island the n'th iterate is a rigid motion, so the
//            wheel is carried around intact. The patch keeps its shape and its
//            colors however far n runs - it just turns.
//   CHAOS    Where the singular rays keep cutting a region into finer and finer
//            pieces, each piece is carried off somewhere unrelated, and the
//            coloring breaks up into a mosaic and then into grain.
//   ESCAPE   An orbit on its way to infinity runs off the bright end of the
//            wheel and fades to black.
//
// ---------------------------------------------------------------------------
// TWO THINGS THAT WERE MEASURED, because both are counterintuitive and both
// decide how this file is written.
// ---------------------------------------------------------------------------
//
// 1. THE HOP COUNT BARELY MATTERS. Past a couple of hundred hops the picture
//    stops changing. Sweeping a pentagon from 400 to 3200 moved the difference
//    between neighbouring pixels by about one part in a hundred; the same sweep
//    on a generic quadrilateral moved it by four parts in a thousand. Outer
//    billiards is a piecewise ISOMETRY - neighbouring points never separate
//    exponentially, only when a singular ray passes between them - so the
//    coloring converges instead of degenerating. Running deeper costs time
//    linearly and buys nothing. Hence the modest targets below.
//
//    What DOES decide how broken-up it looks:
//
// 2. THE TABLE, and then the FRAMING. Measured as a fraction of the difference
//    a genuine white-noise image of the same palette would show between
//    neighbouring pixels:
//
//                        framed at 6      framed at 25
//        regular pentagon     3-5%             9-12%
//        regular heptagon     7%               15%
//        kite                 7-8%             30%
//        generic quadrilateral 10%             40%
//
//    A regular polygon is quasi-rational: almost every point of the plane is
//    periodic, and the periodic islands are enormous - the pentagon's picture is
//    a packing of giant flat disks with a thin fractal gasket between them. It
//    cannot look like noise, at any depth, because there is almost nothing there
//    to be noisy. Take the table off the lattice and off the regular polygons
//    and the islands go away; pull the camera back and what is left goes to
//    grain, because the mosaic gets finer than a pixel. Those two levers, not
//    the iteration count, are what this file uses.
//
// ---------------------------------------------------------------------------
// RATE. A hop is a half turn about a vertex, so one unit of flow_iterations
// completely rearranges the coloring. Ramped fast, consecutive frames are
// unrelated and the whole thing strobes. Everything here is held at 3 hops per
// second or below, and the fraction of a hop is spent TURNING (see
// outer_billiards_turn) rather than sliding toward the destination - sliding
// would put every point of a wedge on its own pivot at the halfway mark and
// flatten the screen to one color per wedge, twice per hop.
//
// The knobs, both on the scene:
//   flow_continuous  1 turns through each hop so the coloring flows; 0 holds
//                    every whole iterate, which is countable but jumps.
//   flow_samples     1 leaves the mosaic as raw per-pixel grain, which is the
//                    honest picture. Raise it and the grain averages to grey -
//                    also a real signal, and much kinder to a video encoder.
// ---------------------------------------------------------------------------

// The kite K(A) Schwartz works with: vertices (-1,0), (0,1), (A,0), (0,-1).
// Any irrational A in (0,1) has unbounded orbits; this is one.
static const float KITE_A = 0.41421356f;   // sqrt(2) - 1

static std::vector<vec2> kite(float a) {
    return {vec2(-1, 0), vec2(0, -1), vec2(a, 0), vec2(0, 1)};
}

// A quadrilateral with nothing going for it: corners at no particular angles and
// no particular radii, so it is neither a lattice polygon nor a regular one nor
// an affine image of either. That is the entire specification, and it is what
// makes the third act look the way it does.
static std::vector<vec2> generic_quad() {
    const float angle[4]  = {0.00f, 1.51f, 2.97f, 4.44f};
    const float radius[4] = {1.00f, 0.93f, 1.07f, 0.97f};
    std::vector<vec2> v;
    for (int i = 0; i < 4; i++) v.push_back(vec2(radius[i] * cosf(angle[i]), radius[i] * sinf(angle[i])));
    return v;
}

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
