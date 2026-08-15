#include "../Scenes/Math/OuterBilliardsScene.h"

// ---------------------------------------------------------------------------
// The same shape arc as SingularityGraph.cpp - pentagon collapsing to
// triangle, then the plane itself bending hyperbolic, then the pentagon
// growing back out of the curved triangle - but read through OrderAndChaos.cpp's
// coloring instead: every point of the plane painted with the color of where
// its own orbit has gone, rather than with the fractal of where the map
// breaks.
//
//     .\go.ps1 FlowGraph 1920 1080 30 -n
//
// The iteration count is set ONCE, to 5000, and never touched again - see
// OrderAndChaos.cpp for why depth barely matters past a couple hundred hops.
// Nothing here ramps it. All the motion is the TABLE: as it morphs, and later
// as the curvature itself changes, the map underneath is a different map every
// frame, and the coloring reorganizes live to match it. Watching the pentagon
// fold into a triangle is watching a mosaic of countless small islands snap
// into one coherent flat tiling in real time - the same fact SingularityGraph
// shows as a fractal collapsing to nothing, said the other way.
//
// flow_scale is left at its default (0, meaning "3 table-radii out") for the
// same reason: the table's circumradius already shrinks and grows with the
// morph, so the color wheel stays correctly calibrated to the shape without
// anything here having to track it by hand.
// ---------------------------------------------------------------------------

static const double FLOW_ITERATIONS = 5000;

// Same disk as SingularityGraph.cpp: an ideal boundary of radius 3.2, leaving
// the unit table comfortably inside it.
static const double BENT = -1.0 / (3.2 * 3.2);

void render_video() {
    OuterBilliardsScene bs;

    // The table is drawn as a bare outline, so it marks what the orbits are
    // turning about without ever competing with the coloring underneath it.
    bs.manager.set("shape_fill_opacity", "0");
    bs.table_color = 0xff000000;

    bs.set_regular_polygon(5, 1.0f, M_PI / 2);
    bs.frame_view(vec2(0, 0), 4.5f);
    bs.set_flow_iterations(FLOW_ITERATIONS);

    // -----------------------------------------------------------------------
    // 1. The pentagon's picture: giant coherent islands - each one a patch
    //    where the map is a rigid motion, so the wheel just turns - packed
    //    around a mosaic gasket where it is not. That gasket is exactly the
    //    singularity graph's territory; here it shows up as grain instead of
    //    line work.
    // -----------------------------------------------------------------------
    stage_macroblock(SilenceBlock(4), 1);
    bs.fade_flow(MICRO, 1.0);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(6), 1);
    bs.render_microblock();

    // -----------------------------------------------------------------------
    // 2. Collapse to an EQUILATERAL triangle, same construction as
    //    SingularityGraph.cpp: pair {v0,v1} and {v2,v3}, leave v4 where it is,
    //    so the three survivors land at 18, 138 and 258 degrees. A lattice
    //    polygon is nothing but islands, so the mosaic should close up
    //    entirely by the time the triangle arrives.
    // -----------------------------------------------------------------------
    const std::vector<vec2> corners = OuterBilliards::regular_polygon(3, 1.0f, M_PI / 10);
    std::vector<vec2> triangle(5);
    triangle[0] = triangle[1] = corners[1];
    triangle[2] = triangle[3] = corners[2];
    triangle[4] = corners[0];

    stage_macroblock(SilenceBlock(15), 1);
    bs.morph_shape(MICRO, triangle);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(4), 1);
    bs.render_microblock();

    // -----------------------------------------------------------------------
    // 3. Bend the plane. Klein coordinates keep the table and every orbit's
    //    hops straight, so nothing here has to be redrawn - but the map is a
    //    different isometry under curvature, and the color wheel itself reads
    //    distance through the curved metric (see flow_wheel in
    //    outer_billiards_flow.cu), so the picture is free to reorganize too.
    // -----------------------------------------------------------------------
    stage_macroblock(SilenceBlock(15), 1);
    bs.frame_view(MICRO, vec2(0, 0), 3.7f);
    bs.bend_space(MICRO, BENT);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(5), 1);
    bs.render_microblock();

    // -----------------------------------------------------------------------
    // 4. Still curved, grow the triangle's doubled corners back out into the
    //    pentagon, then bend further so the whole disk comes into view -
    //    exactly SingularityGraph.cpp's finale, read as a color map instead
    //    of a graph.
    // -----------------------------------------------------------------------
    stage_macroblock(SilenceBlock(15), 1);
    bs.morph_shape(MICRO, OuterBilliards::regular_polygon(5, 1.0f, M_PI / 2));
    bs.render_microblock();

    stage_macroblock(SilenceBlock(15), 1);
    bs.bend_space(MICRO, OuterBilliardsScene::curvature_for_horizon(2.0));
    bs.frame_view(MICRO, vec2(0, 0), 2.4f);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(6), 1);
    bs.render_microblock();
}
