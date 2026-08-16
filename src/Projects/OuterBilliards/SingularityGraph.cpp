#include "../Scenes/Math/OuterBilliardsScene.h"

// ---------------------------------------------------------------------------
// Outer billiards, from one hop to the singularity graph, and out of the plane.
//
//     .\go.ps1 SingularityGraph 1920 1080 30 -n
//
// The arc:
//
//   1. ONE ORBIT. A point hops around a pentagon. Each hop is a straight segment
//      whose midpoint is the corner it turned about, so the picture shows its own
//      construction.
//   2. WHERE IT BREAKS. The map picks the corner the tangent line touches, and
//      which corner that is changes along n rays - the sides of the table
//      extended. Two starts a hundredth apart, one either side of a ray, are put
//      up together: their orbits agree about nothing after the first hop. ON the
//      ray the two answers are equally good and the map has none.
//   3. EVERY SUCH POINT. Not just the rays, but everything that eventually lands
//      on one: S_D = union over k = 0..D of T^-k(rays). Growing D unfolds the
//      fractal.
//   4. WHAT IT LEAVES. The holes are periodic islands - orbits that close up and
//      so never reach a ray. Zoom in; the web is self-similar.
//   5. A TRIANGLE. Slide two corners into their neighbors and the pentagon
//      degenerates into a triangle, whose graph is a plain tiling - every orbit
//      periodic, no fractal at all.
//   6. OUT OF THE PLANE. Bend the curvature negative and the whole plane curls
//      into a Klein disk. Geodesics stay straight, so nothing here has to be
//      redrawn, but the map is a different isometry and the triangle's flat
//      tiling reorganizes into a hyperbolic one.
//   7. BACK TO THE PENTAGON, still curved: its web now packs into the ideal
//      boundary instead of running off to infinity.
//
// The fractal is a distance field evaluated per pixel on the GPU - see
// src/CUDA/outer_billiards_singularity.cu - which is why the depth can be a real
// number that ramps, why the lines stay one pixel wide however far you zoom, and
// why the curvature can be animated at all.
// ---------------------------------------------------------------------------

// Cost is linear in the depth, and 240 layers at 1080p still lands in a few
// milliseconds, so the web gets room to be the thing everything is read against.
static const double FULL_DEPTH = 5000;

// The hyperbolic plane to fall into: an ideal boundary of radius 3.2, which
// leaves the unit table comfortably inside it with room to watch the tiling pack
// up against the edge.
static const double BENT = -1.0 / (3.2 * 3.2);

void render_video() {
    OuterBilliardsScene bs;

    // The pentagon is the interesting case. A square - or any polygon whose
    // corners sit on a lattice - has nothing but periodic orbits.
    bs.set_regular_polygon(5, 1.0f, M_PI / 2);
    bs.frame_view(vec2(0, 0), 4.5f);

    // -----------------------------------------------------------------------
    // 1. The map itself: one point, hopping. Each hop turns about the corner
    //    its tangent line touches, so the corner is worth marking at first.
    // -----------------------------------------------------------------------
    bs.add_orbit("a", vec2(2.35f, 0.9f));
    bs.manager.set("pivot_opacity", "1");

    stage_macroblock(SilenceBlock(5), 1);
    bs.iterate_to(MICRO, 9);
    bs.render_microblock();

    // Let it run. This orbit never closes and never escapes; it fills a band.
    stage_macroblock(SilenceBlock(5), 1);
    bs.manager.transition(MICRO, "pivot_opacity", "0");
    bs.iterate_to(MICRO, 260);
    bs.render_microblock();

    // -----------------------------------------------------------------------
    // 2. Where the map has no answer: the sides of the table, extended.
    // -----------------------------------------------------------------------
    stage_macroblock(SilenceBlock(4), 1);
    bs.iterate_to(MICRO, 0);
    bs.fade_rays(MICRO, 1.0);
    bs.render_microblock();

    // A point on a ray sees two corners at once and cannot choose between them.
    // Straddle it: two starts a fiftieth apart, one on each side.
    const std::vector<vec2> table = bs.shape();
    const vec2 along  = normalize(table[0] - table[1]);   // the ray leaving corner 0
    const vec2 across = vec2(-along.y, along.x);
    const vec2 on_ray = table[0] + along * 1.55f;

    bs.add_orbit("b", on_ray - across * 0.02f, 0xffff4090);
    bs.manager.set("b.opacity", "0");

    stage_macroblock(SilenceBlock(4), 2);
    bs.move_start(MICRO, "a", on_ray + across * 0.02f);
    bs.render_microblock();
    bs.fade_orbit(MICRO, "b", 1.0);
    bs.render_microblock();

    // Both at once. They part company on the very first hop and never meet again.
    stage_macroblock(SilenceBlock(6), 1);
    bs.iterate_to(MICRO, 30);
    bs.render_microblock();

    // -----------------------------------------------------------------------
    // 3. So: which points eventually LAND on a ray? Layer zero of the web is the
    //    rays themselves, so the handoff lands exactly on itself.
    // -----------------------------------------------------------------------
    stage_macroblock(SilenceBlock(4), 1);
    bs.fade_orbit(MICRO, "a", 0.0);
    bs.fade_orbit(MICRO, "b", 0.0);
    bs.manager.set("singularity_glow", "0.35");
    bs.grow_singularities(MICRO, 8);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(7), 1);
    bs.fade_rays(MICRO, 0.0);
    bs.grow_singularities(MICRO, FULL_DEPTH);
    bs.render_microblock();

    // -----------------------------------------------------------------------
    // 4. What the web leaves alone, and how it looks up close.
    // -----------------------------------------------------------------------
    stage_macroblock(SilenceBlock(4), 1);
    bs.fade_islands(MICRO, 0.55);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(7), 1);
    bs.frame_view(MICRO, vec2(1.35f, 2.05f), 0.5f);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(5), 1);
    bs.frame_view(MICRO, vec2(0, 0), 4.5f);
    bs.render_microblock();

    // -----------------------------------------------------------------------
    // 5. Collapse the pentagon to an EQUILATERAL triangle. Vertices are animated
    //    one to one, so the way to lose two corners is to slide them onto their
    //    neighbors - the collapsed sides stop generating rays as they go, and a
    //    corner sitting on top of another never wins the tangency search.
    //
    //    Pairing {v0,v1} and {v2,v3} and leaving v4 where it is puts the three
    //    surviving corners at 18, 138 and 258 degrees - equilateral, and each
    //    corner travels only a little way to get there.
    // -----------------------------------------------------------------------
    const std::vector<vec2> corners = OuterBilliards::regular_polygon(3, 1.0f, M_PI / 10);
    std::vector<vec2> triangle(5);
    triangle[0] = triangle[1] = corners[1];
    triangle[2] = triangle[3] = corners[2];
    triangle[4] = corners[0];

    // Islands stand down BEFORE anything moves, and are not asked back until it
    // has stopped. Halfway between two tables the shape is an irregular pentagon,
    // which genuinely has almost no periodic points; the few it has flash in and
    // out as the corners slide, and that reads as noise rather than as the
    // mathematics it is. Fading them out ACROSS the morph does not help - they
    // are still half-opaque for most of it, which is where the strobing lives.
    stage_macroblock(SilenceBlock(2), 1);
    bs.fade_islands(MICRO, 0.0);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(8), 1);
    bs.morph_shape(MICRO, triangle);
    bs.render_microblock();

    // Arrived: a flat tiling, every orbit periodic, no fractal left.
    stage_macroblock(SilenceBlock(4), 1);
    bs.fade_islands(MICRO, 0.7);
    bs.render_microblock();

    // -----------------------------------------------------------------------
    // 6. Bend the plane. Klein coordinates keep geodesics straight, so the table
    //    and the rays are still line segments - but the map is a different
    //    isometry now, and the tiling it generates is a hyperbolic one.
    //
    //    Bending counts as a morph for the islands' purposes: it is the MAP that
    //    is changing, and periods reorganize under it exactly as they do when a
    //    corner moves.
    // -----------------------------------------------------------------------
    stage_macroblock(SilenceBlock(2), 1);
    bs.fade_islands(MICRO, 0.0);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(9), 1);
    bs.frame_view(MICRO, vec2(0, 0), 3.7f);
    bs.bend_space(MICRO, BENT);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(4), 1);
    bs.fade_islands(MICRO, 0.5);
    bs.render_microblock();

    // -----------------------------------------------------------------------
    // 7. Still curved, grow the triangle's doubled corners back out, then bend
    //    further so the whole hyperbolic tiling comes into view at once. Two
    //    moves back to back, so the islands stay away across both of them and
    //    come back only at the end.
    // -----------------------------------------------------------------------
    stage_macroblock(SilenceBlock(2), 1);
    bs.fade_islands(MICRO, 0.0);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(9), 1);
    bs.morph_shape(MICRO, OuterBilliards::regular_polygon(5, 1.0f, M_PI / 2));
    bs.render_microblock();

    stage_macroblock(SilenceBlock(7), 1);
    bs.bend_space(MICRO, OuterBilliardsScene::curvature_for_horizon(2.0));
    bs.frame_view(MICRO, vec2(0, 0), 2.4f);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(5), 1);
    bs.fade_islands(MICRO, 0.4);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(3), 1);
    bs.render_microblock();
}
