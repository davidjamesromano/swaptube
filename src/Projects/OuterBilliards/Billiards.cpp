#include "../Scenes/Math/OuterBilliardsScene.h"
#include "../Scenes/Common/CompositeScene.h"
#include "../Scenes/Media/LatexScene.h"

// ---------------------------------------------------------------------------
// OUTER BILLIARDS
//
// Pick a point outside a convex polygon. Look at the polygon and pick out the
// corner your line of sight just grazes - the one that leaves the whole polygon
// on your left. Reflect your point through that corner. Repeat forever.
//
// That is the entire rule, and it is enough to produce all of this:
//
//   Act 1  the rule itself, one hop at a time, about a square
//   Act 2  every orbit of a square closes up - period 4 near in, 8 further out
//   Act 3  nudge one corner and the closing stops. The orbit wanders off over a
//          region hundreds of times the size of the table it is orbiting
//   Act 4  a regular pentagon, where a whole family of orbits weaves a web
//
// The framework this drives is in Scenes/Math/OuterBilliardsScene.h; the map
// itself is in DataObjects/OuterBilliards.h. Everything animated below is a
// state variable, so any of it can be transitioned: the table's corners, where
// each orbit sets off from, and how many hops deep to draw.
// ---------------------------------------------------------------------------

// Captions need the LaTeX toolchain. Set to 0 for a picture-only render.
#define SHOW_CAPTIONS 0

static const float SQUARE_RADIUS = 1.0f;

// The point Act 1 sets off from. Its orbit about the square below has period 4.
static const vec2 FIRST_START(1.9f, 0.55f);

static uint32_t main_orbit_color(int index, int total) {
    return HSVtoRGB(0.02 + 0.80 * (double)index / (double)total, 0.72, 1.0);
}

void render_video() {
    CompositeScene cs;

    shared_ptr<OuterBilliardsScene> bs = make_shared<OuterBilliardsScene>();
    cs.add_scene(bs, "bs");

#if SHOW_CAPTIONS
    shared_ptr<LatexScene> caption = make_shared<LatexScene>(
        "\\text{Outer Billiards}", 0.9, vec2(0.9, 0.12));
    cs.add_scene(caption, "caption", vec2(0.5, 0.88));
#endif
    auto say = [&](const string& tex) {
#if SHOW_CAPTIONS
        caption->begin_latex_transition(MICRO, tex);
#else
        (void)tex;
#endif
    };
    auto hold = [&](double seconds) {
        stage_macroblock(SilenceBlock(seconds), 1);
        cs.render_microblock();
    };

    // =====================================================================
    // ACT 1 - the rule
    // =====================================================================
    bs->set_regular_polygon(4, SQUARE_RADIUS, M_PI / 4);   // corners up, down, left, right
    bs->frame_view(vec2(0, 0), 3.0f);
    bs->set_iterations(0);
    bs->manager.set("pivot_opacity", "1");   // mark the corner each hop turns about

    hold(2.0);

    // The point we set off from. Nothing has hopped yet, so this is just a dot.
    bs->add_orbit("p", FIRST_START);
    bs->manager.set("orbit_opacity", "0");
    stage_macroblock(SilenceBlock(1.5), 1);
    bs->manager.transition(MICRO, "orbit_opacity", "1");
    say("\\text{Start at a point outside.}");
    cs.render_microblock();

    // One hop. The segment's midpoint IS the corner it turned about, so the
    // construction draws itself.
    stage_macroblock(SilenceBlock(2.5), 1);
    bs->iterate_to(MICRO, 1);
    say("\\text{Reflect it through the corner you can just see past.}");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(2.0), 1);
    say("T(p) = 2v - p");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(2.0), 1);
    bs->iterate_to(MICRO, 2);
    say("\\text{Then do it again.}");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3.0), 1);
    bs->iterate_to(MICRO, 4);
    cs.render_microblock();

    stage_macroblock(SilenceBlock(2.5), 1);
    say("\\text{Four hops, and it is back where it started.}");
    cs.render_microblock();

    // =====================================================================
    // ACT 2 - every orbit of a square closes
    // =====================================================================
    stage_macroblock(SilenceBlock(4.0), 1);
    bs->manager.transition(MICRO, "pivot_opacity", "0");
    bs->move_start(MICRO, vec2(2.62f, 0.34f));   // out past the period-4 band
    bs->iterate_to(MICRO, 8);
    bs->frame_view(MICRO, vec2(0, 0), 3.8f);
    say("\\text{Set off further out, and it takes eight.}");
    cs.render_microblock();

    hold(2.0);

    // A whole family of starting points at once, each one closing into its own
    // necklace around the table.
    const int FAMILY = 14;
    for (int i = 0; i < FAMILY; i++) {
        const string name = "q" + to_string(i);
        bs->add_orbit(name, vec2(1.12f + 0.115f * i, 0.31f), main_orbit_color(i, FAMILY));
        bs->manager.set(OuterBilliardsScene::orbit_var(name, "opacity"), "0");
    }
    stage_macroblock(SilenceBlock(3.5), 1);
    for (int i = 0; i < FAMILY; i++) bs->fade_orbit(MICRO, "q" + to_string(i), 1);
    bs->manager.transition(MICRO, "dot_size", "0.55");
    say("\\text{Every starting point closes up.}");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(4.0), 1);
    bs->iterate_to(MICRO, 12);
    bs->frame_view(MICRO, vec2(0, 0), 4.4f);
    cs.render_microblock();

    hold(2.5);

    // =====================================================================
    // ACT 3 - move one corner, and the closing stops
    // =====================================================================
    stage_macroblock(SilenceBlock(2.0), 1);
    for (int i = 0; i < FAMILY; i++) bs->fade_orbit(MICRO, "q" + to_string(i), 0);
    say("\\text{Now move one corner of the table.}");
    cs.render_microblock();
    for (int i = 0; i < FAMILY; i++) bs->remove_orbit("q" + to_string(i));

    stage_macroblock(SilenceBlock(3.5), 1);
    bs->manager.transition(MICRO, "dot_size", "1");
    bs->move_start(MICRO, FIRST_START);
    bs->iterate_to(MICRO, 8);
    cs.render_microblock();

    // A nudge of a tenth of the table's radius is all it takes.
    stage_macroblock(SilenceBlock(3.5), 1);
    bs->move_vertex(MICRO, 0, vec2(0.707f + 0.09f, 0.707f + 0.06f));
    cs.render_microblock();

    stage_macroblock(SilenceBlock(5.0), 1);
    bs->iterate_to(MICRO, 40);
    bs->frame_view(MICRO, vec2(0, 0), 11.0f);
    bs->manager.transition(MICRO, "dot_size", "0.5");
    say("\\text{It never comes home again.}");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(6.0), 1);
    bs->iterate_to(MICRO, 170);
    bs->frame_view(MICRO, vec2(0, 0), 27.0f);
    bs->manager.transition(MICRO, "dot_size", "0");
    bs->manager.transition(MICRO, "line_thickness", "0.55");
    bs->manager.transition(MICRO, "rainbow", "0.85");
    bs->manager.transition(MICRO, "rainbow_period", "34");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(3.0), 1);
    say("\\text{One orbit, wandering a region 30 times the table.}");
    cs.render_microblock();

    // =====================================================================
    // ACT 4 - the pentagon
    // =====================================================================
    stage_macroblock(SilenceBlock(2.0), 1);
    bs->manager.transition(MICRO, "orbit_opacity", "0");
    bs->manager.transition(MICRO, "shape_opacity", "0");
    cs.render_microblock();

    bs->remove_orbit("p");
    bs->set_regular_polygon(5, 1.0f, M_PI / 2);
    bs->frame_view(vec2(0, 0), 3.0f);
    bs->set_iterations(0);
    bs->manager.set("rainbow", "0");
    bs->manager.set("dot_size", "0.8");
    bs->manager.set("line_thickness", "1");
    bs->add_orbit("p", vec2(1.62f, 0.28f));

    stage_macroblock(SilenceBlock(2.5), 1);
    bs->manager.transition(MICRO, "shape_opacity", "1");
    bs->manager.transition(MICRO, "orbit_opacity", "1");
    say("\\text{A regular pentagon.}");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(4.0), 1);
    bs->iterate_to(MICRO, 10);
    bs->frame_view(MICRO, vec2(0, 0), 5.6f);
    cs.render_microblock();

    // Every one of these closes too - but they close into necklaces that thread
    // through one another, and the union is the picture the pentagon is famous
    // for.
    const int WEB = 30;
    for (int i = 0; i < WEB; i++) {
        const string name = "w" + to_string(i);
        bs->add_orbit(name, vec2(1.13f + 0.072f * i, 0.17f), main_orbit_color(i, WEB));
        bs->manager.set(OuterBilliardsScene::orbit_var(name, "opacity"), "0");
    }
    stage_macroblock(SilenceBlock(4.0), 1);
    for (int i = 0; i < WEB; i++) bs->fade_orbit(MICRO, "w" + to_string(i), 0.85);
    bs->manager.transition(MICRO, "dot_size", "0.4");
    say("\\text{Every orbit weaves through the others.}");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(5.0), 1);
    bs->iterate_to(MICRO, 40);
    bs->manager.transition(MICRO, "line_thickness", "0.7");
    cs.render_microblock();

    stage_macroblock(SilenceBlock(4.0), 1);
    bs->frame_view(MICRO, vec2(0, 0), 6.4f);
    bs->manager.transition(MICRO, "dot_size", "0");
    say("\\text{Outer Billiards}");
    cs.render_microblock();

    hold(2.5);
}
