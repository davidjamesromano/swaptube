#include "../Scenes/Math/OuterBilliardsScene.h"
#include "../Core/State/StateTester.h"
#include "../Core/Smoketest.h"

// ---------------------------------------------------------------------------
// A bare outer billiards setup to poke at, rather than a scripted video.
//
//     .\go.ps1 BilliardsPlayground 960 540 30 -n
//
// It builds one table and one orbit, then hands you the state tester: an ffplay
// window showing the scene, and a prompt where you retype any state variable and
// watch it change. Type "exit" when you are done and it renders the short clip
// at the bottom of this file so the run leaves something behind.
//
// The prompt takes "variable:equation", where the equation is in RPN and can
// reference this scene's other variables as <local> or globals as {global}:
//
//     iterations:120          how many hops to draw - fractional is fine
//     p.x:2.4                 where the orbit sets off from
//     v0.x:1.3                drag a corner of the table around
//     zoom:-0.6               negative zooms out (CoordinateScene's is inverted)
//     rainbow:1               tint hops by age
//     orbit_fade:0.9          dim the trail behind the head
//     pivot_opacity:1         mark the corner each hop turns about
//     line_thickness:<dot_size> 2 *   ...arithmetic works: + - * / ^ sin cos
//                                exp sqrt abs log floor ceil pi e phi min max
//                                lerp smoothlerp logistic and comparisons
//     print_state             dump every variable and its current value
//
// The singularity graph lives here too, off until you ask for it:
//
//     ray_opacity:1           the n rays where the map is undefined
//     singularity_opacity:1   turn the fractal on...
//     singularity_depth:200   ...and this is how many preimages deep. THE knob.
//                             Cost is linear in it; fractional values are fine.
//     singularity_glow:0.4    soft halo around each line
//     singularity_rainbow:1   tint lines by how deep a preimage they are
//     island_opacity:0.6      fill the gaps the graph leaves, hued by which hop
//     island_max_period:0     brings the orbit back nearest (0 sizes it to the shot)
//     curvature:-0.1          bend the plane hyperbolic (0 is Euclidean). The
//     horizon_opacity:1       ideal boundary sits at 1/sqrt(-curvature).
//
// The tester STEPS, it does not play: one frame is drawn per Enter. {t} only
// advances while frames are being rendered, so a "{t} sin" equation is legal but
// sits at a constant here - it comes to life in the clip rendered after "exit".
// To see motion, retype a value and watch it jump. Do not reference a variable
// inside its own equation; the state manager is a DAG and that is a cycle.
//
// The interesting knob is `iterations`. This table is a square, and every orbit
// of a square closes up - 4 hops near in, 8 further out - so pushing iterations
// past that just retraces. Nudge a corner (v0.x:1.3) and it never closes again:
// crank iterations to a few hundred and zoom out to watch one point wander over
// a region tens of times the size of the table it is orbiting.
//
// Add `-l` to the go.ps1 line above and the closing clip plays straight into an
// ffplay window instead of being encoded, which also skips the hardware encoder.
// ---------------------------------------------------------------------------

void render_video() {
    OuterBilliardsScene bs;

    // A square, corners at 45 degrees, circumradius 1.
    bs.set_regular_polygon(4, 1.0f, M_PI / 4);

    // One orbit. Its start is animatable as p.x / p.y; this particular point has
    // period 4 about this square.
    bs.add_orbit("p", vec2(1.9f, 0.55f));

    bs.set_iterations(4);
    bs.frame_view(vec2(0, 0), 3.0f);            // world units visible above center
    bs.manager.set("pivot_opacity", "1");       // show which corner each hop uses

    // The smoketest has no terminal to prompt at, so it just renders the clip.
    if (!is_smoketest()) open_ui(bs);

    // Whatever state you left it in is where this starts from.
    stage_macroblock(SilenceBlock(4), 1);
    bs.iterate_to(MICRO, 60);
    bs.frame_view(MICRO, vec2(0, 0), 5.0f);
    bs.render_microblock();

    stage_macroblock(SilenceBlock(2), 1);
    bs.render_microblock();
}
