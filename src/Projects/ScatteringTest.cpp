#include "../Scenes/Math/MoleculeSystemScene.h"

// ---------------------------------------------------------------------------
// A molecule and its X-ray diffraction pattern, side by side and locked together.
// The diffraction pattern is computed from the very same atom coordinates the 3D
// view is drawing, so anything you do to the molecule - spin it, or drag a single
// atom around - shows up in both panels at once.
// ---------------------------------------------------------------------------

// What the bonds do when an individual atom moves.
//
//   0  BONDS STRETCH. The bond list is worked out once, from the coordinates
//      below, and then left alone. Move an atom and its bonds follow it like
//      elastic, nothing ever breaks
//
//   1  BONDS RE-FORM. Which atoms count as bonded is re-derived from their
//      distances every frame.
#define RECOMPUTE_BONDS_WHEN_ATOMS_MOVE 0

// Just paste xyz coordinates here, and the program will parse them into a Molecule. The
// first line is the atom count, the second is a comment, and the rest are coordinates 
// and element symbols for each. Order matters for individual motions later.
static const char* C60_XYZ = R"(
60
C60 Buckyball Ground State
C   0.000000   0.700000   3.397871
C   0.000000   0.700000  -3.397871
C   0.000000  -0.700000   3.397871
C   0.000000  -0.700000  -3.397871
C   3.397871   0.000000   0.700000
C  -3.397871   0.000000   0.700000
C   3.397871   0.000000  -0.700000
C  -3.397871   0.000000  -0.700000
C   0.700000   3.397871   0.000000
C   0.700000  -3.397871   0.000000
C  -0.700000   3.397871   0.000000
C  -0.700000  -3.397871   0.000000
C   0.700000   2.532624   2.265248
C   0.700000   2.532624  -2.265248
C   0.700000  -2.532624   2.265248
C   0.700000  -2.532624  -2.265248
C  -0.700000   2.532624   2.265248
C  -0.700000   2.532624  -2.265248
C  -0.700000  -2.532624   2.265248
C  -0.700000  -2.532624  -2.265248
C   2.532624   2.265248   0.700000
C   2.532624  -2.265248   0.700000
C  -2.532624   2.265248   0.700000
C  -2.532624  -2.265248   0.700000
C   2.532624   2.265248  -0.700000
C   2.532624  -2.265248  -0.700000
C  -2.532624   2.265248  -0.700000
C  -2.532624  -2.265248  -0.700000
C   2.265248   0.700000   2.532624
C  -2.265248   0.700000   2.532624
C   2.265248   0.700000  -2.532624
C  -2.265248   0.700000  -2.532624
C   2.265248  -0.700000   2.532624
C  -2.265248  -0.700000   2.532624
C   2.265248  -0.700000  -2.532624
C  -2.265248  -0.700000  -2.532624
C   1.132624   1.400000   2.965248
C   1.132624   1.400000  -2.965248
C   1.132624  -1.400000   2.965248
C   1.132624  -1.400000  -2.965248
C  -1.132624   1.400000   2.965248
C  -1.132624   1.400000  -2.965248
C  -1.132624  -1.400000   2.965248
C  -1.132624  -1.400000  -2.965248
C   1.400000   2.965248   1.132624
C   1.400000  -2.965248   1.132624
C  -1.400000   2.965248   1.132624
C  -1.400000  -2.965248   1.132624
C   1.400000   2.965248  -1.132624
C   1.400000  -2.965248  -1.132624
C  -1.400000   2.965248  -1.132624
C  -1.400000  -2.965248  -1.132624
C   2.965248   1.132624   1.400000
C  -2.965248   1.132624   1.400000
C   2.965248   1.132624  -1.400000
C  -2.965248   1.132624  -1.400000
C   2.965248  -1.132624   1.400000
C  -2.965248  -1.132624   1.400000
C   2.965248  -1.132624  -1.400000
C  -2.965248  -1.132624  -1.400000
)";

// Scaling of the mode
#define MODE_AMPLITUDE 2.0f

// H_g(1) Quadrupolar mode of C60, a "squashing" motion
static const std::vector<vec3> SQUASHING_MODE = {
    { 0.000f, -0.035f,  0.340f}, { 0.000f, -0.035f, -0.340f}, { 0.000f,  0.035f,  0.340f}, { 0.000f,  0.035f, -0.340f},
    {-0.170f,  0.000f,  0.070f}, { 0.170f,  0.000f,  0.070f}, {-0.170f,  0.000f, -0.070f}, { 0.170f,  0.000f, -0.070f},
    {-0.035f, -0.170f,  0.000f}, {-0.035f,  0.170f,  0.000f}, { 0.035f, -0.170f,  0.000f}, { 0.035f,  0.170f,  0.000f},
    {-0.035f, -0.127f,  0.227f}, {-0.035f, -0.127f, -0.227f}, {-0.035f,  0.127f,  0.227f}, {-0.035f,  0.127f, -0.227f},
    { 0.035f, -0.127f,  0.227f}, { 0.035f, -0.127f, -0.227f}, { 0.035f,  0.127f,  0.227f}, { 0.035f,  0.127f, -0.227f},
    {-0.127f, -0.113f,  0.070f}, {-0.127f,  0.113f,  0.070f}, { 0.127f, -0.113f,  0.070f}, { 0.127f,  0.113f,  0.070f},
    {-0.127f, -0.113f, -0.070f}, {-0.127f,  0.113f, -0.070f}, { 0.127f, -0.113f, -0.070f}, { 0.127f,  0.113f, -0.070f},
    {-0.113f, -0.035f,  0.253f}, { 0.113f, -0.035f,  0.253f}, {-0.113f, -0.035f, -0.253f}, { 0.113f, -0.035f, -0.253f},
    {-0.113f,  0.035f,  0.253f}, { 0.113f,  0.035f,  0.253f}, {-0.113f,  0.035f, -0.253f}, { 0.113f,  0.035f, -0.253f},
    {-0.057f, -0.070f,  0.297f}, {-0.057f, -0.070f, -0.297f}, {-0.057f,  0.070f,  0.297f}, {-0.057f,  0.070f, -0.297f},
    { 0.057f, -0.070f,  0.297f}, { 0.057f, -0.070f, -0.297f}, { 0.057f,  0.070f,  0.297f}, { 0.057f,  0.070f, -0.297f},
    {-0.070f, -0.148f,  0.113f}, {-0.070f,  0.148f,  0.113f}, { 0.070f, -0.148f,  0.113f}, { 0.070f,  0.148f,  0.113f},
    {-0.070f, -0.148f, -0.113f}, {-0.070f,  0.148f, -0.113f}, { 0.070f, -0.148f, -0.113f}, { 0.070f,  0.148f, -0.113f},
    {-0.148f, -0.057f,  0.140f}, { 0.148f, -0.057f,  0.140f}, {-0.148f, -0.057f, -0.140f}, { 0.148f, -0.057f, -0.140f},
    {-0.148f,  0.057f,  0.140f}, { 0.148f,  0.057f,  0.140f}, {-0.148f,  0.057f, -0.140f}, { 0.148f,  0.057f, -0.140f},
};

void render_video() {
    // One scene owning one molecule, drawn as two half-screen panels:
    // the diffraction pattern on the left, the 3D molecule on the right.
    MoleculeSystemScene sys(parse_xyz(C60_XYZ));

    // Set the bond mode
    sys.set_bond_mode(RECOMPUTE_BONDS_WHEN_ATOMS_MOVE ? BondMode::DYNAMIC : BondMode::FIXED);

    // Set zoom and brightness for the diffraction pattern. Auto framing
    sys.pattern_panel->manager.set("zoom", "0");
    sys.pattern_panel->manager.set("brightness", "1.5");

    // How big the atoms and bonds are drawn. These are scalars to the existing covalent radii, so 1.0 is "normal" and 2.0 is twice as big.
    sys.molecule_panel->manager.set("atom_size",   "1.2");
    sys.molecule_panel->manager.set("bond_radius", "0.15");

    // Zoom for the molecule panel
    sys.molecule_panel->manager.set("zoom", "0");

    // Hold still
    stage_macroblock(SilenceBlock(0.5), 1);
    sys.render_microblock();

    // Some rotations
    stage_macroblock(SilenceBlock(2), 1);
    sys.manager.transition(MICRO, "rot_roll", "3.1416");
    sys.render_microblock();

    stage_macroblock(SilenceBlock(2), 1);
    sys.manager.transition(MICRO, "rot_pitch", "0.6");
    sys.manager.transition(MICRO, "rot_yaw",   "0.5");
    sys.render_microblock();

    // Shot 4: back to face-on, ready for the vibration
    stage_macroblock(SilenceBlock(2), 1);
    sys.manager.transition(MICRO, "rot_pitch", "1.5708");
    sys.manager.transition(MICRO, "rot_yaw",   "0");
    sys.manager.transition(MICRO, "rot_roll",  "0");
    sys.render_microblock();

    // Do the vibration, going to the extremes of the scaled mode, then back to equilibrium
    float amp = MODE_AMPLITUDE;
    for (int i = 0; i < 20; i++) {
        stage_macroblock(SilenceBlock(0.1), 1);
        sys.displace_atoms(MICRO, SQUASHING_MODE, (i % 2) ? -amp : amp);
        sys.render_microblock();
        amp *= 0.85f; // decay the amplitude so it settles down
    }
    sys.reset_atoms(MICRO); // settle back to equilibrium at the end


    // Bring everything back
    stage_macroblock(SilenceBlock(2), 1);
    sys.manager.transition(MICRO, "rot_pitch", "0");
    sys.render_microblock();
}
