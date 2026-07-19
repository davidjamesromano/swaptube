#pragma once

#include "../Common/CompositeScene.h"
#include "../../DataObjects/Molecule.h"
#include "MoleculeScene.h"
#include "ScatteringScene.h"
#include <memory>
#include <string>

// ---------------------------------------------------------------------------
// A molecule and its diffraction pattern, side by side and always in sync.
//
// This scene is the single owner of the molecular state. Each frame its draw()
//   1. reads the orientation and the per-atom offsets out of its own state,
//   2. computes every atom's world position,
//   3. writes them into the shared Molecule,
//   4. then lets CompositeScene draw the two panels, which both just read it.
//
// Step 4 happening last is what makes this safe: a SuperScene updates all of its
// subscenes during its OWN update phase, so the parent's draw() always runs
// before any child's. Neither panel can ever see a stale position, and there is
// no render-order dependency between them.
//
// State variables:
//   rot_yaw / rot_pitch / rot_roll   orientation of the whole molecule, in radians
//   a<i>.x / a<i>.y / a<i>.z         displacement of atom i from its .xyz position,
//                                    in Angstroms (i is the atom's index in the file)
//
// Rotation is applied AFTER the per-atom offsets, so nudging an atom and then
// spinning the molecule does what you would expect.
//
// The two panels are public so you can drive their own settings directly, e.g.
//   sys.molecule_panel->manager.set("d", "6");        // camera distance
//   sys.pattern_panel->manager.set("brightness", "2");
// ---------------------------------------------------------------------------
class MoleculeSystemScene : public CompositeScene {
public:
    MoleculeSystemScene(const std::vector<Atom>& atoms, const vec2& dimensions = vec2(1, 1));

    void draw() override;
    const StateQuery populate_state_query() const override;

    // Name of the state variable holding atom `index`'s offset along `axis`
    // ("x", "y" or "z"). Use it to build transitions:
    //   sys.manager.transition(MICRO, sys.atom_var(2, "y"), "0.8");
    static std::string atom_var(int index, const std::string& axis);

    // --- Whole-molecule shortcuts ---
    // Each of these issues ONE batched transition covering every variable it
    // touches. That also means you cannot transition the same atom individually
    // in the same microblock afterwards - the state manager rejects a second
    // transition on a variable already in one.

    // Send every atom back to the position it had in the .xyz file.
    void reset_atoms(const TransitionType tt, const bool smooth = true);

    // Displace all atoms at once - one vector per atom, in Angstroms, in the
    // molecule's own frame. This is the natural way to animate a vibrational mode
    // or any other collective distortion. Throws if the count does not match.
    //
    // `scale` multiplies every vector, so a mode can be written down once at
    // whatever amplitude the source gave it and then exaggerated or damped at the
    // call site: pass 3 to triple the motion, 0.5 to halve it, -1 to swing it the
    // other way (handy for the second half of an oscillation).
    void displace_atoms(const TransitionType tt, const std::vector<vec3>& offsets,
                        const float scale = 1.0f, const bool smooth = true);

    // Displace a single atom along all three axes in one call.
    void displace_atom(const TransitionType tt, int index, const vec3& offset, const bool smooth = true);

    // Whether moving an atom re-derives the bond list or just stretches the bonds
    // the molecule started with. See BondMode. Set this before rendering.
    void set_bond_mode(BondMode mode) { molecule->bond_mode = mode; }

    // Distance cutoff used when deriving bonds, as a multiple of the summed
    // covalent radii. Raise it if bonds are missing, lower it if spurious ones
    // appear between next-nearest neighbours.
    void set_bond_tolerance(float tolerance) { molecule->set_bond_tolerance(tolerance); }

    std::shared_ptr<ScatteringScene> pattern_panel;   // left half
    std::shared_ptr<MoleculeScene>   molecule_panel;  // right half

private:
    Molecule* molecule;  // owned here, via add_data_object
};
