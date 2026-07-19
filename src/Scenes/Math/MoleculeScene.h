#pragma once

#include "../Common/ThreeDimensionScene.h"
#include "../../DataObjects/Molecule.h"
#include <vector>

// A 3D ball-and-stick view of a Molecule.
//
// This scene owns no molecular state of its own: every frame it just reads
// molecule->world_positions and redraws. Whoever owns the Molecule (normally
// MoleculeSystemScene) is responsible for filling that in first. Camera controls
// are inherited from ThreeDimensionScene: d (distance), q1/qi/qj/qk (orientation),
// x/y/z (focus point), fov.
//
// Atoms and bonds are drawn by our own shaded kernel (src/CUDA/ball_and_stick.cu)
// rather than by ThreeDimensionScene's flat-disc/hairline renderers, so they look
// like solid balls and tubes and occlude each other correctly.
//
// State variables you can set() or transition():
//   zoom
//   atom_size     scales every atom. 1 is the per-element default from
//                 Molecule.cpp
//   bond_radius   tube radius for bonds, in Angstroms - the same units as the
//                 atom radii, so the two are directly comparable.
// The radii also scale with the inherited points_radius_multiplier.
class MoleculeScene : public ThreeDimensionScene {
public:
    // The molecule is NOT owned here - see the comment on Molecule.
    MoleculeScene(const Molecule* mol, const vec2& dimensions = vec2(1, 1));

    void draw() override;
    const StateQuery populate_state_query() const override;

private:
    const Molecule* molecule;

    // Atoms and bonds live here rather than in the base class's `points`/`lines`,
    // because we render them ourselves. Kept as members so they are not
    // reallocated every frame. 
    std::vector<Point> atom_spheres;
    std::vector<Line>  bond_tubes;
};
