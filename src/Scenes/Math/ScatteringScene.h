#pragma once

#include "../Common/CoordinateScene.h"
#include "../../DataObjects/Molecule.h"

// A CoordinateScene whose 2D plane is reciprocal space (the detector). It renders
// the oriented IAM diffraction pattern of a Molecule.
//
// This scene holds no molecular state: it reads molecule->world_positions each
// frame, so whatever the 3D view is showing is exactly what is being diffracted.
// Orientation and per-atom motion are driven on MoleculeSystemScene, which owns
// the Molecule and fills in those positions before this scene draws.
//
// State variables you can set() or transition():
//   ewald_radius   = 1/wavelength (1/A); large => flat 2D transform
//   brightness     overall gain applied after normalization
//   gamma          contrast curve; <1 brightens faint spots
// (Panning/zooming the detector is inherited from CoordinateScene: center_x,
//  center_y, zoom.)
class ScatteringScene : public CoordinateScene {
public:
    // The molecule is NOT owned here - see the comment on Molecule.
    ScatteringScene(const Molecule* mol, const vec2& dimensions = vec2(1, 1));

    const StateQuery populate_state_query() const override;
    void draw() override;

private:
    const Molecule* molecule;
};
