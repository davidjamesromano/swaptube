#pragma once

#include "../Host_Device_Shared/vec.h"
#include "DataObject.h"
#include <string>
#include <vector>

// One atom of a molecule: which element it is, and where it sits (in Angstroms).
struct Atom {
    std::string element;   // chemical symbol, e.g. "C", "H", "O"
    vec3        position;  // 3D coordinate, in Angstroms
};

// A pair of atom indices that should be drawn connected.
struct Bond {
    int i;
    int j;
};

// Parse a block of text in ".xyz" format - the standard way to write down "a list
// of atoms and their coordinates" - into a vector<Atom>. Each atom line looks like:
//     C   0.000   1.396   0.000
// The usual .xyz header (an atom-count line and a comment line) is tolerated but
// not required: any line that is not "<symbol> <x> <y> <z>" is simply skipped.
std::vector<Atom> parse_xyz(const std::string& xyz_text);

// Rotate a point by yaw (about the vertical y axis), then pitch (about the
// horizontal x axis), then roll (about the beam z axis). Angles are in radians.
vec3 rotate_euler(vec3 p, float yaw, float pitch, float roll);

// What happens to the bonds when you move an individual atom.
enum class BondMode {
    // Work out the bond list once, from the coordinates as loaded, and keep it.
    // Moving an atom stretches its bonds; they never break and none appear.
    // This is the "it stays the same molecule" behavior, and is the default.
    FIXED,

    // Re-derive the bond list from interatomic distances every frame. Pull an
    // atom far enough away and its bonds snap; push two atoms together and a new
    // bond forms. Good for showing bond breaking/forming, but be aware it can
    // flicker if a distance sits right at the cutoff.
    DYNAMIC,
};

// ---------------------------------------------------------------------------
// The shared molecule.
//
// This is the single source of truth that links the 3D view and the diffraction
// pattern. MoleculeSystemScene OWNS it (registers it via add_data_object) and
// writes `world_positions` once per frame; MoleculeScene and ScatteringScene
// hold NON-OWNING pointers and only read. That one-writer/many-readers shape is
// deliberate: Scene's destructor deletes every registered data object, so
// registering the same Molecule with two scenes would double-free it.
// ---------------------------------------------------------------------------
class Molecule : public DataObject {
public:
    Molecule(const std::vector<Atom>& atoms);

    // Positions are pushed in explicitly by MoleculeSystemScene::draw(), so there
    // is nothing to do on the generic per-frame data tick.
    void tick(const StateReturn& state) override {}

    int size() const { return (int)base_positions.size(); }

    // Called once per frame by the owner. Refreshes the bond list too, but only
    // if bond_mode is DYNAMIC.
    void set_world_positions(const std::vector<vec3>& positions);

    // See BondMode. Set this before rendering; changing it mid-render is allowed
    // but FIXED will then freeze whatever bonds happen to be current.
    BondMode bond_mode = BondMode::FIXED;

    // Two atoms count as bonded when they are closer than this multiple of the
    // sum of their covalent radii. Raise it if bonds are missing, lower it if
    // spurious ones appear between next-nearest neighbours. Re-derives the bond
    // list immediately, so this works under FIXED too (where bonds would
    // otherwise have been settled back in the constructor).
    void set_bond_tolerance(float tolerance);

    // --- Fixed for the lifetime of the molecule ---
    std::vector<std::string> elements;
    std::vector<vec3>        base_positions;    // centered on the INITIAL centroid
    std::vector<float>       cromer_mann_flat;  // 9 form-factor coefficients per atom

    // 1 / (total electrons)^2. The central (Sx=Sy=0) pixel has intensity
    // (total electrons)^2, so multiplying by this maps the center to 1.0. This
    // depends only on WHICH elements are present, never on where they are, so it
    // stays correct when atoms move.
    float inv_max_intensity;

    // --- Rewritten every frame ---
    std::vector<vec3> world_positions;  // after per-atom offsets and rotation
    std::vector<Bond> bonds;

private:
    void recompute_bonds();

    float bond_tolerance = 1.3f;
};

// Display properties for the 3D view.
uint32_t element_color(const std::string& element);   // CPK coloring
float    element_radius(const std::string& element);  // dot size, in Point::size units
