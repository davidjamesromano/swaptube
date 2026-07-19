#include "MoleculeSystemScene.h"
#include <stdexcept>

std::string MoleculeSystemScene::atom_var(int index, const std::string& axis) {
    return "a" + std::to_string(index) + "." + axis;
}

void MoleculeSystemScene::reset_atoms(const TransitionType tt, const bool smooth) {
    StateSet equations;
    for (int i = 0; i < molecule->size(); i++) {
        equations[atom_var(i, "x")] = "0";
        equations[atom_var(i, "y")] = "0";
        equations[atom_var(i, "z")] = "0";
    }
    manager.transition(tt, equations, smooth);
}

void MoleculeSystemScene::displace_atoms(const TransitionType tt, const std::vector<vec3>& offsets,
                                         const float scale, const bool smooth) {
    if ((int)offsets.size() != molecule->size()) {
        throw std::runtime_error(
            "MoleculeSystemScene::displace_atoms: given " + std::to_string(offsets.size()) +
            " offsets for a molecule with " + std::to_string(molecule->size()) + " atoms.");
    }
    StateSet equations;
    for (int i = 0; i < molecule->size(); i++) {
        equations[atom_var(i, "x")] = std::to_string(offsets[i].x * scale);
        equations[atom_var(i, "y")] = std::to_string(offsets[i].y * scale);
        equations[atom_var(i, "z")] = std::to_string(offsets[i].z * scale);
    }
    manager.transition(tt, equations, smooth);
}

void MoleculeSystemScene::displace_atom(const TransitionType tt, int index, const vec3& offset, const bool smooth) {
    manager.transition(tt, {
        {atom_var(index, "x"), std::to_string(offset.x)},
        {atom_var(index, "y"), std::to_string(offset.y)},
        {atom_var(index, "z"), std::to_string(offset.z)},
    }, smooth);
}

MoleculeSystemScene::MoleculeSystemScene(const std::vector<Atom>& atoms, const vec2& dimensions)
    : CompositeScene(dimensions) {
    molecule = new Molecule(atoms);
    add_data_object(molecule);  // owned here, and ONLY here

    manager.set({
        {"rot_yaw",   "0"},  // spin about the vertical axis (radians)
        {"rot_pitch", "0"},  // tilt about the horizontal axis (radians)
        {"rot_roll",  "0"},  // spin about the beam axis (radians)
    });

    // One displacement per atom, all starting at zero: the molecule begins in
    // exactly the geometry the .xyz file described.
    for (int i = 0; i < molecule->size(); i++) {
        manager.set(atom_var(i, "x"), "0");
        manager.set(atom_var(i, "y"), "0");
        manager.set(atom_var(i, "z"), "0");
    }

    // Two half-width panels, each filling its half of this scene.
    const vec2 panel_size(dimensions.x * 0.5f, dimensions.y);
    pattern_panel  = std::make_shared<ScatteringScene>(molecule, panel_size);
    molecule_panel = std::make_shared<MoleculeScene>(molecule, panel_size);

    add_scene(pattern_panel,  "pattern",  vec2(0.25, 0.5));  // left
    add_scene(molecule_panel, "molecule", vec2(0.75, 0.5));  // right
}

const StateQuery MoleculeSystemScene::populate_state_query() const {
    StateQuery sq = CompositeScene::populate_state_query();
    state_query_insert_multiple(sq, {"rot_yaw", "rot_pitch", "rot_roll"});
    for (int i = 0; i < molecule->size(); i++) {
        sq.insert(atom_var(i, "x"));
        sq.insert(atom_var(i, "y"));
        sq.insert(atom_var(i, "z"));
    }
    return sq;
}

void MoleculeSystemScene::draw() {
    float yaw   = state["rot_yaw"];
    float pitch = state["rot_pitch"];
    float roll  = state["rot_roll"];

    // Offset each atom, then rotate the whole assembly. Doing it in this order is
    // what makes "move an atom" and "spin the molecule" compose sensibly
    std::vector<vec3> world;
    world.reserve(molecule->size());
    for (int i = 0; i < molecule->size(); i++) {
        vec3 offset((float)state[atom_var(i, "x")],
                    (float)state[atom_var(i, "y")],
                    (float)state[atom_var(i, "z")]);
        world.push_back(rotate_euler(molecule->base_positions[i] + offset, yaw, pitch, roll));
    }
    molecule->set_world_positions(world);

    // Now both panels read the positions just published.
    CompositeScene::draw();
}
