#include "MoleculeScene.h"
#include <algorithm>
#include <cmath>

// Shaded ball-and-stick renderer, implemented in src/CUDA/ball_and_stick.cu.
// Declared here with plain, non-namespaced types; the linker connects it to the
// definition.
extern "C" void render_ball_and_stick_on_gpu(
    unsigned int* d_pixels, const ivec2& wh,
    float geom_mean_size, float opacity, float radius_multiplier,
    Point* h_spheres, int num_spheres,
    Line*  h_tubes,   int num_tubes, float tube_size,
    const quat& camera_direction, const vec3& camera_pos, float fov);

// White bonds
static const int BOND_COLOR = 0xffffffff;

MoleculeScene::MoleculeScene(const Molecule* mol, const vec2& dimensions)
    : ThreeDimensionScene(dimensions), molecule(mol) {
    // --- Auto-framing ---
    // Work out how far back the camera has to sit for this particular molecule to
    // fill the panel, and make that the default.
    float radius = 0.0f;
    for (const vec3& p : mol->base_positions) radius = std::max(radius, std::sqrt(dot(p, p)));
    float biggest_atom = 0.0f;
    for (const std::string& e : mol->elements) biggest_atom = std::max(biggest_atom, element_radius(e));

    ivec2 wh = get_width_height();
    float geom_mean = std::sqrt((float)wh.x * (float)wh.y);
    float shorter   = (float)std::min(wh.x, wh.y);

    // Projection maps a world offset u at depth z to geom_mean*fov*u/z pixels. The
    // nearest part of the molecule sits at depth (d - radius), so solve for the d
    // that puts the far edge at `fill` of the panel's shorter half-dimension.
    const float fill = 0.85f;
    float extent = radius + biggest_atom;
    float fit_distance = (extent > 0.0f)
        ? radius + geom_mean * extent / (0.5f * shorter * fill)
        : 5.0f;   // a lone atom has no extent to frame; any distance will do

    manager.set({
        // Zoom, with the same meaning as CoordinateScene's
        {"zoom",          "0"},
        {"fit_distance",  std::to_string(fit_distance)},
        {"d",             "<fit_distance> 0 <zoom> - exp *"},

        {"atom_size",   "1"},     // scales the per-element atom radii
        {"bond_radius", "0.15"},  // tube radius in Angstroms, like the atom radii
    });
}

const StateQuery MoleculeScene::populate_state_query() const {
    StateQuery sq = ThreeDimensionScene::populate_state_query();
    state_query_insert_multiple(sq, {"atom_size", "bond_radius"});
    return sq;
}

void MoleculeScene::draw() {
    // Rebuild the geometry from scratch each frame, the same way GraphScene does.
    // The atoms may have rotated or been individually nudged since last frame.
    atom_spheres.clear();
    bond_tubes.clear();

    const std::vector<vec3>& pos = molecule->world_positions;
    const float atom_size = state["atom_size"];

    for (int i = 0; i < molecule->size(); i++) {
        atom_spheres.push_back(Point(pos[i],
                                     (int)element_color(molecule->elements[i]),
                                     1,
                                     element_radius(molecule->elements[i]) * atom_size));
    }

    for (const Bond& bond : molecule->bonds) {
        bond_tubes.push_back(Line(pos[bond.i], pos[bond.j], BOND_COLOR, 1, false));
    }

    // Base class draws any surfaces
    ThreeDimensionScene::draw();

    // Atoms and bonds go down together in a single pass, so they depth-test against each other
    if (state["points_opacity"] > .001 && state["points_radius_multiplier"] > 0.001) {
        render_ball_and_stick_on_gpu(
            gpu_pix->get_ptr(),
            get_width_height(),
            get_geom_mean_size(),
            state["points_opacity"],
            state["points_radius_multiplier"],
            atom_spheres.data(), (int)atom_spheres.size(),
            bond_tubes.data(),   (int)bond_tubes.size(), state["bond_radius"],
            camera_direction,
            camera_pos,
            fov
        );
    }
}
