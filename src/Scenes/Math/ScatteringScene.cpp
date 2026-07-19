#include "ScatteringScene.h"

// The GPU launcher, implemented in src/CUDA/scattering.cu.
extern "C" void scattering_render(
    const ivec2& wh,
    const vec2& lx_ty,
    const vec2& rx_by,
    int num_atoms,
    const vec3* atom_positions,
    const float* cromer_mann,
    float ewald_radius,
    float inv_max_intensity,
    float brightness,
    float gamma,
    unsigned int* d_colors
);

ScatteringScene::ScatteringScene(const Molecule* mol, const vec2& dimensions)
    : CoordinateScene(dimensions), molecule(mol) {
    // Only the detector's own display parameters live here.
    manager.set({
        {"ewald_radius", "50"},    // = 1/wavelength (1/A); large -> flat 2D transform
        {"brightness",   "1"},     // overall gain applied after normalization
        {"gamma",        "0.35"},  // <1 brightens faint spots; =1 is linear
    });
}

const StateQuery ScatteringScene::populate_state_query() const {
    // Start from the viewport variables CoordinateScene needs (left_x, right_x, ...),
    // then add the ones draw() reads.
    StateQuery sq = CoordinateScene::populate_state_query();
    state_query_insert_multiple(sq, {"ewald_radius", "brightness", "gamma"});
    return sq;
}

void ScatteringScene::draw() {
    // Hand everything to the GPU
    scattering_render(
        get_width_height(),
        vec2(state["left_x"],  state["top_y"]),
        vec2(state["right_x"], state["bottom_y"]),
        molecule->size(),
        molecule->world_positions.data(),
        molecule->cromer_mann_flat.data(),
        state["ewald_radius"],
        molecule->inv_max_intensity,
        state["brightness"],
        state["gamma"],
        gpu_pix->get_ptr()
    );

    // Let the base class overlay coordinate axes if ticks_opacity was turned up.
    CoordinateScene::draw();
}
