#include "Molecule.h"
#include <array>
#include <cmath>
#include <iostream>
#include <sstream>

using std::array;
using std::string;
using std::vector;

// ---------------------------------------------------------------------------
// Cromer-Mann atomic form-factor coefficients (International Tables for
// Crystallography, Vol. C, table 6.1.1.4, neutral atoms). For each element we
// return nine numbers in the order
//     a1, a2, a3, a4,  b1, b2, b3, b4,  c
// which the GPU uses to evaluate  f(s) = c + sum_i a_i * exp(-b_i * s^2).
// To support more elements, just paste another row from the tables here.
// ---------------------------------------------------------------------------
static array<float, 9> cromer_mann_for(const string& element) {
    // The published tables list values as a1,b1,a2,b2,a3,b3,a4,b4,c; below they are
    // reordered to a1,a2,a3,a4, b1,b2,b3,b4, c to match the loop in the kernel.
    if (element == "H")  return {{0.489918f,0.262003f,0.196767f,0.049879f,  20.6593f, 7.74039f, 49.5519f,  2.20159f,   0.001305f}};
    if (element == "C")  return {{2.310000f,1.020000f,1.588600f,0.865000f,  20.8439f, 10.2075f, 0.568700f, 51.6512f,   0.215600f}};
    if (element == "N")  return {{12.21260f,3.132200f,2.012500f,1.166300f,  0.005700f,9.89330f, 28.9975f,  0.582600f, -11.52900f}};
    if (element == "O")  return {{3.048500f,2.286800f,1.546300f,0.867000f,  13.2771f, 5.70110f, 0.323900f, 32.9089f,   0.250800f}};
    if (element == "F")  return {{3.539200f,2.641200f,1.517000f,1.024300f,  10.2825f, 4.29440f, 0.261500f, 26.1476f,   0.277600f}};
    if (element == "P")  return {{6.434500f,4.179100f,1.780000f,1.490800f,  1.906700f,27.1570f, 0.526000f, 68.1645f,   1.114900f}};
    if (element == "S")  return {{6.905300f,5.203400f,1.437900f,1.586300f,  1.467900f,22.2151f, 0.253600f, 56.1720f,   0.866900f}};
    if (element == "Cl") return {{11.46040f,7.196400f,6.255600f,1.645500f,  0.010400f,1.16620f, 18.5194f,  47.7784f,  -9.557400f}};
    if (element == "Fe") return {{11.76950f,7.357300f,3.522200f,2.304500f,  4.761100f,0.307200f,15.3535f,  76.8805f,   1.036900f}};

    // Unknown element: fall back to carbon and warn, so a typo does not crash a render.
    std::cout << "Molecule: no form factor for element '" << element
              << "', substituting carbon." << std::endl;
    return cromer_mann_for("C");
}

// Covalent radii in Angstroms (Cordero et al. 2008). Used only to guess which
// atoms are bonded for the 3D drawing; they play no part in the physics.
static float covalent_radius(const string& element) {
    if (element == "H")  return 0.31f;
    if (element == "C")  return 0.76f;
    if (element == "N")  return 0.71f;
    if (element == "O")  return 0.66f;
    if (element == "F")  return 0.57f;
    if (element == "P")  return 1.07f;
    if (element == "S")  return 1.05f;
    if (element == "Cl") return 1.02f;
    if (element == "Fe") return 1.32f;
    return 0.76f;  // same carbon fallback as the form factors
}

// CPK colors - the convention every chemistry textbook uses. Carbon is a light
// grey rather than the usual near-black, because these are drawn on a dark background.
uint32_t element_color(const string& element) {
    if (element == "H")  return 0xffffffff;  // white
    if (element == "C")  return 0xff909090;  // grey
    if (element == "N")  return 0xff3050f8;  // blue
    if (element == "O")  return 0xffff0d0d;  // red
    if (element == "F")  return 0xff90e050;  // green
    if (element == "P")  return 0xffff8000;  // orange
    if (element == "S")  return 0xffffff30;  // yellow
    if (element == "Cl") return 0xff1ff01f;  // green
    if (element == "Fe") return 0xffe06633;  // rust
    return 0xffff00ff;                       // magenta: "idk"
}

// Ball radius for the 3D view, in Angstroms - the same units as the coordinates,
// so atoms scale with the molecule as the camera moves. These are the usual
// ball-and-stick radii: well under the real van der Waals sizes, so the bonds
// stay visible. Hydrogen is drawn smaller so it reads as the light atom.
float element_radius(const string& element) {
    if (element == "H") return 0.25f;
    if (element == "C") return 0.35f;
    if (element == "N") return 0.35f;
    if (element == "O") return 0.37f;
    return 0.40f;
}

vector<Atom> parse_xyz(const string& xyz_text) {
    vector<Atom> atoms;
    std::istringstream stream(xyz_text);
    string line;
    while (std::getline(stream, line)) {
        std::istringstream ls(line);
        string element;
        double x, y, z;
        // A valid atom line is "<symbol> <x> <y> <z>". Header/count/comment lines
        // fail this parse and are silently skipped.
        if (ls >> element >> x >> y >> z) {
            atoms.push_back({element, vec3((float)x, (float)y, (float)z)});
        }
    }
    return atoms;
}

// This is just three 2D rotations; we do it on the CPU once per atom per frame,
// which is negligible because molecules have so few atoms. (May want to move this to gpu if I ever move to protiens)
vec3 rotate_euler(vec3 p, float yaw, float pitch, float roll) {
    float cy = std::cos(yaw),   sy = std::sin(yaw);
    vec3 a( p.x * cy + p.z * sy,   p.y,   -p.x * sy + p.z * cy );      // spin about y

    float cp = std::cos(pitch), sp = std::sin(pitch);
    vec3 b( a.x,   a.y * cp - a.z * sp,   a.y * sp + a.z * cp );       // tilt about x

    float cr = std::cos(roll),  sr = std::sin(roll);
    return vec3( b.x * cr - b.y * sr,   b.x * sr + b.y * cr,   b.z );  // spin about z
}

Molecule::Molecule(const vector<Atom>& atoms) {
    // Find the centroid so we can center the molecule. A pure shift does not change
    // the diffraction pattern (|F|^2 ignores a global phase), but centering makes
    // rotations spin the molecule about its own middle.
    //
    // Note this centroid is computed ONCE, from the input coordinates, and then
    // baked into base_positions. We deliberately do not re-center each frame: if we
    // did, nudging a single atom would slide every OTHER atom across the screen to
    // compensate, which reads as the whole molecule drifting. With a fixed origin,
    // moving one atom moves exactly that one atom. The physics is identical either
    // way, since a global translation leaves |F|^2 untouched.
    vec3 centroid(0, 0, 0);
    for (const Atom& atom : atoms) centroid += atom.position;
    if (!atoms.empty()) centroid /= (float)atoms.size();

    float total_electrons = 0.0f;
    for (const Atom& atom : atoms) {
        elements.push_back(atom.element);
        base_positions.push_back(atom.position - centroid);

        array<float, 9> cm = cromer_mann_for(atom.element);
        for (int i = 0; i < 9; i++) cromer_mann_flat.push_back(cm[i]);

        // f(0) = a1 + a2 + a3 + a4 + c equals this atom's electron count.
        total_electrons += cm[0] + cm[1] + cm[2] + cm[3] + cm[8];
    }
    inv_max_intensity = total_electrons > 0.0f ? 1.0f / (total_electrons * total_electrons) : 1.0f;

    // Start out un-rotated and un-nudged, so a Molecule is renderable even before
    // anyone has driven it (e.g. if a scene is used standalone). Bonds are always
    // derived here, from the geometry as loaded - under BondMode::FIXED this is
    // the one and only time they are computed, and the list stands for good.
    world_positions = base_positions;
    recompute_bonds();
}

void Molecule::set_bond_tolerance(float tolerance) {
    bond_tolerance = tolerance;
    recompute_bonds();
    mark_updated();
}

void Molecule::set_world_positions(const vector<vec3>& positions) {
    world_positions = positions;
    // Under FIXED we keep the bond list built from the initial geometry, so bonds
    // simply stretch to follow their atoms. Rotation never changes interatomic
    // distances anyway, so this only matters once atoms move individually.
    if (bond_mode == BondMode::DYNAMIC) recompute_bonds();
    mark_updated();
}

// Guess the bonds from interatomic distances, so any .xyz file works without
// also having to supply a connectivity table. How often this runs - once, or
// every frame - is what BondMode selects.
void Molecule::recompute_bonds() {
    bonds.clear();
    for (size_t i = 0; i < world_positions.size(); i++) {
        for (size_t j = i + 1; j < world_positions.size(); j++) {
            float cutoff = bond_tolerance * (covalent_radius(elements[i]) + covalent_radius(elements[j]));
            vec3 diff = world_positions[i] - world_positions[j];
            if (dot(diff, diff) < cutoff * cutoff) bonds.push_back({(int)i, (int)j});
        }
    }
}
