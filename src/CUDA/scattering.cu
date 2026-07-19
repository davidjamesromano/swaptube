// scattering.cu
// ---------------------------------------------------------------------------
// GPU kernel that renders an ORIENTED X-ray diffraction pattern of a molecule
// using the Independent Atom Model (IAM).
//
// The physics in one paragraph:
//   Every pixel on the "detector" corresponds to a scattering vector S
//   (units: inverse Angstrom, 1/A). The molecule's scattering amplitude - the
//   "structure factor" - is the sum over all atoms of that atom's form factor
//   times a phase that depends on where the atom sits:
//       F(S) = sum_j  f_j(|S|/2) * exp( 2*pi*i * dot(S, r_j) )
//   Here r_j is atom j's position (in Angstroms) and f_j is its atomic form
//   factor (how strongly it scatters; it falls off with angle). What a detector
//   actually measures is the brightness, i.e. the squared magnitude:
//       I(S) = |F(S)|^2.
//
// Parallelism: one GPU thread computes one pixel. Each thread walks over every
// atom of the molecule and accumulates that atom's contribution to F(S).
//
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdint>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "../Host_Device_Shared/Color.h"

constexpr float TWO_PI = 6.283185307179586f;

// ---------------------------------------------------------------------------
// Evaluate a single atom's form factor f(s) with the 4-Gaussian Cromer-Mann
// model.
// ---------------------------------------------------------------------------
__device__ float cromer_mann_form_factor(const float* cm, float s) {
    float s2 = s * s;
    float f = cm[8];                       // constant term c
    f += cm[0] * expf(-cm[4] * s2);        // a1 * exp(-b1 s^2)
    f += cm[1] * expf(-cm[5] * s2);        // a2 * exp(-b2 s^2)
    f += cm[2] * expf(-cm[6] * s2);        // a3 * exp(-b3 s^2)
    f += cm[3] * expf(-cm[7] * s2);        // a4 * exp(-b4 s^2)
    return f;
}

__global__ void scattering_kernel(
    const Cuda::ivec2 wh,               // detector size, in pixels
    const Cuda::vec2  lx_ty,            // reciprocal-space coord of the top-left pixel
    const Cuda::vec2  rx_by,            // reciprocal-space coord of the bottom-right pixel
    const int         num_atoms,
    const Cuda::vec3* atom_positions,   // already-rotated atom positions (Angstrom)
    const float*      cromer_mann,      // 9 form-factor coefficients per atom
    const float       ewald_radius,     // = 1/wavelength (1/A); large => flat detector
    const float       inv_max_intensity,// = 1 / (total electrons)^2, for normalization
    const float       brightness,       // overall gain applied after normalization
    const float       gamma,            // contrast curve (<1 lifts faint detail)
    unsigned int*     colors            // output pixel buffer (ARGB, one uint per pixel)
) {
    // Which pixel does this thread own?
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= wh.x || py >= wh.y) return;         // threads off the edge do nothing
    int pixel_idx = py * wh.x + px;

    // --- Step 1: pixel -> transverse scattering vector (Sx, Sy) ---
    // This is the part of S that lies in the detector plane (perpendicular to the
    // incoming beam). The helper maps pixel coordinates into the coordinate window.
    Cuda::vec2 s_perp = Cuda::pixel_to_point_in_screen(Cuda::vec2(px, py), lx_ty, rx_by, wh);

    // --- Step 2: recover the beam-direction component Sz from the Ewald sphere ---
    // Elastic scattering (no energy lost) forces S to lie on a sphere of radius
    // R = 1/wavelength that passes through the origin. Solving |k_in + S| = |k_in|
    // for the z-component gives:
    //     Sz = -R + sqrt( R^2 - Sx^2 - Sy^2 )
    // When R is large (short wavelength) Sz ~ 0 and we see the flat 2D transform of
    // the molecule; when R is small the sphere's curvature makes the pattern
    // asymmetric front-to-back.
    float s_perp_sq  = s_perp.x * s_perp.x + s_perp.y * s_perp.y;
    float under_root = ewald_radius * ewald_radius - s_perp_sq;
    if (under_root < 0.0f) {
        // Outside the Ewald sphere: this scattering vector is physically
        // unreachable at this wavelength, so leave the pixel black.
        colors[pixel_idx] = 0xff000000;
        return;
    }
    float sz = -ewald_radius + sqrtf(under_root);
    Cuda::vec3 S(s_perp.x, s_perp.y, sz);

    // The form-factor argument is s = |S|/2 = sin(theta)/lambda.
    float stol = 0.5f * Cuda::length(S);

    // --- Step 3: sum each atom's complex contribution ---
    // We track the amplitude as a (real, imaginary) pair, since
    //   exp(i*phase) = cos(phase) + i*sin(phase).
    float f_real = 0.0f;
    float f_imag = 0.0f;
    for (int j = 0; j < num_atoms; j++) {
        float f     = cromer_mann_form_factor(cromer_mann + 9 * j, stol);
        float phase = TWO_PI * Cuda::dot(S, atom_positions[j]);
        f_real += f * cosf(phase);
        f_imag += f * sinf(phase);
    }

    // --- Step 4: brightness is the squared magnitude of the amplitude ---
    float intensity = f_real * f_real + f_imag * f_imag;

    // --- Step 5: map intensity to a color ---
    // The dynamic range is enormous (the center is vastly brighter than the fine
    // detail), so we normalize to [0,1], then apply gamma (<1 brightens faint
    // spots) and an overall brightness gain.
    float n = Cuda::clamp(intensity * inv_max_intensity, 0.0f, 1.0f);
    float v = Cuda::clamp(powf(n, gamma) * brightness, 0.0f, 1.0f);

    // black -> blue -> white ramp (from Color.h); reads nicely as a diffraction pattern.
    colors[pixel_idx] = Cuda::black_to_blue_to_white(v);
}

// ---------------------------------------------------------------------------
// Host-callable launcher. A molecule has only a handful of atoms, so we simply
// copy the two small arrays to the GPU, launch the kernel, and free them again.
// May need to revisit with larger systems but it may be ok
// ---------------------------------------------------------------------------
extern "C" void scattering_render(
    const Cuda::ivec2& wh,
    const Cuda::vec2&  lx_ty,
    const Cuda::vec2&  rx_by,
    int                num_atoms,
    const Cuda::vec3*  atom_positions,   // host memory
    const float*       cromer_mann,      // host memory
    float              ewald_radius,
    float              inv_max_intensity,
    float              brightness,
    float              gamma,
    unsigned int*      d_colors          // already on the GPU (the scene's pixel buffer)
) {
    // Allocate GPU copies of the per-atom data.
    Cuda::vec3* d_positions = nullptr;
    float*      d_cromer    = nullptr;
    size_t pos_bytes = (size_t)num_atoms * sizeof(Cuda::vec3);
    size_t cm_bytes  = (size_t)num_atoms * 9 * sizeof(float);
    cudaMalloc(&d_positions, pos_bytes);
    cudaMalloc(&d_cromer,    cm_bytes);
    cudaMemcpy(d_positions, atom_positions, pos_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cromer,    cromer_mann,    cm_bytes,  cudaMemcpyHostToDevice);

    // One thread per pixel, in 16x16 blocks (same layout the other kernels use).
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((wh.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (wh.y + threadsPerBlock.y - 1) / threadsPerBlock.y);
    scattering_kernel<<<numBlocks, threadsPerBlock>>>(
        wh, lx_ty, rx_by, num_atoms, d_positions, d_cromer,
        ewald_radius, inv_max_intensity, brightness, gamma, d_colors);
    cudaDeviceSynchronize();

    // Release the temp GPU buffers.
    cudaFree(d_positions);
    cudaFree(d_cromer);
}
