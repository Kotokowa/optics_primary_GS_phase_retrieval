# GS Phase Retrieval Package

This package contains:
- `gs_phase_retrieval.py`: a command-line Gerchberg–Saxton implementation using centered FFTs.
- `dataset/`: 64 test files (4 patterns × 4 sizes × 4 file types).

## File conventions

Each dataset folder (e.g., `dataset/star_100`) contains exactly four files:
- `real_amplitude.npy` — A(x), the real-space amplitude constraint (circular aperture).
- `freq_amplitude_centered.npy` — B(k), the centered Fourier magnitude constraint.
- `target_object_complex.npz` — ground-truth complex object with key `obj_complex`.
- `target_phase.png` — visualization of target phase (for human inspection).

**All Fourier-domain arrays are CENTERED.**

## Recommended iteration counts for GS
- 50×50: 300 iters
- 100×100: 500 iters
- 300×300: 1000 iters
- 800×800: 1800 iters

## Usage example

```bash
# Example: run on the 100×100 star
python gs_phase_retrieval.py   --real_amp dataset/star_100/real_amplitude.npy   --freq_amp dataset/star_100/freq_amplitude_centered.npy   --val_file dataset/star_100/target_object_complex.npz   --outdir outputs_star_100   --max_iters 500
```

The script will save:
- `convergence.csv` and `convergence.png` (iteration errors and curve),
- `recovered_object_complex.npz` (complex field),
- `recovered_freq_complex_centered.npz` (centered complex spectrum),
- log-amplitude images in both domains, and amplitude/phase maps in object domain,
- `summary.json` including optional validation metrics.
