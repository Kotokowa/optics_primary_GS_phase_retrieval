
# GS Phase Retrieval (Improved)

- **Auto scaling** with Parseval (`--auto-rescale`).
- **Centering helpers**: `--assume-centered false` or `--flip-centered`.
- **Soft projection**: `--alpha`, or separate `--alpha-obj` / `--alpha-fou`.
- **Phase quantization**: `--phase-levels "[-1.570796,1.570796]"` (for binary phase letters).
- **Multi-start**: `--restarts N`.

Example:
```bash
python gs_phase_retrieval_plus.py --method gs   --fourier-mag A_192x192_fourier_mag.npy   --object-amp A_192x192_object_amp.npy   --iters 400 --restarts 5 --alpha 1.0 --auto-rescale   --phase-levels "[-1.570796,1.570796]"   --outdir results_A --save-png
```

```powershell
python .\gs_phase_retrieval_plus.py --method gs `
  --fourier-mag A_192x192_fourier_mag.npy `
  --object-amp A_192x192_object_amp.npy `
  --assume-centered false `
  --iters 400 --restarts 5 --alpha 1.0 --auto-rescale `
  --phase-levels "[-1.570796,1.570796]" `
  --outdir results_A_plus --save-png --save-npy
```