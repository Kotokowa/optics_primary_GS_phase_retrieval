python gs_phase_retrieval.py `
  --object_amp my_test_sets_121102/1_object_amplitude.npy `
  --fourier_amp my_test_sets_121102/1_fourier_amplitude.npy `
  --target_complex my_test_sets_121102/1_target_object_complex.npy `
  --max_iter 2000 `
  --tol 1e-5 `
  --crop_height 200 `
  --crop_width 400 `
  --output_dir temp_121102

recommended iters: 50 200 500 1000



python make_gs_testset.py `
  --amp_npy   my_test_sets_121102/my_amp.npy `
  --phase_npy my_test_sets_121102/my_phase.npy `
  --geometry  fourier `
  --output_dir my_test_sets_121102 `
  --prefix    1




python make_gs_testset_with_noise.py `
  --amp_npy   my_test_sets_noise/my_amp.npy `
  --phase_npy my_test_sets_noise/my_phase.npy `
  --geometry fourier `
  --output_dir my_test_sets_noise `
  --prefix 1 `
  --fourier_noise_std_rel 0.2 `
  --seed 123
