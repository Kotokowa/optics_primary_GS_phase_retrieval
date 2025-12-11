python gs_phase_retrieval.py `
  --object_amp my_test_sets_gs600/1_object_amplitude.npy `
  --fourier_amp my_test_sets_gs600/1_fourier_amplitude.npy `
  --target_complex my_test_sets_gs600/1_target_object_complex.npy `
  --max_iter 2000 `
  --tol 1e-5 `
  --crop_height 200 `
  --crop_width 300 `
  --output_dir temp_200+200p

  --crop_size 200 ` !<可选>

recommended iters: 50 200 500 1000

python gs_phase_retrieval_new.py `
  --object_amp my_test_sets_gs600/1_object_amplitude.npy `
  --fourier_amp my_test_sets_gs600/1_fourier_amplitude.npy `
  --target_complex my_test_sets_gs600/1_target_object_complex.npy `
  --pad 128

python gs_phase_retrieval_fresnel.py `
  --object_amp my_object_amp.npy `
  --sensor_amp my_sensor_amp.npy `
  --wavelength 532e-9 `
  --dx 6.5e-6 `
  --dz 0.1 `
  --target_object_complex my_object_complex.npy `
  --target_sensor_complex my_sensor_complex.npy `
  --max_iter 400 `
  --tol 1e-6 `
  --output_dir fresnel_results_300



python make_gs_testset.py `
  --amp_npy   my_test_sets_gs600/my_amp.npy `
  --phase_npy my_test_sets_gs600/my_phase.npy `
  --geometry  fourier `
  --output_dir my_test_sets_gs600 `
  --prefix    1


python make_gs_testset.py `
  --amp_npy   my_amp.npy `
  --phase_npy my_phase.npy `
  --geometry  fresnel `
  --wavelength  532e-9 `
  --dx         6.5e-6 `
  --dz         0.1 `
  --output_dir lensless_heart_300 `
  --prefix    heart300


python make_gs_testset_with_noise.py `
  --amp_npy   my_test_sets_noise/my_amp.npy `
  --phase_npy my_test_sets_noise/my_phase.npy `
  --geometry fourier `
  --output_dir my_test_sets_noise `
  --prefix 1 `
  --fourier_noise_std_rel 0.2 `
  --seed 123
