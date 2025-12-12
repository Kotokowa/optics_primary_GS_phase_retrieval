python gs_phase_retrieval_fresnel.py `
  --object_amp case1_object_amp.npy `
  --sensor_amp case1_sensor_amp.npy `
  --wavelength 532e-9 `
  --dx 6.5e-6 `
  --dz 0.1 `
  --target_object_complex case1_object_complex.npy `
  --target_sensor_complex case1_sensor_complex.npy `
  --max_iter 1000 `
  --tol 1e-7 `
  --init_mode backprop `
  --relaxation 0.9 `
  --seed 0 `
  --output_dir fresnel_results_case1




python make_gs_testset.py `
  --amp_npy   GS/my_amp.npy `
  --phase_npy GS/my_phase.npy `
  --geometry  fresnel `
  --wavelength  532e-9 `
  --dx         6.5e-6 `
  --dz         0.1 `
  --output_dir GS `
  --prefix    1





python pad_object.py --in_complex object_complex.npy --pad 128 --out_prefix object_pad_ --show



or: 

> make_object.py -> make_dataset.py -> gs_fresnel.py

python make_object_v2.py

python make_dataset.py

python gs_fresnel.py --dataset gs_dataset.npz --max_iters 500








python make_object_v2.py --mode snellen --prefix snellen_ --show
python make_dataset.py --obj_prefix snellen_ --out gs_snellen.npz --wavelength 532e-9 --pixel_size 8e-6 --z 0.10 --show
python gs_fresnel.py --dataset gs_snellen.npz --out_prefix snellen_ --max_iters 500


python make_object_v2.py --mode optics --prefix optics_ --show
python make_dataset.py --obj_prefix optics_ --out gs_optics.npz --wavelength 532e-9 --pixel_size 8e-6 --z 0.10 --show
python gs_fresnel.py --dataset gs_optics.npz --out_prefix optics_ --max_iters 500


