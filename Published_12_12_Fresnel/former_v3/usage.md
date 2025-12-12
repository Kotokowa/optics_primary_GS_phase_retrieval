python make_object_cli.py --mode optics --prefix optics_ --show
python make_dataset_cli.py --obj_prefix optics_ --out gs_optics.npz --z1 0.10 --z2 0.13 --show
python gs_fresnel_cli.py --dataset gs_optics.npz --out_prefix optics_ --max_iters 700 --init random --restarts 5 --gamma_obj 0.9 --gamma_sen 0.9


python make_object_cli.py --mode snellen --prefix snellen_ --show
python make_dataset_cli.py --obj_prefix snellen_ --out gs_snellen.npz --z1 0.10 --z2 0.13 --show
python gs_fresnel_cli.py --dataset gs_snellen.npz --out_prefix snellen_ --max_iters 500
