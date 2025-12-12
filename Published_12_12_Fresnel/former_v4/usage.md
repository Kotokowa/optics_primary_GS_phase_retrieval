python make_object_cli_v4.py --mode snellen --prefix snellen_ --show
python make_dataset_cli_v2.py --obj_prefix snellen_ --out gs_snellen.npz --z1 0.10 --z2 0.13 --show
python gs_fresnel_raar.py --dataset gs_snellen.npz --out_prefix snellen_ --restarts 3 --max_iters 500 --init random

python make_object_cli_v4.py --mode optics --prefix optics_ --optics_font 80 --fiducials --show
python make_dataset_cli_v2.py --obj_prefix optics_ --out gs_optics.npz --z1 0.10 --z2 0.13 --show
python gs_fresnel_raar.py --dataset gs_optics.npz --out_prefix optics_ --restarts 8 --max_iters 800 # --init random --beta 0.85 --gamma_sen 0.9
