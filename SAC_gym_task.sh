python runner.py main/SAC_aux_gym.py --config configs/SAC_gym_exp.yaml --exp-dir results/CMC_SAC_phi_gauss_aux0.0_0.0 --beta 0.0 --aux_coef 0.0
python runner.py main/SAC_aux_gym.py --config configs/SAC_gym_exp.yaml --exp-dir results/CMC_SAC_phi_gauss_aux0.1_0.0 --beta 0.0 --aux_coef 0.1
python runner.py main/SAC_aux_gym.py --config configs/SAC_gym_exp.yaml --exp-dir results/CMC_SAC_phi_gauss_aux0.0_10.0 --beta 10 --aux_coef 0.0
python runner.py main/SAC_aux_gym.py --config configs/SAC_gym_exp.yaml --exp-dir results/CMC_SAC_phi_gauss_aux0.1_10.0 --beta 10 --aux_coef 0.1