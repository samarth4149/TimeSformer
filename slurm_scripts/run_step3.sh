# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# A script with a list of commands for submitting SLURM jobs

# Synthetic training
# # Imagenet 1K MAE pretrained
# JOB_NAME=MiniSynthetic_step3_in1k_mae_pt_ft
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt.yaml --job_dir expts/step3/${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/mae_in1k_400.pt TRAIN.BATCH_SIZE 16 TRAIN.DEL_INTERMEDIATE_CHECKPOINTS False TRAIN.CHECKPOINT_PERIOD 1

# JOB_NAME=MiniSynthetic_step3_in1k_mae_pt_tatt_only_ft
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_tatt_only_ft.yaml --job_dir expts/step3//${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/mae_in1k_400.pt TRAIN.BATCH_SIZE 16 TRAIN.DEL_INTERMEDIATE_CHECKPOINTS False TRAIN.CHECKPOINT_PERIOD 1

# # JOB_NAME=MiniSynthetic_step3_in1k_mae_pt_stadapter
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_st_adapter.yaml --job_dir expts/step3//${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/mae_in1k_400.pt TRAIN.BATCH_SIZE 16 TRAIN.DEL_INTERMEDIATE_CHECKPOINTS False TRAIN.CHECKPOINT_PERIOD 1

# # PASS MAE pretrained
# JOB_NAME=MiniSynthetic_step3_pass_mae_pt_ft
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt.yaml --job_dir expts/step3/${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/mae_pass_400.pt TRAIN.BATCH_SIZE 16 TRAIN.DEL_INTERMEDIATE_CHECKPOINTS False TRAIN.CHECKPOINT_PERIOD 1

# JOB_NAME=MiniSynthetic_step3_pass_mae_pt_tatt_only_ft
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_tatt_only_ft.yaml --job_dir expts/step3//${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/mae_pass_400.pt TRAIN.BATCH_SIZE 16 TRAIN.DEL_INTERMEDIATE_CHECKPOINTS False TRAIN.CHECKPOINT_PERIOD 1

# # JOB_NAME=MiniSynthetic_step3_pass_mae_pt_stadapter
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_st_adapter.yaml --job_dir expts/step3//${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/mae_pass_400.pt TRAIN.BATCH_SIZE 16 TRAIN.DEL_INTERMEDIATE_CHECKPOINTS False TRAIN.CHECKPOINT_PERIOD 1

# # PASS - K150inp
# JOB_NAME=MiniSynthetic_step3_pass_mae_k150inp_mae_ft
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt.yaml --job_dir expts/step3/${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/omnimae_vitb_b128_ep200_NB256_ckpt:vitb_b512_ep400_mae_pass-load_ckpt:no_heads.yaml_dset_mae_inpk150_config_200.pt TRAIN.BATCH_SIZE 16 TRAIN.DEL_INTERMEDIATE_CHECKPOINTS False TRAIN.CHECKPOINT_PERIOD 5

# JOB_NAME=MiniSynthetic_step3_pass_mae_k150inp_mae_tatt_only_ft
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_tatt_only_ft.yaml --job_dir expts/step3//${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/omnimae_vitb_b128_ep200_NB256_ckpt:vitb_b512_ep400_mae_pass-load_ckpt:no_heads.yaml_dset_mae_inpk150_config_200.pt TRAIN.BATCH_SIZE 16 TRAIN.DEL_INTERMEDIATE_CHECKPOINTS False TRAIN.CHECKPOINT_PERIOD 5

# JOB_NAME=MiniSynthetic_step3_pass_mae_k150inp_mae_stadapter
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_st_adapter.yaml --job_dir expts/step3//${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/omnimae_vitb_b128_ep200_NB256_ckpt:vitb_b512_ep400_mae_pass-load_ckpt:no_heads.yaml_dset_mae_inpk150_config_200.pt TRAIN.BATCH_SIZE 16 TRAIN.DEL_INTERMEDIATE_CHECKPOINTS False TRAIN.CHECKPOINT_PERIOD 5

# # - K150inp
# JOB_NAME=MiniSynthetic_step3_k150inp_mae_ft
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt.yaml --job_dir expts/step3/${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_config_200.pt TRAIN.BATCH_SIZE 16 TRAIN.DEL_INTERMEDIATE_CHECKPOINTS False TRAIN.CHECKPOINT_PERIOD 5

# JOB_NAME=MiniSynthetic_step3_k150inp_mae_tatt_only_ft
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_tatt_only_ft.yaml --job_dir expts/step3//${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_config_200.pt TRAIN.BATCH_SIZE 16 TRAIN.DEL_INTERMEDIATE_CHECKPOINTS False TRAIN.CHECKPOINT_PERIOD 5

# JOB_NAME=MiniSynthetic_step3_k150inp_mae_stadapter
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_st_adapter.yaml --job_dir expts/step3//${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_config_200.pt TRAIN.BATCH_SIZE 16 TRAIN.DEL_INTERMEDIATE_CHECKPOINTS False TRAIN.CHECKPOINT_PERIOD 5

# Imagenet-1K - K150 MAE
JOB_NAME=MiniKinetics_step3_in1k_mae_k150_mae_ft
python tools/submit.py --cfg configs/MiniKinetics/TimeSformer_divST_8x32_224_pt.yaml --job_dir expts/step3/${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/omnimae_vitb_b128_ep200_NB256_ckpt:vitb_b512_ep400_mae_ini1k-load_ckpt:no_heads.yaml_dset_mae_k150_config_200.pt TRAIN.BATCH_SIZE 16 TRAIN.DEL_INTERMEDIATE_CHECKPOINTS False TRAIN.CHECKPOINT_PERIOD 5

# - K150 MAE
JOB_NAME=MiniKinetics_step3_k150_mae_ft
python tools/submit.py --cfg configs/MiniKinetics/TimeSformer_divST_8x32_224_pt.yaml --job_dir expts/step3/${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/omnimae_vitb_b128_ep200_NB256__dset_mae_k150_config_200.pt TRAIN.BATCH_SIZE 16 TRAIN.DEL_INTERMEDIATE_CHECKPOINTS False TRAIN.CHECKPOINT_PERIOD 5