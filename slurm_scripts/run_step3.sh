# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# A script with a list of commands for submitting SLURM jobs

#### Synthetic training
# Imagenet 1K training
# JOB_NAME=MiniSynthetic_vit_s_in1k_pt
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pt.yaml --job_dir expts/${JOB_NAME}/ --num_shards 4 --num_gpus 4 --name ${JOB_NAME}

# Scratch training
# JOB_NAME=MiniSynthetic_vit_s_scratch
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_scratch.yaml --job_dir expts/${JOB_NAME}/ --num_shards 4 --num_gpus 4 --name ${JOB_NAME}

# JOB_NAME=MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_tatt_only_ft.yaml --job_dir expts/${JOB_NAME}/ --num_shards 4 --num_gpus 4 --name ${JOB_NAME}

JOB_NAME=MiniSynthetic_step3_pass_mae_pt_ft
python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt.yaml --job_dir expts/${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch_16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/mae_pass_400.pt TRAIN.BATCH_SIZE 16

JOB_NAME=MiniSynthetic_step3_pass_mae_pt_tatt_only_ft
python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_tatt_only_ft.yaml --job_dir expts/${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch_16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/mae_pass_400.pt TRAIN.BATCH_SIZE 16

JOB_NAME=MiniSynthetic_step3_pass_mae_pt_stadapter
python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_st_adapter.yaml --job_dir expts/${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME} DATA.NUM_FRAMES 16 MODEL.MODEL_NAME vit_base_patch_16_224 TIMESFORMER.PRETRAINED_MODEL /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/ccc_models/mae_pass_400.pt TRAIN.BATCH_SIZE 16

# JOB_NAME=MiniSynthetic_vit_s_pass_dino_pt
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt.yaml --job_dir expts/${JOB_NAME}/ --num_shards 4 --num_gpus 4 --name ${JOB_NAME}

#JOB_NAME=TimeSformer_jointST_8x32_224
#python tools/submit.py --cfg configs/Kinetics/TimeSformer_jointST_8x32_224.yaml --job_dir  /your/job/dir/${JOB_NAME}/   --num_shards 4 --partition learnfair --comment "" --name ${JOB_NAME} --use_volta32

#JOB_NAME=TimeSformer_spaceOnly_8x32_224
#python tools/submit.py --cfg configs/Kinetics/TimeSformer_spaceOnly_8x32_224.yaml --job_dir  /your/job/dir/${JOB_NAME}/   --num_shards 4 --partition learnfair --comment "" --name ${JOB_NAME} --use_volta32

#### Kinetics inference
#JOB_NAME=TimeSformer_divST_8x32_224_TEST_3clips
#python tools/submit.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224_TEST.yaml --job_dir /your/job/dir/${JOB_NAME}/  --num_shards 4 --partition dev --comment "" --name ${JOB_NAME} --use_volta32


##### SSv2 training
#JOB_NAME=TimeSformer_divST_8_224
#python tools/submit.py --cfg configs/SSv2/TimeSformer_divST_8_224.yaml --job_dir  /your/job/dir/${JOB_NAME}/   --num_shards 4 --partition learnfair --comment "" --name ${JOB_NAME} --use_volta32

##### Sth-Sth_v2 inference
#JOB_NAME=TimeSformer_divST_8_224_TEST_3clips
#python tools/submit.py --cfg configs/SSv2/TimeSformer_divST_8_224_TEST.yaml --job_dir  /your/job/dir/${JOB_NAME}/   --num_shards 4 --partition learnfair --comment "" --name ${JOB_NAME} --use_volta32
