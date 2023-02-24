#!/bin/bash
#### Kinetics training
# for dset in hmdb51 ucf101;
for dset in hmdb51 ucf101 uav ikea_furniture diving48 mini_ssv2;
# for dset in mini_ssv2;
do
    JOB_NAME=${dset}_finetune
    python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml --job_dir  expts/downstream/from_synthetic_only_tatt_pass_pt/${JOB_NAME}/  --num_shards 2 --num_gpus 4 --name ${JOB_NAME} MODEL.PRETRAINED False TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/MiniSynthetic_TimeSformer_divST_8x32_224_pass_dino_pt_vit_s_tatt_only_ft/checkpoints/checkpoint_epoch_00030.pyth MODEL.MODEL_NAME vit_small_patch16_224

    JOB_NAME=${dset}_lin_probe
    python tools/submit.py --cfg configs/Downstream/${dset}_lin_probe.yaml --job_dir  expts/downstream/from_synthetic_only_tatt_pass_pt/${JOB_NAME}/  --num_shards 2 --num_gpus 4 --name ${JOB_NAME} MODEL.PRETRAINED False TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/MiniSynthetic_TimeSformer_divST_8x32_224_pass_dino_pt_vit_s_tatt_only_ft/checkpoints/checkpoint_epoch_00030.pyth MODEL.MODEL_NAME vit_small_patch16_224;
done