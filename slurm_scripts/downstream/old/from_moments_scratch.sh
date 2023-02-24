#!/bin/bash
#### Kinetics training
for dset in ucf101 uav ikea_furniture diving48 hmdb51 mini_ssv2;
# for dset in mini_ssv2;
do
    JOB_NAME=${dset}_finetune
    python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml --job_dir  expts/downstream/from_moments_scratch/${JOB_NAME}/  --num_shards 2 --num_gpus 4 --name ${JOB_NAME} TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/MiniMoments_TimeSformer_divST_8x32_224_scratch/checkpoints/checkpoint_epoch_00030.pyth

    JOB_NAME=${dset}_lin_probe
    python tools/submit.py --cfg configs/Downstream/${dset}_lin_probe.yaml --job_dir  expts/downstream/from_moments_scratch/${JOB_NAME}/  --num_shards 2 --num_gpus 4 --name ${JOB_NAME} TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/MiniMoments_TimeSformer_divST_8x32_224_scratch/checkpoints/checkpoint_epoch_00030.pyth;
done