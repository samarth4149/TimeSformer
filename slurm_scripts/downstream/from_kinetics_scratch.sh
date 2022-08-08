#### Kinetics training
JOB_NAME=ucf101_finetune
python tools/submit.py --cfg configs/Downstream/ucf101_finetune.yaml --job_dir  expts/downstream/from_kinetics_scratch/${JOB_NAME}/  --num_shards 1 --num_gpus 4 --name ${JOB_NAME} TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/MiniKinetics_TimeSformer_divST_8x32_224_scratch/checkpoints/checkpoint_epoch_00030.pyth

JOB_NAME=ucf101_lin_probe
python tools/submit.py --cfg configs/Downstream/ucf101_lin_probe.yaml --job_dir  expts/downstream/from_kinetics_scratch/${JOB_NAME}/  --num_shards 1 --num_gpus 4 --name ${JOB_NAME} TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/MiniKinetics_TimeSformer_divST_8x32_224_scratch/checkpoints/checkpoint_epoch_00030.pyth
