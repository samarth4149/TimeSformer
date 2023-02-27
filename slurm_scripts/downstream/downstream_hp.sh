# for backbone_dir in MiniSynthetic_vit_s_in1k_pt MiniSynthetic_vit_s_pass_dino_pt MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft MiniSynthetic_vit_s_scratch;
for backbone_dir in MiniKinetics_step3_k150_mae_ft;
do
    for dset in ucf101 uav ikea_furniture diving48 hmdb51 mini_ssv2;
    # for dset in hmdb51;
    do
        for base_lr in 0.1 0.01;
        do
            # for head_factor in 1. 10.;
            # do
            base_lr_str=$(printf "%.0e\n" $base_lr) 
            # head_factor_str=$(printf "%.0e\n" $head_factor) 
            # JOB_NAME=${backbone_dir}_${dset}_finetune_base_lr_${base_lr_str}_head_factor_${head_factor_str}
            # python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml --job_dir expts/downstream/hp_tune/from_${backbone_dir}/${dset}_finetune/base_lr_${base_lr_str}_head_factor_${head_factor_str}  --num_shards 2 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_small_patch16_224 TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/${backbone_dir}/checkpoints/checkpoint_epoch_00075.pyth SOLVER.BASE_LR ${base_lr} SOLVER.HEAD_LR_FACTOR ${head_factor}

            # JOB_NAME=${backbone_dir}_${dset}_finetune_base_lr_${base_lr_str}_head_factor_${head_factor_str}
            # python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml --job_dir expts/downstream/hp_tune/from_${backbone_dir}/${dset}_lpft/base_lr_${base_lr_str}_head_factor_${head_factor_str}  --num_shards 2 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_small_patch16_224 TRAIN.CHECKPOINT_FILE_PATH expts/downstream/hp_tune/from_${backbone_dir}/${dset}_linprobe/base_lr_1e-02/checkpoints/checkpoint_epoch_00020.pyth SOLVER.BASE_LR ${base_lr} SOLVER.HEAD_LR_FACTOR ${head_factor}

            JOB_NAME=${backbone_dir}_${dset}_linprobe_base_lr_${base_lr_str}
            python tools/submit.py --cfg configs/Downstream/${dset}_lin_probe.yaml --job_dir expts/downstream/hp_tune/from_${backbone_dir}/${dset}_linprobe/base_lr_${base_lr_str}  --num_shards 1 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_small_patch16_224 TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/step3/${backbone_dir}/checkpoints/checkpoint_epoch_00075.pyth SOLVER.BASE_LR ${base_lr} TRAIN.BATCH_SIZE 32 DATA.PATH_TO_DATA_DIR /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/data_files/${dset}_val

            # JOB_NAME=${backbone_dir}_${dset}_lin_probe
            # python tools/submit.py --cfg configs/Downstream/${dset}_lin_probe.yaml --job_dir  expts/downstream/from_${backbone_dir}/${JOB_NAME}/  --num_shards 2 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_small_patch16_224 TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/${backbone_dir}/checkpoints/checkpoint_epoch_00075.pyth;
            # done
        done
    done
done