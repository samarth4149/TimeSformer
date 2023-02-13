# for backbone_dir in MiniSynthetic_vit_s_in1k_pt MiniSynthetic_vit_s_pass_dino_pt MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft MiniSynthetic_vit_s_scratch;
for backbone_dir in MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft MiniSynthetic_vit_s_pass_dino_pt;
do
    # for dset in ucf101 uav ikea_furniture diving48 hmdb51 mini_ssv2;
    for dset in hmdb51;
    do
        for base_lr in 1.e-4;
        do
            for head_factor in 10.;
            do
                base_lr_str=$(printf "%.0e\n" $base_lr) 
                head_factor_str=$(printf "%.0e\n" $head_factor) 
                # JOB_NAME=${backbone_dir}_${dset}_finetune_base_lr_${base_lr_str}_head_factor_${head_factor_str}
                # python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml --job_dir expts/downstream/hp_tune/from_${backbone_dir}/${dset}_finetune/base_lr_${base_lr_str}_head_factor_${head_factor_str}  --num_shards 2 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_small_patch16_224 TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/${backbone_dir}/checkpoints/checkpoint_epoch_00075.pyth SOLVER.BASE_LR ${base_lr} SOLVER.HEAD_LR_FACTOR ${head_factor}

                JOB_NAME=${backbone_dir}_${dset}_finetune_base_lr_${base_lr_str}_head_factor_${head_factor_str}
                python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml --job_dir expts/downstream/hp_tune/from_${backbone_dir}/${dset}_lpft/base_lr_${base_lr_str}_head_factor_${head_factor_str}  --num_shards 2 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_small_patch16_224 TRAIN.CHECKPOINT_FILE_PATH expts/downstream/hp_tune/from_${backbone_dir}/${dset}_linprobe/base_lr_1e-02/checkpoints/checkpoint_epoch_00020.pyth SOLVER.BASE_LR ${base_lr} SOLVER.HEAD_LR_FACTOR ${head_factor}

                # JOB_NAME=${backbone_dir}_${dset}_linprobe_base_lr_${base_lr_str}
                # python tools/submit.py --cfg configs/Downstream/${dset}_lin_probe.yaml --job_dir expts/downstream/hp_tune/from_${backbone_dir}/${dset}_linprobe/base_lr_${base_lr_str}  --num_shards 2 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_small_patch16_224 TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/${backbone_dir}/checkpoints/checkpoint_epoch_00075.pyth SOLVER.BASE_LR ${base_lr}

                # JOB_NAME=${backbone_dir}_${dset}_lin_probe
                # python tools/submit.py --cfg configs/Downstream/${dset}_lin_probe.yaml --job_dir  expts/downstream/from_${backbone_dir}/${JOB_NAME}/  --num_shards 2 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_small_patch16_224 TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/${backbone_dir}/checkpoints/checkpoint_epoch_00075.pyth;
            done
        done
    done
done