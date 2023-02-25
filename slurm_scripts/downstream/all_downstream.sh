# for backbone_dir in MiniSynthetic_vit_s_in1k_pt MiniSynthetic_vit_s_pass_dino_pt MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft MiniSynthetic_vit_s_scratch;
for backbone_dir in MiniKinetics_step3_k150_mae_ft;
do
    for dset in ucf101 uav ikea_furniture diving48 hmdb51 mini_ssv2;
    # for dset in hmdb51;
    do
        # JOB_NAME=${backbone_dir}_${dset}_finetune
        # python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml --job_dir  expts/downstream/from_${backbone_dir}/${JOB_NAME}/  --num_shards 1 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_base_patch16_224 TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/step3/${backbone_dir}/checkpoints/checkpoint_epoch_00075.pyth TRAIN.BATCH_SIZE 32

        # JOB_NAME=${backbone_dir}_epoch25_${dset}_lin_probe
        # python tools/submit.py --cfg configs/Downstream/${dset}_lin_probe.yaml --job_dir  expts/downstream/from_${backbone_dir}/${JOB_NAME}/  --num_shards 1 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_base_patch16_224 TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/step3/${backbone_dir}/checkpoints/checkpoint_epoch_00025.pyth TRAIN.BATCH_SIZE 32 SOLVER.BASE_LR 0.01

        JOB_NAME=${backbone_dir}_epoch25_${dset}_lpft
        python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml --job_dir  expts/downstream/from_${backbone_dir}/${JOB_NAME}/  --num_shards 1 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_base_patch16_224 TRAIN.CHECKPOINT_FILE_PATH expts/downstream/from_${backbone_dir}/${backbone_dir}_epoch25_${dset}_lin_probe/checkpoint_epoch_00020.pyth TRAIN.BATCH_SIZE 32 SOLVER.BASE_LR 0.001
    done
done