for backbone_dir in MiniSynthetic_vit_s_in1k_pt MiniSynthetic_vit_s_pass_dino_pt MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft MiniSynthetic_vit_s_scratch;
do
    for dset in ucf101 uav ikea_furniture diving48 hmdb51 mini_ssv2;
    do
        JOB_NAME=${backbone_dir}_${dset}_finetune
        python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml --job_dir  expts/downstream/from_${backbone_dir}/${JOB_NAME}/  --num_shards 2 --num_gpus 4 --name ${JOB_NAME} TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/${backbone_dir}/checkpoints/checkpoint_epoch_00075.pyth

        JOB_NAME=${backbone_dir}_${dset}_lin_probe
        python tools/submit.py --cfg configs/Downstream/${dset}_lin_probe.yaml --job_dir  expts/downstream/from_${backbone_dir}/${JOB_NAME}/  --num_shards 2 --num_gpus 4 --name ${JOB_NAME} TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDsmms/scratch/projects/TimeSformer/expts/${backbone_dir}/checkpoints/checkpoint_epoch_00075.pyth;
    done
done