# for backbone_dir in MiniSynthetic_vit_s_in1k_pt MiniSynthetic_vit_s_pass_dino_pt MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft MiniSynthetic_vit_s_scratch;
EPOCH=75
epoch_str=$(printf "%05d\n" $EPOCH)
LR=0.005 #for LIN PROBE
base_lr_str=$(printf "%.0e\n" $LR)
# MiniSynthetic_vit_b_pass_inpk150_mae_pt divided ST for Inpk150 with PASS pretrain #DONT USER
# MiniSynthetic_vit_b_inpk150_mae_pt divided ST for Inpk150
# MiniSynthetic_vit_b_inpk150_mae_pt_tatt_only_ft for Inpk150 TATT
# MiniSynthetic_vit_b_inpk150_mae_pt_st_adap_only_ft for Inpk150 ST
# MiniSynthetic_vit_b_scratch divided st from scratch
# MiniSynthetic_vit_b_scratch_tatt_only_ft tatt from scratch
# MiniSynthetic_vit_b_scratch_st_adap_only_ft st adap from scratch

#for backbone_dir in MiniSynthetic_vit_b_inpk150_mae_pt MiniSynthetic_vit_b_inpk150_mae_pt_tatt_only_ft;
for backbone_dir in MiniSynthetic_vit_b_scratch MiniSynthetic_vit_b_scratch_tatt_only_ft;
#for backbone_dir in MiniSynthetic_vit_b_inpk150_mae_pt MiniSynthetic_vit_b_inpk150_mae_pt_tatt_only_ft MiniSynthetic_vit_b_scratch_tatt_only_ft MiniSynthetic_vit_b_scratch;
do
#     for dset in hmdb51 ucf101 uav ikea_furniture diving48 mini_ssv2;
#    for dset in hmdb51 ucf101 diving48 mini_ssv2;
    for dset in mini_ssv2 uav ikea_furniture;
    do
         if [ ${dset} == ikea_furniture ]; then BATCH_SIZE=16; else BATCH_SIZE=32; fi

         JOB_NAME=${backbone_dir}_${dset}_lin_probe_lr${base_lr_str}
         python tools/submit.py --cfg configs/Downstream/${dset}_lin_probe.yaml \
         --job_dir  expts/downstream/from_${backbone_dir}_ep${EPOCH}/${JOB_NAME}/ \
         --num_shards 1 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_base_patch16_224 SOLVER.BASE_LR ${LR} \
         TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/expts/${backbone_dir}/checkpoints/checkpoint_epoch_${epoch_str}.pyth \
         DATA.NUM_FRAMES 16 TRAIN.BATCH_SIZE ${BATCH_SIZE} TEST.BATCH_SIZE ${BATCH_SIZE}

         HEAD_FACTOR=10.
         head_factor_str=$(printf "%.0e\n" $HEAD_FACTOR)
         JOB_NAME=${backbone_dir}_${dset}_finetune_lr${base_lr_str}_hf${head_factor_str}
         python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml \
         --job_dir  expts/downstream/from_${backbone_dir}_ep${EPOCH}/${JOB_NAME}/ \
         --num_shards 1 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_base_patch16_224 SOLVER.BASE_LR ${LR} \
         TRAIN.CHECKPOINT_FILE_PATH /gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/expts/${backbone_dir}/checkpoints/checkpoint_epoch_${epoch_str}.pyth \
         DATA.NUM_FRAMES 16 TRAIN.BATCH_SIZE ${BATCH_SIZE} TEST.BATCH_SIZE ${BATCH_SIZE} SOLVER.HEAD_LR_FACTOR ${HEAD_FACTOR}
    done
done