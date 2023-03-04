# for backbone_dir in MiniSynthetic_vit_s_in1k_pt MiniSynthetic_vit_s_pass_dino_pt MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft MiniSynthetic_vit_s_scratch;
EPOCH=20
epoch_str=$(printf "%05d\n" $EPOCH)
LR=0.005 #for LIN PROBE
base_lr_str=$(printf "%.0e\n" $LR)
#omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_config_200  #inpk150
#omnimae_vitb_b128_ep200_NB256_ckpt:vitb_b512_ep400_mae_pass-load_ckpt:no_heads.yaml_dset_mae_inpk150_config_200 #pass inpk150
# omnimae_vitb_b128_ep200_NB256__dset_mae_k150_config_200 k150
# omnimae_vitb_b128_ep200_NB256_ckpt:vitb_b512_ep400_mae_ini1k-load_ckpt:no_heads.yaml_dset_mae_k150_config_200 in1k

for backbone_dir in omnimae_vitb_b128_ep200_NB256__dset_mae_k150_config_200;
#for backbone_dir in omnimae_vitb_b128_ep200_NB256__dset_mae_k150_config_200;
do
#     for dset in ucf101 diving48 hmdb51 mini_ssv2;
     for dset in hmdb51
#    for dset in ucf101 diving48 hmdb51 mini_ssv2 uav ikea_furniture;
    do
         if [ ${dset} == ikea_furniture ]; then BATCH_SIZE=16; else BATCH_SIZE=16; fi

         JOB_NAME=ccc_${backbone_dir}_${dset}_joint_att_notimeembed_lin_probe_lr${base_lr_str}
         python tools/submit.py --cfg configs/Downstream/${dset}_lin_probe.yaml \
         --job_dir  expts_new/downstream/from_ccc_${backbone_dir}_ep${EPOCH}/${JOB_NAME}/ \
         --num_shards 1 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_base_patch16_224 SOLVER.BASE_LR ${LR} \
         MODEL.PRETRAINED True TIMESFORMER.PRETRAINED_MODEL ccc_models/${backbone_dir}.pt TRAIN.CHECKPOINT_FILE_PATH "" TRAIN.FINETUNE False \
         TRAIN.BATCH_SIZE ${BATCH_SIZE} TEST.BATCH_SIZE ${BATCH_SIZE} DATA.NUM_FRAMES 16 \
         TIMESFORMER.TUBLET_DIM 2 TIMESFORMER.ATTENTION_TYPE "joint_space_time" TIMESFORMER.USE_TIME_EMBED False

         HEAD_FACTOR=10.
         head_factor_str=$(printf "%.0e\n" $HEAD_FACTOR)
         JOB_NAME=ccc_${backbone_dir}_${dset}_joint_att_notimeembed_finetune_lr${base_lr_str}_hf${head_factor_str}
         python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml \
         --job_dir  expts_new/downstream/from_ccc_${backbone_dir}_ep${EPOCH}/${JOB_NAME}/ \
         --num_shards 1 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_base_patch16_224 SOLVER.BASE_LR ${LR} \
         MODEL.PRETRAINED True TIMESFORMER.PRETRAINED_MODEL ccc_models/${backbone_dir}.pt TRAIN.CHECKPOINT_FILE_PATH "" TRAIN.FINETUNE False \
         TRAIN.BATCH_SIZE ${BATCH_SIZE} TEST.BATCH_SIZE ${BATCH_SIZE} DATA.NUM_FRAMES 16 \
         TIMESFORMER.TUBLET_DIM 2 TIMESFORMER.ATTENTION_TYPE "joint_space_time" TIMESFORMER.USE_TIME_EMBED False

    done
done