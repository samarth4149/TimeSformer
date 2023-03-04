# for backbone_dir in MiniSynthetic_vit_s_in1k_pt MiniSynthetic_vit_s_pass_dino_pt MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft MiniSynthetic_vit_s_scratch;
LR=0.005 #for LIN PROBE
base_lr_str=$(printf "%.0e\n" $LR)

#  for dset in ucf101 diving48 hmdb51 mini_ssv2 uav ikea_furniture;
   for dset in uav ikea_furniture mini_ssv2;
#  for dset in ucf101;
  do
       if [ ${dset} == ikea_furniture ]; then BATCH_SIZE=16; else BATCH_SIZE=32; fi

       JOB_NAME=scratch_${dset}_finetune_lr${base_lr_str}
       python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml \
       --job_dir  expts/downstream/from_scratch/${JOB_NAME}/ \
       --num_shards 1 --num_gpus 4 --name ${JOB_NAME} MODEL.MODEL_NAME vit_base_patch16_224 SOLVER.BASE_LR ${LR} \
       TIMESFORMER.PRETRAINED_MODEL "" TRAIN.CHECKPOINT_FILE_PATH "" TRAIN.FINETUNE False DATA.NUM_FRAMES 16 \
       TRAIN.BATCH_SIZE ${BATCH_SIZE} TEST.BATCH_SIZE ${BATCH_SIZE}
  done