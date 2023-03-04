#HP Finetune from Step2

# for backbone_dir in MiniSynthetic_vit_s_in1k_pt MiniSynthetic_vit_s_pass_dino_pt MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft MiniSynthetic_vit_s_scratch;
EPOCH=20
epoch_str=$(printf "%05d\n" $EPOCH)


#  for dset in ucf101 uav ikea_furniture diving48 hmdb51 mini_ssv2;
#  for dset in diving48 ikea_furniture uav;
#ucf101 hmdb51 diving48 ikea_furniture mini_ssv2 uav
  for dset in ucf101 hmdb51;
  do
       for LR in 0.0001 0.001;
#         for LR in 1.;
       do
         base_lr_str=$(printf "%.0e\n" $LR)
         JOB_NAME=scratch_${dset}_finetune_lr${base_lr_str}
         python tools/submit.py --cfg configs/Downstream/${dset}_lin_probe.yaml \
         --job_dir  expts/downstream/hp_tune/from_scratch_ep${EPOCH}/finetune/${JOB_NAME}/  \
         --num_shards 1 --num_gpus 4 --name hp_${JOB_NAME} MODEL.MODEL_NAME vit_base_patch16_224 SOLVER.BASE_LR ${LR} \
         TIMESFORMER.PRETRAINED_MODEL "" TRAIN.CHECKPOINT_FILE_PATH "" TRAIN.FINETUNE False \
         DATA.PATH_TO_DATA_DIR data_files/${dset}_val DATA_LOADER.NUM_WORKERS 8 \
         TRAIN.BATCH_SIZE 32 TEST.BATCH_SIZE 32 DATA.NUM_FRAMES 16

#         JOB_NAME =ccc_${backbone_dir}_${dset}_lin_probe_lr${base_lr_str}_ft
       done
  done