#HP Finetune from Step2

# for backbone_dir in MiniSynthetic_vit_s_in1k_pt MiniSynthetic_vit_s_pass_dino_pt MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft MiniSynthetic_vit_s_scratch;
EPOCH=20
epoch_str=$(printf "%05d\n" $EPOCH)

#omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_config_200  #inpk150
#omnimae_vitb_b128_ep200_NB256_ckpt:vitb_b512_ep400_mae_pass-load_ckpt:no_heads.yaml_dset_mae_inpk150_config_200 #pass inpk150
# omnimae_vitb_b128_ep200_NB256__dset_mae_k150_config_200 k150
# omnimae_vitb_b128_ep200_NB256_ckpt:vitb_b512_ep400_mae_ini1k-load_ckpt:no_heads.yaml_dset_mae_k150_config_200 in1k

for backbone_dir in omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_config_200 omnimae_vitb_b128_ep200_NB256__dset_mae_k150_config_200;
do
#    for dset in hmdb51 ucf101 diving48  mini_ssv2 uav ikea_furniture ;
    for dset in hmdb51 ucf101 diving48 mini_ssv2 uav ikea_furniture;
    do
         for LR in 0.001 0.01 0.0001;
         do
           HEAD_FACTOR=10
           base_lr_str=$(printf "%.0e\n" $LR)
           JOB_NAME=ccc_${backbone_dir}_${dset}_finetune_lr${base_lr_str}_hf${HEAD_FACTOR}
           python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml \
           --job_dir  expts/downstream/hp_tune/from_ccc_${backbone_dir}_ep${EPOCH}/finetune/${JOB_NAME}/ \
           --num_shards 1 --num_gpus 4 --name hp_${JOB_NAME} MODEL.MODEL_NAME vit_base_patch16_224 SOLVER.BASE_LR ${LR} \
           MODEL.PRETRAINED True TIMESFORMER.PRETRAINED_MODEL ccc_models/${backbone_dir}.pt TRAIN.CHECKPOINT_FILE_PATH "" TRAIN.FINETUNE False \
           DATA.PATH_TO_DATA_DIR data_files/${dset}_val DATA_LOADER.NUM_WORKERS 8 \
           TRAIN.BATCH_SIZE 32 TEST.BATCH_SIZE 32 DATA.NUM_FRAMES 16
         done
    done
done