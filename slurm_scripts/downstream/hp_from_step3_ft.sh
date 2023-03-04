#HP Finetune from Step2

# for backbone_dir in MiniSynthetic_vit_s_in1k_pt MiniSynthetic_vit_s_pass_dino_pt MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft MiniSynthetic_vit_s_scratch;
EPOCH=20
epoch_str=$(printf "%05d\n" $EPOCH)

# MiniSynthetic_vit_b_pass_inpk150_mae_pt divided ST for Inpk150 with PASS pretrain #DONT USER
# MiniSynthetic_vit_b_inpk150_mae_pt divided ST for Inpk150
# MiniSynthetic_vit_b_inpk150_mae_pt_tatt_only_ft for Inpk150 TATT
# MiniSynthetic_vit_b_inpk150_mae_pt_st_adap_only_ft for Inpk150 ST
# MiniSynthetic_vit_b_scratch divided st from scratch
# MiniSynthetic_vit_b_scratch_tatt_only_ft tatt from scratch
# MiniSynthetic_vit_b_scratch_st_adap_only_ft st adap from scratch



for backbone_dir in MiniSynthetic_vit_b_inpk150_mae_pt MiniSynthetic_vit_b_inpk150_mae_pt_tatt_only_ft MiniSynthetic_vit_b_scratch MiniSynthetic_vit_b_scratch_tatt_only_ft;
#for backbone_dir in MiniSynthetic_vit_b_inpk150_mae_pt_tatt_only_ft;

do
#    for dset in ucf101 uav ikea_furniture diving48 hmdb51 mini_ssv2;
#    for dset in diving48 ikea_furniture uav;
#    for dset in diving48 ikea_furniture uav;
    for dset in ucf101 hmdb51 diving48 mini_ssv2 ikea_furniture uav;
#    for dset in hmdb51 diving48
    do
         for LR in 0.01 0.001 0.0001;
#         for LR in 0.1;
         do
           base_lr_str=$(printf "%.0e\n" $LR)
           HEAD_FACTOR=10
           JOB_NAME=${backbone_dir}_${dset}_finetune_lr${base_lr_str}_hf${HEAD_FACTOR}
           python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml \
           --job_dir  expts/downstream/hp_tune/from_${backbone_dir}_ep${EPOCH}/finetune/${JOB_NAME}/ \
           --num_shards 1 --num_gpus 4 --name hp_${JOB_NAME} MODEL.MODEL_NAME vit_base_patch16_224 SOLVER.BASE_LR ${LR} \
           TRAIN.CHECKPOINT_FILE_PATH expts/${backbone_dir}/checkpoints/checkpoint_epoch_00075.pyth \
           DATA.PATH_TO_DATA_DIR data_files/${dset}_val DATA_LOADER.NUM_WORKERS 8 DATA.NUM_FRAMES 16 \
           TRAIN.BATCH_SIZE 32 TEST.BATCH_SIZE 32
         done
    done
done