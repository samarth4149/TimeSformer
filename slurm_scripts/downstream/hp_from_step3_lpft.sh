#HP Finetune from Step3

# for backbone_dir in MiniSynthetic_vit_s_in1k_pt MiniSynthetic_vit_s_pass_dino_pt MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft MiniSynthetic_vit_s_scratch;
EPOCH=20
epoch_str=$(printf "%05d\n" $EPOCH)
declare -A LP_LR_DIC=( ["ucf101"]="0.1" ["hmdb51"]="0.1" ["diving48"]="0.11" ["mini_ssv2"]="0.01" ["ikea_furniture"]="0.01" ["uav"]="0.01")

# MiniSynthetic_vit_b_pass_inpk150_mae_pt divided ST for Inpk150 with PASS pretrain #DONT USER
# MiniSynthetic_vit_b_inpk150_mae_pt divided ST for Inpk150
# MiniSynthetic_vit_b_inpk150_mae_pt_tatt_only_ft for Inpk150 TATT
for backbone_dir in MiniSynthetic_vit_b_inpk150_mae_pt MiniSynthetic_vit_b_inpk150_mae_pt_tatt_only_ft;
do
#    for dset in ucf101 uav ikea_furniture diving48 hmdb51 mini_ssv2;
#    for dset in hmdb51;
    for dset in ucf101 diving48 hmdb51;
    do
         LP_LR=${LP_LR_DIC[$dset]}
         lp_lr_str=$(printf "%.0e\n" $LP_LR)
         echo $lp_lr_str
         for LR in 0.001 0.0001 0.01;
#         for LR in 0.001 0.01 0.1;
         do
           base_lr_str=$(printf "%.0e\n" $LR)
           JOB_NAME=${backbone_dir}_${dset}_lp${lp_lr_str}_ft${base_lr_str}
           LP_JOB_NAME=${backbone_dir}_${dset}_lin_probe_lr${lp_lr_str}
           echo expts/downstream/hp_tune/from_${backbone_dir}_ep${EPOCH}/lin_probe/${LP_JOB_NAME}/checkpoints/checkpoint_epoch_00020.pyth
           echo expts/downstream/hp_tune/from_${backbone_dir}_ep${EPOCH}/lpft/${JOB_NAME}/

           python tools/submit.py --cfg configs/Downstream/${dset}_finetune.yaml \
           --job_dir expts/downstream/hp_tune/from_${backbone_dir}_ep${EPOCH}/lpft/${JOB_NAME}/ \
           --num_shards 1 --num_gpus 4 --name hpft_${JOB_NAME} MODEL.MODEL_NAME vit_base_patch16_224 SOLVER.BASE_LR ${LR} \
           TRAIN.CHECKPOINT_FILE_PATH expts/downstream/hp_tune/from_${backbone_dir}_ep${EPOCH}/lin_probe/${LP_JOB_NAME}/checkpoints/checkpoint_epoch_00020.pyth \
           DATA.PATH_TO_DATA_DIR data_files/${dset}_val DATA_LOADER.NUM_WORKERS 8 \
           DATA.NUM_FRAMES 16 TRAIN.BATCH_SIZE 32 TEST.BATCH_SIZE 32 SOLVER.HEAD_LR_FACTOR ${HEAD_FACTOR}
         done
    done
done