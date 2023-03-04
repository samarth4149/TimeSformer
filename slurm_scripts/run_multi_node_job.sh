# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# A script with a list of commands for submitting SLURM jobs
DEFAULT_LR=0.005
DEFAULT_BATCH_SIZE=16
calc() { awk "BEGIN{ printf \"%.2f\n\", $* }"; }

#### Synthetic training
# Imagenet 1K training
# JOB_NAME=MiniSynthetic_vit_s_in1k_pt
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pt.yaml --job_dir expts/${JOB_NAME}/ --num_shards 4 --num_gpus 4 --name ${JOB_NAME}

# Scratch training
# JOB_NAME=MiniSynthetic_vit_s_scratch
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_scratch.yaml --job_dir expts/${JOB_NAME}/ --num_shards 4 --num_gpus 4 --name ${JOB_NAME}

# JOB_NAME=MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_tatt_only_ft.yaml --job_dir expts/${JOB_NAME}/ --num_shards 4 --num_gpus 4 --name ${JOB_NAME}
#
#JOB_NAME=MiniSynthetic_vit_s_pass_dino_pt_st_adapter
#python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_st_adapter.yaml --job_dir expts/${JOB_NAME}/ --num_shards 1 --num_gpus 4 --name ${JOB_NAME}

# JOB_NAME=MiniSynthetic_vit_s_pass_dino_pt
# python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt.yaml --job_dir expts/${JOB_NAME}/ --num_shards 4 --num_gpus 4 --name ${JOB_NAME}

#JOB_NAME=TimeSformer_jointST_8x32_224
#python tools/submit.py --cfg configs/Kinetics/TimeSformer_jointST_8x32_224.yaml --job_dir  /your/job/dir/${JOB_NAME}/   --num_shards 4 --partition learnfair --comment "" --name ${JOB_NAME} --use_volta32

#JOB_NAME=TimeSformer_spaceOnly_8x32_224
#python tools/submit.py --cfg configs/Kinetics/TimeSformer_spaceOnly_8x32_224.yaml --job_dir  /your/job/dir/${JOB_NAME}/   --num_shards 4 --partition learnfair --comment "" --name ${JOB_NAME} --use_volta32

#### Kinetics inference
#JOB_NAME=TimeSformer_divST_8x32_224_TEST_3clips
#python tools/submit.py --cfg configs/Kinetics/TimeSformer_divST_8x32_224_TEST.yaml --job_dir /your/job/dir/${JOB_NAME}/  --num_shards 4 --partition dev --comment "" --name ${JOB_NAME} --use_volta32


##### SSv2 training
#JOB_NAME=TimeSformer_divST_8_224
#python tools/submit.py --cfg configs/SSv2/TimeSformer_divST_8_224.yaml --job_dir  /your/job/dir/${JOB_NAME}/   --num_shards 4 --partition learnfair --comment "" --name ${JOB_NAME} --use_volta32

##### Sth-Sth_v2 inference
#JOB_NAME=TimeSformer_divST_8_224_TEST_3clips
#python tools/submit.py --cfg configs/SSv2/TimeSformer_divST_8_224_TEST.yaml --job_dir  /your/job/dir/${JOB_NAME}/   --num_shards 4 --partition learnfair --comment "" --name ${JOB_NAME} --use_volta32


# TEST Performance and Parameters

JOB_NAME=MiniSynthetic_vit_b_inpk150_mae_pt_tatt_only_ft
BATCH_SIZE=64 NUM_SHARDS=2 NUM_GPUS=4
BATCH_SIZE_PER_NODE=$((${BATCH_SIZE}/${NUM_SHARDS}))
SCALED_LR=${DEFAULT_LR}
JOB_NAME=${JOB_NAME}_b${BATCH_SIZE}_lr${SCALED_LR}
python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_tatt_only_ft.yaml --job_dir expts_new/${JOB_NAME}/ \
--num_shards ${NUM_SHARDS} --num_gpus ${NUM_GPUS} --name ${JOB_NAME} TRAIN.BATCH_SIZE ${BATCH_SIZE_PER_NODE} TEST.BATCH_SIZE ${BATCH_SIZE_PER_NODE} \
SOLVER.BASE_LR ${SCALED_LR} MODEL.MODEL_NAME vit_base_patch16_224 DATA.NUM_FRAMES 16 \
TIMESFORMER.PRETRAINED_MODEL 'ccc_models/omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_config_200.pt' \

## Scratch Divided-FT for VIT b
#JOB_NAME=MiniSynthetic_vit_b_scratch
#BATCH_SIZE=64 NUM_SHARDS=2 NUM_GPUS=4
#BATCH_SIZE_PER_NODE=$((${BATCH_SIZE}/${NUM_SHARDS}))
#python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_scratch.yaml --job_dir expts/${JOB_NAME}/ \
#--num_shards ${NUM_SHARDS} --num_gpus ${NUM_GPUS} --name ${JOB_NAME} TRAIN.BATCH_SIZE ${BATCH_SIZE_PER_NODE} TEST.BATCH_SIZE ${BATCH_SIZE_PER_NODE} \
#MODEL.MODEL_NAME vit_base_patch16_224 DATA.NUM_FRAMES 16
#
## Scratch TATT training for VIT b
#JOB_NAME=MiniSynthetic_vit_b_scratch_tatt_only_ft
#BATCH_SIZE=64 NUM_SHARDS=2 NUM_GPUS=4
#BATCH_SIZE_PER_NODE=$((${BATCH_SIZE}/${NUM_SHARDS}))
#python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_scratch.yaml --job_dir expts/${JOB_NAME}/ \
#--num_shards ${NUM_SHARDS} --num_gpus ${NUM_GPUS} --name ${JOB_NAME} TRAIN.BATCH_SIZE ${BATCH_SIZE_PER_NODE} TEST.BATCH_SIZE ${BATCH_SIZE_PER_NODE} \
#MODEL.MODEL_NAME vit_base_patch16_224 DATA.NUM_FRAMES 16 MODEL.TATT_ONLY_FT True
#
#
## Scratch ST ADAPTER training for VIT b
#JOB_NAME=MiniSynthetic_vit_b_scratch_st_adap_only_ft
#BATCH_SIZE=64 NUM_SHARDS=2 NUM_GPUS=4
#BATCH_SIZE_PER_NODE=$((${BATCH_SIZE}/${NUM_SHARDS}))
#python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_scratch.yaml --job_dir expts/${JOB_NAME}/ \
#--num_shards ${NUM_SHARDS} --num_gpus ${NUM_GPUS} --name ${JOB_NAME} TRAIN.BATCH_SIZE ${BATCH_SIZE_PER_NODE} TEST.BATCH_SIZE ${BATCH_SIZE_PER_NODE} \
#MODEL.MODEL_NAME vit_base_patch16_224 DATA.NUM_FRAMES 16 MODEL.ST_ADAPTER_ONLY_FT True TIMESFORMER.ST_ADAPTER True TIMESFORMER.ST_ADAPTER_DIM 172

## Scratch Joint ST training for VIT b
#JOB_NAME=MiniSynthetic_vit_b_joint_att
#BATCH_SIZE=16 NUM_SHARDS=2 NUM_GPUS=4
#BATCH_SIZE_PER_NODE=$((${BATCH_SIZE}/${NUM_SHARDS}))
#python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_scratch.yaml --job_dir expts/${JOB_NAME}/ \
#--num_shards ${NUM_SHARDS} --num_gpus ${NUM_GPUS} --name ${JOB_NAME} TRAIN.BATCH_SIZE ${BATCH_SIZE_PER_NODE} TEST.BATCH_SIZE ${BATCH_SIZE_PER_NODE} \
#MODEL.MODEL_NAME vit_base_patch16_224 DATA.NUM_FRAMES 16 TIMESFORMER.ATTENTION_TYPE "joint_space_time"





## inpk150 with pass pretrained. tatt
#JOB_NAME=MiniSynthetic_vit_b_pass_inpk150_mae_pt_tatt_only_ft
#BATCH_SIZE=64 NUM_SHARDS=2 NUM_GPUS=4
#BATCH_SIZE_PER_NODE=$((${BATCH_SIZE}/${NUM_SHARDS}))
#python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_tatt_only_ft.yaml --job_dir expts/${JOB_NAME}/ \
#--num_shards ${NUM_SHARDS} --num_gpus ${NUM_GPUS} --name ${JOB_NAME} TRAIN.BATCH_SIZE ${BATCH_SIZE_PER_NODE} TEST.BATCH_SIZE ${BATCH_SIZE_PER_NODE} \
#MODEL.MODEL_NAME vit_base_patch16_224 DATA.NUM_FRAMES 16 TIMESFORMER.PRETRAINED_MODEL 'ccc_models/omnimae_vitb_b128_ep200_NB256_ckpt:vitb_b512_ep400_mae_pass-load_ckpt:no_heads.yaml_dset_mae_inpk150_config_200.pt'
##
##### inpk150 with pass pretrained.
JOB_NAME=MiniSynthetic_vit_b_pass_inpk150_mae_pt
BATCH_SIZE=64 NUM_SHARDS=2 NUM_GPUS=4
BATCH_SIZE_PER_NODE=$((${BATCH_SIZE}/${NUM_SHARDS}))
python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt.yaml --job_dir expts_new/${JOB_NAME}/ \
--num_shards ${NUM_SHARDS} --num_gpus ${NUM_GPUS} --name ${JOB_NAME} TRAIN.BATCH_SIZE ${BATCH_SIZE_PER_NODE} TEST.BATCH_SIZE ${BATCH_SIZE_PER_NODE} \
MODEL.MODEL_NAME vit_base_patch16_224 DATA.NUM_FRAMES 16 TIMESFORMER.PRETRAINED_MODEL 'ccc_models/omnimae_vitb_b128_ep200_NB256_ckpt:vitb_b512_ep400_mae_pass-load_ckpt:no_heads.yaml_dset_mae_inpk150_config_200.pt'
##
##
##### inpk150. tatt
#JOB_NAME=MiniSynthetic_vit_b_inpk150_mae_pt_tatt_only_ft
#BATCH_SIZE=64 NUM_SHARDS=2 NUM_GPUS=4
#BATCH_SIZE_PER_NODE=$((${BATCH_SIZE}/${NUM_SHARDS}))
#python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt_tatt_only_ft.yaml --job_dir expts/${JOB_NAME}/ \
#--num_shards ${NUM_SHARDS} --num_gpus ${NUM_GPUS} --name ${JOB_NAME} TRAIN.BATCH_SIZE ${BATCH_SIZE_PER_NODE} TEST.BATCH_SIZE ${BATCH_SIZE_PER_NODE} \
#MODEL.MODEL_NAME vit_base_patch16_224 DATA.NUM_FRAMES 16 TIMESFORMER.PRETRAINED_MODEL 'ccc_models/omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_config_200.pt'

##
##### inpk150. divided
#JOB_NAME=MiniSynthetic_vit_b_inpk150_mae_pt
#BATCH_SIZE=64 NUM_SHARDS=2 NUM_GPUS=4
#BATCH_SIZE_PER_NODE=$((${BATCH_SIZE}/${NUM_SHARDS}))
#python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt.yaml --job_dir expts/${JOB_NAME}/ \
#--num_shards ${NUM_SHARDS} --num_gpus ${NUM_GPUS} --name ${JOB_NAME} TRAIN.BATCH_SIZE ${BATCH_SIZE_PER_NODE} TEST.BATCH_SIZE ${BATCH_SIZE_PER_NODE} \
#MODEL.MODEL_NAME vit_base_patch16_224 DATA.NUM_FRAMES 16 \
#TIMESFORMER.PRETRAINED_MODEL 'ccc_models/omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_config_200.pt' \

## inpk150 Joint attention training for VIT b
#JOB_NAME=MiniSynthetic_vit_b_inpk150_mae_pt_joint_att
#BATCH_SIZE=16 NUM_SHARDS=2 NUM_GPUS=4
#BATCH_SIZE_PER_NODE=$((${BATCH_SIZE}/${NUM_SHARDS}))
#python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt.yaml --job_dir expts/${JOB_NAME}/ \
#--num_shards ${NUM_SHARDS} --num_gpus ${NUM_GPUS} --name ${JOB_NAME} TRAIN.BATCH_SIZE ${BATCH_SIZE_PER_NODE} TEST.BATCH_SIZE ${BATCH_SIZE_PER_NODE} \
#MODEL.MODEL_NAME vit_base_patch16_224 DATA.NUM_FRAMES 16 TIMESFORMER.PRETRAINED_MODEL 'ccc_models/omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_config_200.pt' \
#TIMESFORMER.ATTENTION_TYPE "joint_space_time"


#
#### inpk150 + pass. divided
JOB_NAME=MiniSynthetic_vit_b_inpk150_pass_mae_pt
BATCH_SIZE=64 NUM_SHARDS=2 NUM_GPUS=4
BATCH_SIZE_PER_NODE=$((${BATCH_SIZE}/${NUM_SHARDS}))
python tools/submit.py --cfg configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt.yaml --job_dir expts_new/${JOB_NAME}/ \
--num_shards ${NUM_SHARDS} --num_gpus ${NUM_GPUS} --name ${JOB_NAME} TRAIN.BATCH_SIZE ${BATCH_SIZE_PER_NODE} TEST.BATCH_SIZE ${BATCH_SIZE_PER_NODE} \
MODEL.MODEL_NAME vit_base_patch16_224 DATA.NUM_FRAMES 16 \
TIMESFORMER.PRETRAINED_MODEL 'ccc_models/omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_pass_config_200.pt' \

#KINETICS
JOB_NAME=MiniKinetics_vit_b_inpk150_mae_pt
BATCH_SIZE=64 NUM_SHARDS=2 NUM_GPUS=4
BATCH_SIZE_PER_NODE=$((${BATCH_SIZE}/${NUM_SHARDS}))
SCALED_LR=${DEFAULT_LR}
JOB_NAME=${JOB_NAME}_b${BATCH_SIZE}_lr${SCALED_LR}
python tools/submit.py --cfg configs/MiniKinetics/TimeSformer_divST_8x32_224_pass_pt.yaml --job_dir expts_new/${JOB_NAME}/ \
--num_shards ${NUM_SHARDS} --num_gpus ${NUM_GPUS} --name ${JOB_NAME} TRAIN.BATCH_SIZE ${BATCH_SIZE_PER_NODE} TEST.BATCH_SIZE ${BATCH_SIZE_PER_NODE} \
SOLVER.BASE_LR ${SCALED_LR} MODEL.MODEL_NAME vit_base_patch16_224 DATA.NUM_FRAMES 16 \
TIMESFORMER.PRETRAINED_MODEL 'ccc_models/omnimae_vitb_b128_ep200_NB256__dset_mae_k150_config_200.pt' \
