import random
import subprocess
from collections import OrderedDict
from cvar_pyutils.ccc import submit_dependant_jobs
import torch

pretrained_model = 'ccc_models/omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_pass_config_200.pt'
dataset=  'minikinetics_ccc'
job_name =
cmd_str = f"tools/run_net.py --cfg $configs/MiniSynthetic/TimeSformer_divST_8x32_224_pass_pt.yaml NUM_GPUS 1 TRAIN.BATCH_SIZE 2 " \
          f"MODEL.MODEL_NAME vit_base_patch16_224 DATA.NUM_FRAMES 16 TIMESFORMER.PRETRAINED_MODEL {pretrained_model} " \
          f"OUTPUT_DIR expts_test/{job_name} DATA.PATH_TO_DATA_DIR data_files/{dataset}"

for cmd in run_cmds:
    jcmd = cmd[0]
    jname = cmd[1]
    jname = jname_prefix + jname + jname_suffix

    # for hydra
    jcmd += f" ++launcher.experiment_log_dir={os.path.join(EXP_DIR, jname.replace('=', ':'))}"

    if jname in existing_jobs:
        print(f'=> skipping {jname} it is already running or pending...')
    else:
        print(f'{jname} - {jcmd}\n')
        if not print_only:
            submit_dependant_jobs(
                command_to_run=jcmd,
                name=jname,
                mem=mem, num_cores=num_cores, num_gpus=num_gpus, num_nodes=num_nodes,
                duration=duration, number_of_rolling_jobs=number_of_jobs,
                gpu_type=gpu_type, out_file=f'/dccstor/lwll_data/logs/{jname}',
                verbose_output=True
            )