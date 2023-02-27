import os
import sys
sys.path.append(os.path.abspath('.'))
import subprocess
import pandas as pd

if __name__ == '__main__':
    datasets = ['hmdb51', 'ucf101', 'mini_ssv2', 'diving48', 'ikea_furniture', 'uav']
    backbones = ['MiniSynthetic_step3_k150inp_mae_stadapter']
    linprobe_results = pd.read_csv('expts/downstream/new_hp_results_linprobe.csv', index_col=[0, 1])
    # Get best base_lr for each dataset and backbone
    base_lrs = [0.001, 0.0001]
    
    for b in backbones:
        for d in datasets:
            for base_lr in base_lrs:
                job_name = f'{b}_{d}_lpft'
                best_lr = linprobe_results.loc[(b, slice(None)), d].idxmax()[1]
                proc_arr = [
                    'python', 'tools/submit.py', 
                    '--cfg', f'configs/Downstream/${d}_finetune.yaml',
                    '--job_dir', f'expts/downstream/hp_tune/from_{b}/{d}_lpft/base_lr_{base_lr:.0e}/',
                    '--num_shards' , '1',
                    '--num_gpus', '4',
                    '--name', job_name,
                    'MODEL.MODEL_NAME', 'vit_base_patch16_224',
                    'TRAIN.CHECKPOINT_FILE_PATH', f'expts/downstream/hp_tune/from_{b}/{d}_lin_probe/base_lr_{best_lr}/checkpoints/checkpoint_epoch_00020.pyth',
                    'SOLVER.BASE_LR', str(base_lr),
                    'TRAIN.BATCH_SIZE', '32',
                    'DATA.PATH_TO_DATA_DIR', f'/gpfs/u/home/DPLD/DPLDhwrg/scratch/TimeSformer/data_files/{d}_val']
                process = subprocess.Popen()
                process.wait()
    
    
    