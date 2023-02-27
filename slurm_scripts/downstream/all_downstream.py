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
    
    for b in backbones:
        for d in datasets:
            job_name = f'{b}_{d}_linprobe'
            best_lr = linprobe_results.loc[(b, slice(None)), d].idxmax()[1]
            proc_arr = [
                'python', 'tools/submit.py', 
                '--cfg', f'configs/Downstream/${d}_linprobe.yaml',
                '--job_dir', f'expts/downstream/from_{b}/{job_name}/',
                '--num_shards' , '1',
                '--num_gpus', '4',
                '--name', job_name,
                'MODEL.MODEL_NAME', 'vit_base_patch16_224',
                'TRAIN.CHECKPOINT_FILE_PATH', f'expts/step3/{b}/checkpoints/checkpoint_epoch_00075.pyth',
                'TRAIN.BATCH_SIZE', '32',
                'SOLVER.BASE_LR', str(best_lr)]
            process = subprocess.Popen(proc_arr)
            process.wait()
    
    
    