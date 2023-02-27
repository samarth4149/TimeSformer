import os
import sys
sys.path.append(os.path.abspath('.'))
import pandas as pd
import json
from json import JSONDecodeError
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    downstream_datasets = ['ucf101', 'hmdb51', 'mini_ssv2', 'diving48', 'ikea_furniture', 'uav',]
    # downstream_datasets = ['hmdb51']
    pt_methods = ['MiniSynthetic_step3_k150inp_mae_stadapter']
    downstream_modes = ['linprobe']
    # downstream_modes = ['lpft']
    # base_lrs = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 3.e-5, 1.e-5, 3.e-6, 1.e-6]
    base_lrs = [0.1, 0.01]
    # head_factors = [1., 10.]
    
    for m in downstream_modes:
        df = pd.DataFrame(columns=downstream_datasets, index=pd.MultiIndex.from_product([pt_methods, base_lrs], names=['pt_method', 'base_lr']))
        if os.path.exists and not args.overwrite:
            df = pd.read_csv(f'expts/downstream/new_hp_results_{m}.csv', index_col=[0,1])
        
        for p in pt_methods:
            # df = pd.DataFrame(columns=downstream_datasets, index=pd.MultiIndex.from_product([base_lrs, head_factors], names=['base_lr', 'head_factors']))
            for d in downstream_datasets:
                # df = pd.DataFrame(columns=downstream_datasets, index=base_lrs)
                # df.index.name = 'base_lr'
                for b in base_lrs:
                    curr_path = f'expts/downstream/hp_tune/from_{p}/{d}_{m}/base_lr_{b:.0e}/stdout.log'
                    with open(curr_path, 'r') as f:
                        lines = f.read().splitlines()
                        for l in lines[::-1]:
                            if '{' in l:
                                try:
                                    log_dict = json.loads('{' + l.split('{')[1])
                                except JSONDecodeError:
                                    print('Something wrong with log at {}'.format(curr_path))
                                    break
                                if log_dict['_type']!='val_epoch' and log_dict['epoch']!='20/20':
                                    print('Something wrong with log at {}'.format(curr_path))
                                else:
                                    df.loc[(p, b), d] = 100. - float(log_dict['top1_err'])
                                break                            

        df.to_csv(f'expts/downstream/new_hp_results_{m}.csv')