import os
import sys
sys.path.append(os.path.abspath('.'))
import pandas as pd
import json
from json import JSONDecodeError

if __name__ == '__main__':
    # downstream_datasets = ['ucf101', 'hmdb51', 'mini_ssv2', 'diving48', 'ikea_furniture', 'uav',]
    downstream_datasets = ['hmdb51']
    pt_methods = ['MiniSynthetic_vit_s_pass_dino_pt', 'MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft']
    downstream_modes = ['linprobe']
    base_lrs = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 3.e-5, 1.e-5, 3.e-6, 1.e-6]
    # head_factors = [1., 10.]
    
    for p in pt_methods:
        for m in downstream_modes:
            for d in downstream_datasets:
                # df = pd.DataFrame(columns=downstream_datasets, index=pd.MultiIndex.from_product([base_lrs], names=['base_lr']))
                df = pd.DataFrame(columns=downstream_datasets, index=base_lrs)
                df.index.name = 'base_lr'
                for b in base_lrs:
                    # for h in head_factors:
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
                                    df.loc[b, d] = 100. - float(log_dict['top1_err'])
                                break                            

                df.to_csv(f'expts/downstream/hp_tune/from_{p}/{d}_{m}/hp_results.csv')