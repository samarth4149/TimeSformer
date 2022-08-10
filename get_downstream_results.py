import os
import sys
sys.path.append(os.path.abspath('.'))
import pandas as pd
import json

if __name__ == '__main__':
    downstream_datasets = ['ucf101', 'hmdb51', 'mini_ssv2', 'diving48', 'ikea_furniture', 'uav',]
    pt_methods = ['kinetics_pt', 'kinetics_scratch']
    downstream_modes = ['lin_probe', 'finetune']
    df = pd.DataFrame(columns=downstream_datasets, index=pd.MultiIndex.from_product([pt_methods, downstream_modes], names=['pt_method', 'mode']))
    
    for p in pt_methods:
        for m in downstream_modes:
            for d in downstream_datasets:
                curr_path = f'expts/downstream/from_{p}/{d}_{m}/stdout.log'
                with open(curr_path, 'r') as f:
                    lines = f.read().splitlines()
                    for l in lines[::-1]:
                        if '{' in l:
                            log_dict = json.loads('{' + l.split('{')[1])
                            if log_dict['_type']!='val_epoch':
                                print('Something wrong with log at {}'.format(curr_path))
                            else:
                                df.loc[(p, m), d] = 100. - float(log_dict['top1_err'])
                            break                            

    df.to_csv('downstream_results.csv')