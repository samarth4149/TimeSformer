import os
import sys
sys.path.append(os.path.abspath('.'))
import pandas as pd
import json
from json import JSONDecodeError

if __name__ == '__main__':
    downstream_datasets = ['ucf101', 'hmdb51', 'mini_ssv2', 'diving48', 'ikea_furniture', 'uav',]
    # downstream_datasets = ['ikea_furniture', 'uav']
    # pt_methods = ['MiniSynthetic_vit_s_pass_dino_pt', 'MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft']
    pt_methods = [
        'ccc_omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_config_200_ep20',
        'ccc_omnimae_vitb_b128_ep200_NB256__dset_mae_k150_config_200_ep20',
        'ccc__ep20',
        'MiniSynthetic_vit_b_inpk150_mae_pt_ep20',
        'MiniSynthetic_vit_b_inpk150_mae_pt_tatt_only_ft_ep20',
        'MiniSynthetic_vit_b_scratch'
    ]
    downstream_modes = ['lin_probe']
    # base_lrs = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 3.e-5, 1.e-5, 3.e-6, 1.e-6]
    # base_lrs = [1.e-3, 3.e-4, 1.e-4, 3.e-5, 1.e-5, 3.e-6]
    # head_factors = [1., 10.]
    base_lrs = [1e-3,1e-2,1e-1, 1]

    df = pd.DataFrame(columns=downstream_datasets,
                      index=pd.MultiIndex.from_product([pt_methods, base_lrs], names=['pt_method', 'base_lr']))

    for p in pt_methods:
        new_p = '_'.join(p.split('_')[:-1]) # remove _ep20
        # df = pd.DataFrame(columns=downstream_datasets, index=base_lrs)
        # df.index.name = 'base_lr'
        for m in downstream_modes:
            for d in downstream_datasets:
                for b in base_lrs:
                    curr_path = f'expts/downstream/hp_tune/from_{p}/{m}/{new_p}_{d}_{m}_lr{b:.0e}/stdout.log'
                    if not os.path.exists(curr_path):
                        print("DOESNT EXIST", curr_path)
                        continue
                    with open(curr_path, 'r') as f:
                        lines = f.read().splitlines()
                        for l in lines[::-1]:
                            if 'val_epoch' not in l:
                                continue
                            if '{' in l:
                                try:
                                    log_dict = json.loads('{' + l.split('{')[1])
                                except JSONDecodeError:
                                    print('JSONDecodeError: Something wrong with log at {}'.format(curr_path))
                                    break
                                if log_dict['_type']!='val_epoch' and log_dict['epoch']!='20/20':
                                    # print(l)
                                    # print(log_dict)
                                    print('Something wrong with log at {}'.format(curr_path))
                                    print(m, d, b)
                                    # df.loc[(p,b,), d] = 0
                                else:
                                    df.loc[(p,b,), d] = 100. - float(log_dict['top1_err'])
                                    print(p, m, d, b, 100. - float(log_dict['top1_err']))
                                break

    df.to_csv(f'hp_lp_results.csv')