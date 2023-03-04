import os
import sys
sys.path.append(os.path.abspath('.'))
import pandas as pd
import json
from json import JSONDecodeError

if __name__ == '__main__':
    downstream_datasets = ['ucf101', 'hmdb51', 'mini_ssv2', 'diving48',]
    # downstream_datasets = ['ikea_furniture', 'uav']
    # pt_methods = ['MiniSynthetic_vit_s_pass_dino_pt', 'MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft']
    pt_methods = [
        'ccc_omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_config_200_ep20',
        'ccc_omnimae_vitb_b128_ep200_NB256__dset_mae_k150_config_200_ep20',
        'ccc__ep20',
        'MiniSynthetic_vit_b_inpk150_mae_pt_ep20',
        'MiniSynthetic_vit_b_inpk150_mae_pt_tatt_only_ft_ep20'
    ]
    downstream_modes = ['lpft']
    lp_lrs = [1e-1, 1e-2]
    base_lrs = [1e-3,1e-2,1e-1]

    df = pd.DataFrame(columns=downstream_datasets,
                      index=pd.MultiIndex.from_product([pt_methods, lp_lrs, base_lrs], names=['pt_method', 'lp_lr', 'ft_lr']))
    for p in pt_methods:
        new_p = '_'.join(p.split('_')[:-1]) # remove _ep20
        # df = pd.DataFrame(columns=downstream_datasets, index=base_lrs)
        # df.index.name = 'base_lr'
        for m in downstream_modes:
            for d in downstream_datasets:
                for lplr in lp_lrs:
                    for b in base_lrs:
                        curr_path = f'expts/downstream/hp_tune/from_{p}/{m}/{new_p}_{d}_lp{lplr:.0e}_ft{b:.0e}/stdout.log'
                        if not os.path.exists(curr_path):
                            print("DOESNT EXIST", curr_path)
                            continue
                        with open(curr_path, 'r') as f:
                            lines = f.read().splitlines()
                            for l in lines[::-1]:
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
                                        # df.loc[(p, lplr, b,), d] = 0
                                    else:
                                        df.loc[(p, lplr, b,), d] = 100. - float(log_dict['top1_err'])
                                        print(p, m, d,lplr, b, 100. - float(log_dict['top1_err']))
                                    break

    df.to_csv(f'hp_lpft_results.csv')