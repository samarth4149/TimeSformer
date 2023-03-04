import os
import sys
sys.path.append(os.path.abspath('.'))
import pandas as pd
import json
from json import JSONDecodeError

if __name__ == '__main__':
    downstream_datasets = ['ucf101', 'hmdb51', 'mini_ssv2', 'diving48', 'ikea_furniture', 'uav',]
    # pt_methods = ['MiniSynthetic_vit_s_in1k_pt', 'MiniSynthetic_vit_s_pass_dino_pt', 'MiniSynthetic_vit_s_pass_dino_pt_tatt_only_ft', 'MiniSynthetic_vit_s_scratch']

    # pt_methods = ['ccc_omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_config_200_ep20']
    pt_methods = [
        # 'MiniSynthetic_vit_b_pass_inpk150_mae_pt', # divided ST for Inpk150 with PASS pretrain #DONT USER
        'MiniSynthetic_vit_b_inpk150_mae_pt', #divided ST for Inpk150
        'MiniSynthetic_vit_b_inpk150_mae_pt_tatt_only_ft', # for Inpk150 TATT
        # 'MiniSynthetic_vit_b_inpk150_mae_pt_st_adap_only_ft',  # for Inpk150 TATT
        'MiniSynthetic_vit_b_scratch', # divided st from scratch
        'MiniSynthetic_vit_b_scratch_tatt_only_ft', # tatt from scratch
        # 'MiniSynthetic_vit_b_scratch_st_adap_only_ft', #st adapter from scratch
        'scratch',  # divided st from scratch
        'ccc_omnimae_vitb_b128_ep200_NB256__dset_mae_inpk150_config_200',
        'ccc_omnimae_vitb_b128_ep200_NB256__dset_mae_k150_config_200',
    ]
    lr_arr = ["5e-03"]
    epoch = 75
    m = 'finetune'
    hf = '1e+01'
    for eval_epoch in [4,8,12,16,20]:
        df = pd.DataFrame(columns=downstream_datasets,
                          index=pd.MultiIndex.from_product([pt_methods, lr_arr], names=['pt_method', 'lr']))
        for lr in lr_arr:
            for p in pt_methods:
                    for d in downstream_datasets:
                        if p == 'scratch':
                            epoch_string = ""
                        elif p[:4] == 'ccc_':
                            epoch_string = "_ep20"
                        else:
                            epoch_string = f"_ep{epoch}"
                        hf_string = f"_hf{hf}" if p != "scratch" else ""
                        curr_path = f'expts/downstream/from_{p}{epoch_string}/{p}_{d}_{m}_lr{lr}{hf_string}/stdout.log'
                        if not os.path.exists(curr_path):
                            print("DOESNT EXIST", curr_path)
                            continue
                        with open(curr_path, 'r') as f:
                            lines = f.read().splitlines()
                            exists = False
                            for l in lines[::-1]:
                                if 'val_epoch' not in l:
                                    continue
                                if '{' in l:
                                    try:
                                        log_dict = json.loads('{' + l.split('{')[1])
                                    except JSONDecodeError:
                                        print('Something wrong with log at {}'.format(curr_path))
                                        break
                                    if log_dict['_type']=='val_epoch' and log_dict['epoch']==f'{eval_epoch}/20':
                                        # print('Something wrong with log at {}'.format(curr_path))
                                        df.loc[(p, lr), d] = 100. - float(log_dict['top1_err'])
                                        print(p, lr, d, 100. - float(log_dict['top1_err']))
                                        exists = True
                                    else:
                                        continue
                                    break
                            if not exists:
                                print(f'Epoch {eval_epoch} doesnt exist at {curr_path}')
        df.to_csv(f'downstream_ft_results_ep{eval_epoch}.csv')