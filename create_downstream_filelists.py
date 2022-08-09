import os
import sys
sys.path.append(os.path.abspath('.'))
import csv
from pathlib import Path
import yaml


if __name__ == '__main__':
    with open('configs/downstream_datasets.yaml', 'r') as f:        
        datasets = yaml.load(f, Loader=yaml.FullLoader)
    
    for data_name, dataset in datasets.items():
        if data_name != 'hmdb51':
            continue
        print('Processing {}'.format(data_name))
        data_base = Path(dataset['path'])
        out_path = Path('data_files') / data_name
        os.makedirs(out_path, exist_ok=True)
        for mode in ['train', 'val']:
            fout = open(out_path / f'{mode}.csv', 'w')
            filelist = dataset[f'{mode}_file']
            with open(filelist, 'r') as f, open(out_path / 'missing.txt', 'w') as fout_missing:
                reader = csv.reader(f, delimiter=dataset['separator'])
                
                for row in reader:
                    fname = row[0]
                    if len(row) == 4:
                        label = row[3]
                    else:
                        label = row[1]
                    
                    curr_path = data_base / ('/'.join(fname.split('/')[dataset['strip_prefix']:]) + dataset['add_suffix'])
                    if not curr_path.exists():
                        print('Missing file in {}: {}'.format(data_name, curr_path))
                        print(curr_path, file=fout_missing)
                    print(f'{curr_path} {label}', file=fout)
            
            fout.close()