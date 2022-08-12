import sys
import os
from pathlib import Path
import csv
from tqdm import tqdm


if __name__ == '__main__':
    
    data_file_base = Path('/gpfs/u/home/DPLD/DPLDpndr/scratch-shared/datasets/synapt/miniSynthetic_files')
    data_file_paths = [data_file_base / 'sim150_1000e_train.txt', data_file_base / 'sim150_val.txt']
    vid_base = Path('/gpfs/u/home/DPLD/DPLDpndr/scratch-shared/datasets/synapt/synthetic_data/videos')

    out_dir = Path('data_files/minisim')
    os.makedirs(out_dir, exist_ok=True)
    
    for df in data_file_paths:
        if 'train' in df.name:
            outfile = out_dir / 'train.csv'
        else:
            outfile = out_dir / 'val.csv'
        with open(df, 'r') as f, open(outfile, 'w') as fout:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                fname = row[0]
                label = row[3]
                curr_path = vid_base / ('/'.join(fname.split('/')[6:]) + '.mp4')
                if not os.path.exists(curr_path):
                    raise Exception(f'{curr_path} does not exist')
                
                print(f'{curr_path};{label}', file=fout)
    