import sys
import os
from pathlib import Path
import csv
from tqdm import tqdm


if __name__ == '__main__':
    
    data_file_base = Path('/gpfs/u/home/DPLD/DPLDsmms/scratch-shared/datasets/Moments')
    data_file_paths = [data_file_base / 'moments150_kinetics_overlap_train.txt', data_file_base / 'moments150_kinetics_overlap_val.txt']
    vid_base = Path('/gpfs/u/home/DPLD/DPLDsmms/scratch-shared/datasets/Moments/videos')

    out_dir = Path('data_files/minimoments')
    os.makedirs(out_dir, exist_ok=True)
    
    for df in data_file_paths:
        if 'train' in df.name:
            outfile = out_dir / 'train.csv'
            split = 'train'
        else:
            outfile = out_dir / 'val.csv'
            split = 'val'
        with open(df, 'r') as f, open(outfile, 'w') as fout:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                fname = row[0]
                label = row[3]
                curr_path = vid_base / split / fname
                if not os.path.exists(curr_path):
                    raise Exception(f'{curr_path} does not exist')
                
                print(f'{curr_path};{label}', file=fout)
    