import pandas as pd
import os
import shutil
from argparse import ArgumentParser

"""Script that copies data in standard format and makes file paths absolute for eval purposes
"""

def get_args()->ArgumentParser:
    args = ArgumentParser()
    args.add_argument("--root_path", type=str, required=True, help="path to find loaded data")
    args.add_argument("--output_path", type=str, required=True, help="path to copy of loaded data with contrastive examples")
    args.add_argument("--seed", type=int, default=42)
    return args.parse_args()


def main():
    args = get_args()
    root_path = args.root_path
    
    shutil.copytree(root_path, args.output_path, dirs_exist_ok=True) #copy over data to new folder so original is not messed with
    root_path = args.output_path #work in new copied folder
    for csv_file in ["train_data.csv", "gallery_data.csv"]:
        df = pd.read_csv(os.path.join(root_path, csv_file))
        img_paths = df['file_path'].tolist()
        for img_path in img_paths:
            abs_path = os.path.join(root_path, img_path)
            df.loc[df['file_path'] == img_path, 'file_path'] = abs_path
        df.to_csv(os.path.join(root_path, csv_file), index=False)



    



if __name__ == "__main__":
    main()