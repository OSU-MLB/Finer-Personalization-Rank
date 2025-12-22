import pandas as pd
import os
from argparse import ArgumentParser

"""Script to be run after load_data.py to remove any subjects that have been removed checking viewpoint manually. Will update csv files accordingly.
"""

def get_args()->ArgumentParser:
    args = ArgumentParser()
    args.add_argument("--root_path", type=str, required=True, help="path to find loaded data")
    return args.parse_args()

def main():
    args = get_args()
    root_path = args.root_path

    train_df = pd.read_csv(os.path.join(root_path, "train_data.csv"))
    gallery_df = pd.read_csv(os.path.join(root_path, "gallery_data.csv"))
    train_img_paths = train_df['file_path'].tolist()
    gallery_img_paths = gallery_df['file_path'].tolist()
    for train_img_path in train_img_paths:
        if not os.path.exists(os.path.join(root_path,train_img_path)):
            print(f"train img path {train_img_path} does not exist, removing from dataframe")
            train_df = train_df[train_df['file_path'] != train_img_path]
            continue
        #adjust path to be relative to root path
        train_df.loc[train_df['file_path'] == train_img_path, 'file_path'] = os.path.relpath(train_img_path, root_path)
        
    train_df.to_csv(os.path.join(root_path, "train_data.csv"), index=False)
    for gallery_img_path in gallery_img_paths:
        if not os.path.exists(os.path.join(root_path,gallery_img_path)):
            print(f"gallery img path {gallery_img_path} does not exist, removing from dataframe")
            gallery_df = gallery_df[gallery_df['file_path'] != gallery_img_path]
            continue
        #adjust path to be relative to root path
        gallery_df.loc[gallery_df['file_path'] == gallery_img_path, 'file_path'] = os.path.relpath(gallery_img_path, root_path)
    gallery_df.to_csv(os.path.join(root_path, "gallery_data.csv"), index=False)

if __name__ == "__main__":
    main()