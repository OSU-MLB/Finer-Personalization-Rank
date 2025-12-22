from argparse import ArgumentParser
import os
import pandas as pd
import shutil
from PIL import Image
"""
Script that loads cub data into format for deambooth, creates train.csv/gallery.csv
cub ids:
[193,194,195,196,197] wren
[143,144,145,146,147] tern
[113,114,115,116,117] sparrow
"""
def get_args()-> ArgumentParser:
    args = ArgumentParser()
    args.add_argument("--root_path", type=str, help='where cub root is', required=True)
    args.add_argument("--out_path", type=str, required=True, help="where to output")
    args.add_argument("--gallery_num", type=int, default=10)
    args.add_argument("--train_num", type=int, default=10)
    args.add_argument("--seed", type=int,default=101)
    return args.parse_args()

def load_class(group_ids, group, data_root, cub_df:pd.DataFrame, args):
    class_path = os.path.join(data_root,f'{group}')
    os.makedirs(class_path, exist_ok=True)
    cub_df['species'] = group
    cub_df = cub_df[cub_df['class_id'].isin(group_ids)]
    df_gallery = cub_df.groupby('name', group_keys=False).sample(n=args.gallery_num, random_state=args.seed)
    cub_df = cub_df[~cub_df['file_path'].isin(df_gallery['file_path'])] # remove gallery for train set

    df_train = cub_df.groupby('name', group_keys=False).sample(n=args.train_num, random_state=args.seed)

    names = df_gallery['name'].unique()
    for name in names:
        name_path = os.path.join(class_path, name)
        os.makedirs(name_path,exist_ok=True)

        train_path = os.path.join(name_path,'train')
        gallery_path = os.path.join(name_path,'gallery')
        os.makedirs(train_path,exist_ok=True)
        os.makedirs(gallery_path,exist_ok=True)

        name_gallery_df = df_gallery[df_gallery['name'] == name]
        name_train_df = df_train[df_train['name'] == name]

        gpaths = name_gallery_df['file_path'].tolist()
        tpaths = name_train_df['file_path'].tolist()

        for index,o_path in enumerate(gpaths):
            new_path = os.path.join(gallery_path,f'{index}.png')
            img = Image.open(o_path)
            img.save(new_path)
            df_gallery.loc[df_gallery['file_path'] == o_path, 'file_path'] = os.path.join('data',group,name,'gallery',f'{index}.png')
        
        for index,o_path in enumerate(tpaths):
            new_path = os.path.join(train_path,f'{index}.png')
            img = Image.open(o_path)
            img.save(new_path)
            df_train.loc[df_train['file_path'] == o_path, 'file_path'] = os.path.join('data',group,name,'train',f'{index}.png')

    return df_gallery, df_train


def main():
    args = get_args()

    root = args.root_path

    images_df = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ', names=['img_id', 'img_path'])
    labels_df = pd.read_csv(os.path.join(root, 'image_class_labels.txt'), sep=' ', names=['img_id', 'class_id'])
    split_df = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ', names=['img_id', 'is_train'])
    classes_df = pd.read_csv(os.path.join(root, 'classes.txt'), sep=' ', names=['class_id', 'class_name'])
    cub_df = images_df.merge(labels_df, on='img_id').merge(split_df, on='img_id').merge(classes_df, on='class_id')
    cub_df[['class_num', 'name']] = cub_df['class_name'].str.split('.', expand=True)
    cub_df['file_path'] = cub_df['img_path'].apply(lambda x: os.path.join(root, 'images', x))

    
    #hardcoding split
    group1_ids = [193,194,195,196,197]#wren
    group2_ids = [143,144,145,146,147]#tern
    group3_ids = [113,114,115,116,117]#sparrow

    output_path = args.out_path
    os.makedirs(output_path, exist_ok=True)
    data_path = os.path.join(output_path,'data')
    cub_df = cub_df[cub_df['class_id'].isin((group1_ids + group2_ids + group3_ids))]
    g1_gallery, g1_train = load_class(group1_ids,'wren',data_path,cub_df,args)
    g2_gallery, g2_train = load_class(group2_ids,'tern',data_path,cub_df,args)
    g3_gallery, g3_train = load_class(group3_ids,'sparrow',data_path,cub_df,args)

    gallery_df = pd.concat([g1_gallery, g2_gallery, g3_gallery], axis=0).reset_index(drop=True)

    train_df = pd.concat([g1_train, g2_train, g3_train], axis=0).reset_index(drop=True)

    gallery_csv_path = os.path.join(output_path,'gallery_data.csv')
    train_csv_path = os.path.join(output_path,'train_data.csv')
    gallery_df = gallery_df.drop(columns=['img_id','img_path','class_id','is_train','class_num','class_name'])
    train_df = train_df.drop(columns=['img_id','img_path','class_id','is_train','class_num','class_name'])

    gallery_df['file_path'] = gallery_df['file_path'].apply(lambda x: os.path.join(output_path,x))
    train_df['file_path'] = train_df['file_path'].apply(lambda x: os.path.join(output_path,x))
    gallery_df.to_csv(gallery_csv_path)
    train_df.to_csv(train_csv_path)


if __name__== "__main__":
    main()
