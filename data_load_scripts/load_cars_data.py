from argparse import ArgumentParser
import os
import pandas as pd
import shutil
from PIL import Image
from datasets import load_dataset
"""
Script that loads cub data into format for deambooth, creates train.csv/gallery.csv, loads proper specific cars ids:[1,2,3,4,6] + [11,12,13,18,19] + [25,27,28,33,34] --> acura audi bmw

"""
def get_args()-> ArgumentParser:
    args = ArgumentParser()
    args.add_argument("--out_path", type=str, required=True, help="where to output")
    args.add_argument("--gallery_num", type=int, default=10)
    args.add_argument("--train_num", type=int, default=10)
    args.add_argument("--seed", type=int,default=101)
    return args.parse_args()

def load_class(group_ids, group, data_root,ds,args):
    df_train = pd.DataFrame(columns=['name','species','file_path'])
    df_gallery = pd.DataFrame(columns=['name','species','file_path'])
    class_path = os.path.join(data_root,group)
    os.makedirs(class_path, exist_ok=True)
    for id in group_ids:
        name = str(id)
        name_path = os.path.join(class_path,name)

        train_path = os.path.join(name_path,'train')
        gallery_path = os.path.join(name_path,'gallery')
        os.makedirs(train_path,exist_ok=True)
        os.makedirs(gallery_path,exist_ok=True)

        imgs = [s['image'] for s in ds if s['label'] == id]
        gallery_imgs = imgs[:args.gallery_num]
        train_imgs = imgs[args.gallery_num: args.train_num + args.gallery_num]

        rows = []
        for index, img in enumerate(gallery_imgs):
            save_path = os.path.join(gallery_path,f'{index}.png')
            img.save(save_path)
            file_path = os.path.join('data',group,name,'gallery',f'{index}.png')
            row = {'name': name, 'species':group, 'file_path': file_path }
            rows.append(row)
        df_gallery = pd.concat([df_gallery, pd.DataFrame(rows)], ignore_index=True)


        rows = []
        for index, img in enumerate(train_imgs):
            save_path = os.path.join(train_path,f'{index}.png')
            img.save(save_path)
            file_path = os.path.join('data',group,name,'train',f'{index}.png')
            row = {'name': name, 'species':group, 'file_path': file_path }
            rows.append(row)
        df_train = pd.concat([df_train, pd.DataFrame(rows)], ignore_index=True)


    return df_train, df_gallery





def main():
    args = get_args()

    out_path = args.out_path
    os.makedirs(out_path,exist_ok=True)
    ds = load_dataset("Donghyun99/Stanford-Cars")
    test_set = ds['test']
    # acura audi bmw
    group1_ids = [1,2,3,4,6] + [11,12,13,18,19] + [25,27,28,33,34]


    
    data_path = os.path.join(out_path,'data')
    os.makedirs(data_path,exist_ok=True)

    g1_train, g1_gallery = load_class(group_ids=group1_ids, group='car' , data_root=data_path, ds=test_set,args=args)
    g1_train['file_path'] = g1_train['file_path'].apply(lambda x: os.path.join(out_path,x))
    g1_gallery['file_path'] = g1_gallery['file_path'].apply(lambda x: os.path.join(out_path,x))
    g1_train.to_csv(os.path.join(out_path,'train_data.csv'))
    g1_gallery.to_csv(os.path.join(out_path,'gallery_data.csv'))
    

    

if __name__== "__main__":
    main()
