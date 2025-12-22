from argparse import ArgumentParser
import os
import pandas as pd
import shutil
"""
Script that loads wildlife10k data into a standard format for miewID. Still needs to go through format_data.py/ viewpoint check for some datasets, such as CowDataset.
"""
def get_args()-> ArgumentParser:
    parser = ArgumentParser(description="Load and format data")
    parser.add_argument("--wildlife10k_path", type=str,required=True, help="Path to the Wildlife10K dataset folder")
    parser.add_argument("--wildlife10k_csv_path", type=str, required=True, help="Path to the Wildlife10K CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the formatted data")
    parser.add_argument("--species_list", type=str, nargs='+', required=True, help="List of species to include {form --species_list SPECIE1 SPECIE2 .... }")
    parser.add_argument("--dataset_list", type=str, nargs='+', required=True, help="List of datasets to include from wildlifereid10k {from --dataset_list DATASET1 DATASET2 .... }") #['AAUZebraFish' 'AerialCattle2017' 'AmvrakikosTurtles' 'ATRW' 'BelugaID''BirdIndividualID' 'CatIndividualImages' 'Chicks4FreeID' 'CowDataset''Cows2021' 'CTai' 'CZoo' 'DogFaceNet' 'FriesianCattle2015''FriesianCattle2017' 'Giraffes' 'GiraffeZebraID' 'HyenaID2022' 'IPanda50''LeopardID2022' 'MPDD' 'MultiCamCows2024' 'NDD20' 'NyalaData''OpenCows2020' 'PolarBearVidID' 'PrimFace' 'ReunionTurtles' 'SealID''SeaStarReID2023' 'SeaTurtleID2022' 'SMALST' 'SouthernProvinceTurtles''StripeSpotter' 'WhaleSharkID' 'ZakynthosTurtles' 'ZindiTurtleRecall']
    parser.add_argument("--max_per_subject", type=int, default=10, help="Max number of images to use per subject for training set")
    parser.add_argument("--min_per_subject", type=int, default=4, help="Min number of images to use per subject for training set")
    parser.add_argument("--per_species", type=int, default=10, help="number of subjects per species to use")
    parser.add_argument("--seed",type=int,default=42, help="Random seed")
    parser.add_argument("--max_gallery_per_subject", type=int, default=15, help="Max number of images to use per subject for gallery set")
    parser.add_argument("--min_gallery_per_subject", type=int, default=5, help="Min number of images to use per subject for gallery set")
    return parser.parse_args()


def main():
    args = get_args()
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    #load data and proccess data
    df = pd.read_csv(args.wildlife10k_csv_path)
    df = df[df['species'].isin(args.species_list)]
    df = df[df['dataset'].isin(args.dataset_list)]
    #moving data into miewID format
    df.rename(columns={'identity':'name', 'orientation': 'viewpoint', 'path':'file_path'}, inplace=True)

    #training data for model
    data_path = os.path.join(output_path, "data")
    os.makedirs(data_path, exist_ok=True)
    train_df = df.groupby('name').filter(lambda x: len(x) >= args.min_per_subject + args.min_gallery_per_subject) #need to have enough data for both training and gallery
    train_df = train_df.groupby('name', group_keys=False).apply(lambda x: x.sample(n=min(len(x)-args.min_gallery_per_subject, args.max_per_subject), random_state=args.seed))
    all_sampled_names = []
    species = args.species_list
    for specie in species:#takes largest #per_species subjects per specie
        specie_df = train_df[train_df['species']==specie]
        name_counts = specie_df['name'].value_counts()
        top_names = name_counts.head(args.per_species).index.tolist()
        all_sampled_names.extend(top_names)
    train_df = train_df[df['name'].isin(all_sampled_names)]
    df = df[df['file_path'].isin(train_df['file_path']) == False] #remove training data from df so it isnt used in gallery or contrastive examples

    #gallery data for miewID
    gallery_df = df[df['name'].isin(all_sampled_names)]
    gallery_df = gallery_df.groupby('name', group_keys=False).apply(lambda x: x.sample(n=min(len(x), args.max_gallery_per_subject), random_state=args.seed))
    df = df[df['file_path'].isin(gallery_df['file_path']) == False] #remove gallery data from df so it isnt used in contrastive examples


    #copy training/gallery data into species/name/train/img#.png format
    for specie in species:
        specie_path = os.path.join(data_path,specie)
        os.makedirs(specie_path, exist_ok=True)
        specie_train_df = train_df[train_df['species']==specie]
        specie_gallery_df = gallery_df[gallery_df['species']==specie]
        specie_names = specie_train_df['name'].unique()
        for name in specie_names:
            name_path = os.path.join(specie_path,name)
            os.makedirs(name_path, exist_ok=True)
            name_train_df = specie_train_df[specie_train_df['name']==name]
            orig_train_img_paths = name_train_df['file_path'].tolist()

            train_path = os.path.join(name_path, "train")
            os.makedirs(train_path, exist_ok=True)
            for i, orig_img_path in enumerate(orig_train_img_paths):
                img_new_filename = f"{i}.png"
                new_img_path = os.path.join(train_path,img_new_filename)
                orig_full_path = os.path.join(args.wildlife10k_path, orig_img_path)
                shutil.copyfile(orig_full_path, new_img_path)
                train_df.loc[train_df['file_path'] == orig_img_path, 'file_path'] = new_img_path

            name_gallery_df = specie_gallery_df[specie_gallery_df['name']==name]
            orig_gallery_img_paths = name_gallery_df['file_path'].tolist()
            gallery_path = os.path.join(name_path, "gallery")
            os.makedirs(gallery_path, exist_ok=True)
            for i, orig_img_path in enumerate(orig_gallery_img_paths):
                img_new_filename = f"{i}.png"
                new_img_path = os.path.join(gallery_path,img_new_filename)
                orig_full_path = os.path.join(args.wildlife10k_path, orig_img_path)
                shutil.copyfile(orig_full_path, new_img_path)
                gallery_df.loc[gallery_df['file_path'] == orig_img_path, 'file_path'] = new_img_path
    train_df.to_csv(os.path.join(output_path, "train_data.csv"), index=False)
    gallery_df.to_csv(os.path.join(output_path, "gallery_data.csv"), index=False)

    #copy rest of files into extra folder

    extra_path = os.path.join(output_path, "extra")
    os.makedirs(extra_path, exist_ok=True)
    df = df.groupby('name').filter(lambda x: len(x) >= 1)
    all_names = df['name'].unique()
    for name in all_names:
        name_path = os.path.join(extra_path,name)
        os.makedirs(name_path, exist_ok=True)
        name_df = df[df['name']==name]
        orig_img_paths = name_df['file_path'].tolist()
        for i, orig_img_path in enumerate(orig_img_paths):
            img_new_filename = f"{i}.png"
            new_img_path = os.path.join(name_path,img_new_filename)
            orig_full_path = os.path.join(args.wildlife10k_path, orig_img_path)
            shutil.copyfile(orig_full_path, new_img_path)
            df.loc[df['file_path'] == orig_img_path, 'file_path'] = new_img_path
    df.to_csv(os.path.join(output_path, "extra_data.csv"), index=False)
    
    




        











if __name__ == "__main__":
    main()