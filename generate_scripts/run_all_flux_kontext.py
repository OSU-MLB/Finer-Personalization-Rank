from argparse import ArgumentParser
import pandas as pd
from utils import get_prompts
import os
import torch
from PIL import Image
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import sys
from huggingface_hub import snapshot_download
import random


"""script that generates all flux-kontext images given data folder"""

TARGET_PATH = "path/to/flux/Kontext"
NEGATIVE_PROMPT = "low quality, cropped, deformed, text, ugly, blurry, zoomed in, not showing whole subject, cartoony, unrealistic, lowres, not showing whole subject, animated"



def getargs():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help='root folder that holds data and csv files')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--prompt_type", type=str, default="reid")
    parser.add_argument("--skip_already_run",action='store_true',default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--skip",default=False, action='store_true') # dont use
    return parser.parse_args()


def main():
    args = getargs()
    random.seed(101)
    root_path = args.root
    data_path = os.path.join(root_path, "data")
    local_model_dir = TARGET_PATH
    repo_id = "black-forest-labs/FLUX.1-Kontext-dev"

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_model_dir,
        local_dir_use_symlinks=False
    )
    if not args.skip:
        pipe = FluxKontextPipeline.from_pretrained(
            local_model_dir,
            torch_dtype=torch.bfloat16
        ).to(args.device)


    #save csv file in form for miewID
    df = pd.DataFrame(columns=['name', 'file_path', 'species','ref_path' ])

    species_folders = os.listdir(data_path)

    for specie_folder in species_folders:
        print(f'starting species : {specie_folder}')

        species_path = os.path.join(data_path, specie_folder)
        name_folders = os.listdir(species_path)
        for name in name_folders:
            print(f'starting subject {name}')
            name_path =  os.path.join(species_path,name)

            save_path = os.path.join(name_path, 'generated_images')
            if(args.skip_already_run and os.path.exists(os.path.join(save_path,'9.png'))):
                print(f'skipping gen of {name}')
                rows = []
                for img_save_path in os.listdir(save_path):
                    rows.append({'name': name, 'file_path':os.path.join(save_path,img_save_path), 'species':specie_folder})
                    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
                continue
            #get all prior imgs
            train_path = os.path.join(name_path, 'train')
            paths = []
            priors_paths = os.listdir(train_path)
            for img_path in priors_paths:
                paths.append(os.path.join(train_path,img_path))

            prompts = get_prompts(specie_folder, prompt_type=args.prompt_type)


            prompts = get_prompts(specie_folder, prompt_type=args.prompt_type)
            gens = []
            sample_paths = []
            for idx in range(0,len(prompts), args.batch_size):
                prompts_batch = prompts[idx:min((idx + args.batch_size), len(prompts))]
                pths = paths
                if(len(sample_paths) > len(prompts_batch)):
                    pths = random.sample(pths, len(prompts_batch))
                sample_paths.extend(pths)
                pimgs = []
                if not args.skip:
                    for p in pths:
                        pimg += Image.open(p).convert('RGB').resize((512,512))
                    gen = pipe(image=pimgs, 
                            prompt=prompts_batch,
                            guidance_scale=5,
                            num_inference_steps=30,
                            negative_prompt=NEGATIVE_PROMPT).images
                    gens += gen
            #save 
            os.makedirs(save_path, exist_ok=True)
            save_paths = []
            if args.skip:
                print('hey')
                for index in range(len(prompts)):
                    save_path_img = os.path.join(save_path, f'{index}.png')
                    save_paths.append(save_path_img)
            else:
                for index,gen_img in enumerate(gens):
                    save_path_img = os.path.join(save_path, f'{index}.png')
                    save_paths.append(save_path_img)
    
                    if not args.skip:
                        gen_img.save(save_path_img)

            rows = []
            print(len(sample_paths))
            for img_save_path,pth in zip(save_paths,sample_paths):
                rows.append({'name': name, 'file_path':img_save_path, 'species':specie_folder,  'ref_path':pth})
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    df.to_csv(os.path.join(root_path, 'generated_data.csv'), index=False)

if __name__=="__main__":
    main()