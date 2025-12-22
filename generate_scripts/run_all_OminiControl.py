from argparse import ArgumentParser
import pandas as pd
from utils import get_prompts, get_class_prompt, get_instance_prompt
import os
import torch
from diffusers.pipelines import FluxPipeline
from PIL import Image
import random

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from OminiControl.omini.pipeline.flux_omini import Condition, generate, seed_everything


"""generates all OminiControl images given data folder"""

def getargs():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help='root folder that holds data and csv files')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--num_prior_imgs", type=int,default=1)
    parser.add_argument("--prompt_type", type=str, default="reid")
    return parser.parse_args()


def main():
    args = getargs()
    root_path = args.root
    data_path = os.path.join(root_path, "data")
    #os.makedirs(root_path, exist_ok=True)

    #start ominiControl pipeline
    pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
    )
    pipe = pipe.to(args.device)
    pipe.load_lora_weights(
        "Yuanshi/OminiControl",
        weight_name=f"omini/subject_512.safetensors",
        adapter_name="subject",
    )
    seed_everything()
    random.seed(101)

    #save csv file in form
    df = pd.DataFrame(columns=['name', 'file_path', 'species'])

    species_folders = os.listdir(data_path)

    for specie_folder in species_folders:
        print(f'starting species : {specie_folder}')

        species_path = os.path.join(data_path, specie_folder)
        name_folders = os.listdir(species_path)
        for name in name_folders:
            name_path =  os.path.join(species_path,name)

            #get all prior imgs
            train_path = os.path.join(name_path, 'train')
            paths = []
            priors_paths = os.listdir(train_path)
            for img_path in priors_paths:
                paths.append(os.path.join(train_path,img_path))

            prompts = get_prompts(specie_folder, prompt_type=args.prompt_type)

            #gen img
            save_path = os.path.join(name_path, 'generated_images')
            save_paths = []
            sample_paths = []

            os.makedirs(save_path, exist_ok=True)
            for index,prompt in enumerate(prompts):
                path = random.sample(paths, args.num_prior_imgs)[0]
                sample_paths.append(path)
                
                img = Image.open(path).convert('RGB').resize((512,512))
                condition = Condition(img, "subject",position_delta=(0, 32))
                gen = generate(
                    pipe,
                    prompt=prompt,
                    conditions=condition,
                    num_inference_steps=30,
                    height=512,
                    width=512,
                ).images[0]
                #save 
                save_path_img = os.path.join(save_path, f'{index}.png')
                save_paths.append(save_path_img)
                gen.save(save_path_img)

            rows = []
            for img_save_path,pth in zip(save_paths,sample_paths):
                rows.append({'name': name, 'file_path':img_save_path, 'species':specie_folder, 'ref_path':pth})
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    df.to_csv(os.path.join(root_path, 'generated_data.csv'), index=False)

if __name__=="__main__":
    main()