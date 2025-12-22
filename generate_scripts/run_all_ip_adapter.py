from argparse import ArgumentParser
import pandas as pd
from utils import get_prompts, get_class_prompt, get_instance_prompt
import os
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import sys
from huggingface_hub import snapshot_download
import random



"""generates all ip-adapter images given data folder"""


SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
TARGET_PATH = "/your/path"

NEGATIVE_PROMPT = "low quality, cropped, deformed, text, ugly, blurry, zoomed in, not showing whole subject, cartoony, unrealistic, lowres, not showing whole subject, animated"



def getargs():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help='root folder that holds data and csv files')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--prompt_type", type=str, default="reid")
    parser.add_argument("--adapter_scale", type=int,default=0.6)
    return parser.parse_args()


def main():
    args = getargs()
    root_path = args.root
    data_path = os.path.join(root_path, "data")

    random.seed(101)
    pipe = AutoPipelineForText2Image.from_pretrained(
        TARGET_PATH, 
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(args.device) 

    pipe.load_ip_adapter(
        "h94/IP-Adapter", 
        subfolder="sdxl_models", 
        weight_name="ip-adapter_sdxl.bin" 
    )

    pipe.set_ip_adapter_scale(args.adapter_scale) 

    #save csv file in form for miewID
    df = pd.DataFrame(columns=['name', 'file_path', 'species'])

    species_folders = os.listdir(data_path)

    for specie_folder in species_folders:
        print(f'starting species : {specie_folder}')

        species_path = os.path.join(data_path, specie_folder)
        name_folders = os.listdir(species_path)
        for name in name_folders:
            print(f'starting subject {name}')
            name_path =  os.path.join(species_path,name)

            #get all prior imgs
            train_path = os.path.join(name_path, 'train')
            priors_paths = os.listdir(train_path)
            img_paths = []
            for img_path in priors_paths:
                img = os.path.join(train_path,img_path)
                img_paths.append(img)

            prompts = get_prompts(specie_folder, prompt_type=args.prompt_type)
            gens=[]
            sample_paths = []
            for prompt in prompts:
                prior_path = random.sample(img_paths,1)[0]
                sample_paths.append(prior_path)
                prior_img = load_image(prior_path).convert('RGB').resize((512,512))
                gen = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    ip_adapter_image=prior_img,
                    num_inference_steps=50,
                    guidance_scale=5.0,
                ).images
                gens += gen
            #save 
            save_path = os.path.join(name_path, 'generated_images')
            os.makedirs(save_path, exist_ok=True)
            save_paths = []
            for index,gen_img in enumerate(gens):
                save_path_img = os.path.join(save_path, f'{index}.png')
                save_paths.append(save_path_img)
                gen_img.save(save_path_img)
            rows = []
            for img_save_path,prior_path in zip(save_paths,sample_paths):
                rows.append({'name': name, 'file_path':img_save_path, 'species':specie_folder, 'ref_path':prior_path})
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    df.to_csv(os.path.join(root_path, 'generated_data.csv'), index=False)

if __name__=="__main__":
    main()