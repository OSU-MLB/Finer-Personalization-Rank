from argparse import ArgumentParser
import pandas as pd
from utils import get_prompts, get_class_prompt, get_instance_prompt
import os
import torch
from diffusers.pipelines import FluxPipeline
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
import sys
from huggingface_hub import snapshot_download
import random


"""generates all ip-adapter-plus images given data folder"""

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
TARGET_PATH ="replace/../..."
IP_ADAPTER_MODEL_ID = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT_NAME = "ip-adapter-plus_sdxl_vit-h.bin"
IMAGE_ENCODER_SUBFOLDER = "models/image_encoder"

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
    #os.makedirs(root_path, exist_ok=True)
    random.seed(101)
    ip_adapter_local_folder = os.path.join(TARGET_PATH, "ip_adapter_weights")
    os.makedirs(ip_adapter_local_folder, exist_ok=True)
    snapshot_download(
    repo_id=IP_ADAPTER_MODEL_ID,
    local_dir=ip_adapter_local_folder,
    local_dir_use_symlinks=False,
    allow_patterns=[
        f"sdxl_models/{IP_ADAPTER_WEIGHT_NAME}",  
        f"{IMAGE_ENCODER_SUBFOLDER}/*",         
        f"{IMAGE_ENCODER_SUBFOLDER}/config.json", 
    ],
    ignore_patterns=[".gitattributes"])

    pipe = StableDiffusionXLPipeline.from_pretrained(
        TARGET_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(args.device)

    pipe.load_ip_adapter(
        ip_adapter_local_folder, 
        subfolder="sdxl_models", 
        weight_name=IP_ADAPTER_WEIGHT_NAME, 
        image_encoder_folder=IMAGE_ENCODER_SUBFOLDER,
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
            paths = []
            priors_paths = os.listdir(train_path)
            for img_path in priors_paths:
                paths.append(os.path.join(train_path,img_path))

            sample_paths = []
            gens = []
            prompts = get_prompts(specie_folder, prompt_type=args.prompt_type)
            for idx in range(len(prompts)):
                prompts_batch = prompts[idx]
                path = random.sample(paths, 1)[0]
                sample_paths.append(path)
                img = Image.open(path).convert('RGB').resize((512,512))
                gen = pipe(
                    prompt=prompts_batch,
                    negative_prompt=NEGATIVE_PROMPT,
                    ip_adapter_image=img,
                    num_inference_steps=50,
                    guidance_scale=5.0,
                ).images
                gens += gen
            #save 
            save_path = os.path.join(name_path, 'generated_images')
            os.makedirs(save_path, exist_ok=True)
            save_paths = []
            if args.skip:
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
            for img_save_path,pth in zip(save_paths,sample_paths):
                rows.append({'name': name, 'file_path':img_save_path, 'species':specie_folder,'ref_path':pth})
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    df.to_csv(os.path.join(root_path, 'generated_data.csv'), index=False)

if __name__=="__main__":
    main()