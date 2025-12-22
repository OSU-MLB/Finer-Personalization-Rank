from argparse import ArgumentParser
import pandas as pd
from utils import get_prompts, get_class_prompt, get_instance_prompt
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from diffusers import DiffusionPipeline
import sys
import random
"""
Script that generates all EMU2 images given root of data
This is  set up for single GPU, multi-gpu generation needs to be added
"""

def getargs():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help='root folder that holds data and csv files')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--max_num_prior_imgs", type=int,default=1)
    parser.add_argument("--prompt_type", type=str, default="reid")
    parser.add_argument("--skip_already_run", action='store_true' , default=False)
    return parser.parse_args()


def main():
    args = getargs()
    root_path = args.root
    data_path = os.path.join(root_path, "data")
    random.seed(101)
    #os.makedirs(root_path, exist_ok=True)

    #emu2 pipeline stuff
    path = snapshot_download(
    "BAAI/Emu2-Gen",
    cache_dir="model_cache",
    repo_type="model"
    )

    multimodal_encoder = AutoModelForCausalLM.from_pretrained(
        f"{path}/multimodal_encoder",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="bf16"
    )

    tokenizer = AutoTokenizer.from_pretrained(f"{path}/tokenizer")

    pipe = DiffusionPipeline.from_pretrained(
        path,
        custom_pipeline="pipeline_emu2_gen",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="bf16",
        multimodal_encoder=multimodal_encoder,
        tokenizer=tokenizer,
    )
    # For the non-first time of using, you can init the pipeline directly
    pipe = DiffusionPipeline.from_pretrained(
        path,
        custom_pipeline="pipeline_emu2_gen",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="bf16",
    )

    pipe.to(args.device)


    #save csv file in form for miewID
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

            

            #gen imgs

            prompts = get_prompts(specie_folder, prompt_type=args.prompt_type)
            save_path = os.path.join(name_path, 'generated_images')
            if(args.skip_already_run and os.path.exists(os.path.join(save_path,'9.png'))):
                rows = []
                for img_save_path in os.listdir(save_path):
                    rows.append({'name': name, 'file_path':os.path.join('data',specie_folder,name,'generated_images',f'{index}.png'), 'species':specie_folder})
                    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
                continue
            os.makedirs(save_path, exist_ok=True)
            save_paths = []
            sample_paths = []
            for index,prompt in enumerate(prompts):
                path = random.sample(paths, 1)[0]
                sample_paths.append(path)
                pimg = Image.open(path).convert('RGB').resize((512,512))
                emu_prompt = pimg + [prompt]
                gen_img = pipe(emu_prompt)
                save_path_img = os.path.join(save_path, f'{index}.png')
                save_paths.append(os.path.join('data',specie_folder,name,'generated_images',f'{index}.png'))
                gen_img.image.save(save_path_img)

            rows = []
            for img_save_path,pth in zip(save_paths,sample_paths):
                rows.append({'name': name, 'file_path':img_save_path, 'species':specie_folder, 'ref_path':pth})
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df['file_path'] = df['file_path'].apply(lambda x: os.path.join(root_path,x))
    df.to_csv(os.path.join(root_path, 'generated_data.csv'), index=False)

if __name__=="__main__":
    main()