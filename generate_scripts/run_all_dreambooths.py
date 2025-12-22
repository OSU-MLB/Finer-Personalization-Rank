from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import os
import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from diffusers import DiffusionPipeline
import torch
import pandas as pd
import gc


from utils import get_prompts, get_class_prompt, get_instance_prompt

"""script that trains/and runs all dreambooth images, given loaded data root

    Needs to be run from root code folder
"""
def getargs(): 
    """
    Expeting root in format from notebook such that:
    root/data/specie/name/train/#.png

    will train and run model for each subject for every specie.

    """

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help='root folder that holds data and csv files')
    parser.add_argument("--token", required=False, type=str, default="sks")
    parser.add_argument("--prior_loss_weight" ,type=float, default=.5)
    parser.add_argument("--with_prior_pres", action='store_true', default=False)
    parser.add_argument("--just_gen", action='store_true',default=False)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--skip_already_run", action='store_true', default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt_type", type=str, default="reid")
    return parser.parse_args()


def train_model(instance_prompt, instance_data_dir, model_path,class_data_path,class_prompt, args):
    if(args.with_prior_pres):
        os.system(
             f"accelerate launch generate_scripts/train_dreambooth_sd3.py "
             f"--instance_data_dir {instance_data_dir} "
             f"--output_dir={model_path} "
             f"--instance_prompt=\"{instance_prompt}\" "
             f"--gradient_accumulation_steps=1 "
             f"--train_batch_size=1 "
             f"--resolution=768 "
             f"--use_8bit_adam "
             f"--with_prior_preservation "
             #f"--train_text_encoder "
             f"--max_train_steps={args.steps} "
             f"--lr_warmup_steps=0 "
             f"--lr_scheduler=\"constant\" "
             f"--learning_rate=5e-6 "
             f"--mixed_precision=\"bf16\" "
             f"--pretrained_model_name_or_path=\"stabilityai/stable-diffusion-3-medium-diffusers\" "
             f"--class_prompt=\"{class_prompt}\" "
             f"--class_data_dir=\"{class_data_path}\" "
             f"--prior_loss_weight={args.prior_loss_weight} "
             f"--prior_generation_precision \"bf16\" "
             f"--checkpoints_total_limit=0 "
             f"--checkpointing_steps=5000 "
            # f"--dataloader_num_workers=4 "
        )
    else:
        os.system(
             f"accelerate launch generate_scripts/train_dreambooth_sd3.py "
             f"--instance_data_dir {instance_data_dir} "
             f"--output_dir={model_path} "
             f"--instance_prompt=\"{instance_prompt}\" "
             f"--gradient_accumulation_steps=1 "
             f"--train_batch_size=1 "
             f"--resolution=768 "
             f"--use_8bit_adam "
             #f"--train_text_encoder "
             f"--max_train_steps={args.steps} "
             f"--lr_warmup_steps=0 "
             f"--lr_scheduler=\"constant\" "
             f"--learning_rate=5e-6 "
             f"--mixed_precision=\"bf16\" "
             f"--pretrained_model_name_or_path=\"stabilityai/stable-diffusion-3-medium-diffusers\" "
             f"--class_prompt=\"{instance_prompt}\" "
             f"--class_data_dir=\"{class_data_path}\" "
             f"--checkpoints_total_limit=0 "
             f"--checkpointing_steps=5000 "
            # f"--dataloader_num_workers=4 "
        )

def generate_images_subject(prompts, subject_root_path, model_path, device):
    #load model
    pipeline = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16
    )
    #pipeline.load_lora_weights(model_path)

    pipeline.to(device)

    #generate and save images
    save_path = os.path.join(subject_root_path, 'generated_images')
    os.makedirs(save_path, exist_ok=True)

    images  = pipeline(prompts, num_inference_steps=30, guidance_scale=5.0).images
    save_paths = []
    for index,img in enumerate(images):
        save_path_img = os.path.join(save_path, f'{index}.png')
        save_paths.append(save_path_img)
        img.save(save_path_img)
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return save_paths
def main():
    args = getargs()
    root_path = args.root
    data_path = os.path.join(root_path, "data")
    df = pd.DataFrame(columns=['name', 'file_path', 'viewpoint', 'species'])

    species_folders = os.listdir(data_path)

    for specie_folder in species_folders:
        print(f'starting species : {specie_folder}')

        species_path = os.path.join(data_path, specie_folder)
        name_folders = os.listdir(species_path)
        class_data_path = os.path.join(species_path, 'class_prior_imgs')
        for name in name_folders:
            if name == 'class_prior_imgs':
                continue
            print(f'running {name}')

            name_path =  os.path.join(species_path,name)
            model_path = os.path.join(name_path, 'model')

            #continue if crashed, save from re running alr run subjects
            if( args.skip_already_run and os.path.exists(os.path.join(model_path,'model_index.json'))):
                gen_path  = name_path + '/generated_images'
                print(f'skipping {name}')

                if(os.path.exists((gen_path +'/9.png'))):
                    rows = []
                    for img_save_path in os.listdir(gen_path):
                        rows.append({'name': name, 'file_path':gen_path + '/' + img_save_path, 'species':specie_folder})
                        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
                    continue
                else:
                    print("generating imgs")
                    prompts = get_prompts(specie_folder,prompt_type=args.prompt_type, token=args.token )
                    save_paths = generate_images_subject(prompts=prompts, subject_root_path=name_path,model_path=model_path, device=args.device)
                    rows = []
                    for img_save_path in save_paths:
                        rows.append({'name': name, 'file_path':img_save_path, 'species':specie_folder})
                    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


                continue

            os.makedirs(model_path,exist_ok=True)
            train_path = os.path.join(name_path, 'train')
            if(not args.just_gen):
                #train
                train_model(
                    instance_prompt=get_instance_prompt(args.token, specie_folder),
                    instance_data_dir=train_path,
                    model_path=model_path,
                    class_data_path=class_data_path,
                    class_prompt=get_class_prompt(specie_folder),
                    args=args
                            )
            #run and save img
            prompts = get_prompts(img_class=specie_folder,prompt_type=args.prompt_type, token=args.token)
            save_paths = generate_images_subject(prompts=prompts, subject_root_path=name_path,model_path=model_path, device=args.device)
            rows = []
            for img_save_path in save_paths:
                rows.append({'name': name, 'file_path':img_save_path, 'species':specie_folder})
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    df.to_csv(os.path.join(root_path, 'generated_data.csv'), index=False)

if __name__=="__main__":
    main()