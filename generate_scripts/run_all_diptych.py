import torch
import torch.nn.functional as F
import pandas as pd
from diffusers.utils import load_image, check_min_version
import os
import numpy as np
from PIL import Image
import argparse
import random

from diffusers.models.attention_processor import Attention

from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import cv2
from transformers import AutoProcessor, pipeline, AutoModelForMaskGeneration
from utils import get_prompts
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DiptychPrompting.controlnet_flux import FluxControlNetModel
from DiptychPrompting.diptych_prompting_inference import grounded_segmentation, CustomFluxAttnProcessor2_0
from DiptychPrompting.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

"""script to generate diptych images"""

NEGATIVE_PROMPT = "low quality, cropped, deformed, text, ugly, blurry, zoomed in, not showing whole subject, cartoony, unrealistic, lowres, not showing whole subject, animated"
def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help='root folder that holds data and csv files')
    parser.add_argument("--prompt_type", type=str, default="reid")
    parser.add_argument('--attn_enforce', type=float, default=1.3)
    parser.add_argument('--ctrl_scale', type=float, default=0.95)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--pixel_offset', type=int, default=8)
    parser.add_argument('--device', type=str,default='cuda')
    parser.add_argument("--skip",default=False, action='store_true') # dont use skip.
    return parser.parse_args()


def generate_images(save_folder, pipe,conditions,size, subject_name,target_prompts, generator, ctrl_scale, height,width,args):
    base_prompt = f"a photo of {subject_name}"
    save_paths = []
    ref_paths = []
    for index,target_prompt in enumerate(target_prompts):
        diptych_text_prompt = f"A diptych with two side-by-side images of same {subject_name}. On the left, {base_prompt}. On the right, replicate this {subject_name} exactly but as {target_prompt}"
        condition = random.sample(conditions,1)
        mask_image,diptych_image_prompt, path = condition[0]
        # Inpaint
        if not args.skip:
            result = pipe(
                prompt=diptych_text_prompt,
                height=size[1],
                width=size[0],
                control_image=diptych_image_prompt,
                control_mask=mask_image,
                num_inference_steps=30,
                generator=generator,
                controlnet_conditioning_scale=ctrl_scale,
                guidance_scale=3.5,
                negative_prompt=NEGATIVE_PROMPT,
                true_guidance_scale=3.5
            ).images[0]

            result = result.crop((width, 0, width*2, height))
            result = result.crop((args.pixel_offset, args.pixel_offset, width-args.pixel_offset, height-args.pixel_offset))
        save_path = os.path.join(save_folder,f'{index}.png')
        save_paths.append(save_path)
        if not args.skip:
            result.save(save_path)
        ref_paths.append(path)
    return save_paths,ref_paths



def main():
    args = getargs()
    device = args.device

    root_path = args.root
    data_path = os.path.join(root_path, "data")
    random.seed(101)

    #adapted from diptych prompting code

    # Build pipeline
    controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16
    ).to(device)
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    base_attn_procs = pipe.transformer.attn_processors.copy()

    width = args.width + args.pixel_offset * 2
    height = args.height + args.pixel_offset * 2
    size = (width*2, height)

    new_attn_procs = base_attn_procs.copy()
    for i, (k, v) in enumerate(new_attn_procs.items()):
        new_attn_procs[k] = CustomFluxAttnProcessor2_0(height=height // 16, width=width // 16 * 2, attn_enforce=args.attn_enforce)
    pipe.transformer.set_attn_processor(new_attn_procs)

    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    segment_processor = AutoProcessor.from_pretrained(segmenter_id)
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=torch.device(device))

    def segment_image(image, object_name):
        image_array, detections = grounded_segmentation(
            device,
            object_detector,
            segmentator,
            segment_processor,
            image=image,
            labels=object_name,
            threshold=0.3,
            polygon_refinement=True,
        )
        segment_result = image_array * np.expand_dims(detections[0].mask / 255, axis=-1) + np.ones_like(image_array) * (
                1 - np.expand_dims(detections[0].mask / 255, axis=-1)) * 255
        segmented_image = Image.fromarray(segment_result.astype(np.uint8))
        return segmented_image


    def make_diptych(image):
        ref_image = np.array(image)
        ref_image = np.concatenate([ref_image, np.zeros_like(ref_image)], axis=1)
        ref_image = Image.fromarray(ref_image)
        return ref_image
    

    

    generator = torch.Generator(device=device).manual_seed(42)
    ctrl_scale=args.ctrl_scale

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
            conditions = []
            for prior_path in priors_paths:
                prior_img_path = os.path.join(train_path,prior_path)
                reference_image = load_image(prior_img_path).resize((width, height)).convert("RGB")

                prompts = get_prompts(specie_folder, prompt_type=args.prompt_type)
                
                subject_name = specie_folder
                segmented_image = segment_image(reference_image, subject_name)
                mask_image = np.concatenate([np.zeros((height, width, 3)), np.ones((height, width, 3))*255], axis=1)
                mask_image = Image.fromarray(mask_image.astype(np.uint8))
                diptych_image_prompt = make_diptych(segmented_image)
                conditions.append((mask_image,diptych_image_prompt, prior_img_path))

            #gen time
            save_folder = os.path.join(name_path,'generated_images')
            os.makedirs(save_folder,exist_ok=True)
            save_paths,sample_paths = generate_images(save_folder=save_folder, pipe=pipe,conditions=conditions,size=size,subject_name=subject_name,target_prompts=prompts,generator=generator, ctrl_scale=ctrl_scale,height=height,width=width,args=args)

            rows = []
            for img_save_path,pth in zip(save_paths,sample_paths):
                rows.append({'name': name, 'file_path':img_save_path, 'species':specie_folder,'ref_path':pth})
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    df.to_csv(os.path.join(root_path, 'generated_data.csv'), index=False)

if __name__ == '__main__':
    main()



    

