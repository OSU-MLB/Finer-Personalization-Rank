import os
import torch
import random
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn.functional as F
import open_clip
from transformers import CLIPModel, CLIPProcessor,AutoImageProcessor
from cars_classifier import get_cars_transform
from embeds_extractors import extract_miewID_embeds, cosine_distance, mean_average_precision,avg_cos_sim, extract_clipI_embeds, extract_dino_embeds, extract_bioclip2_embeds, extract_cars_embeds
import random
from carvekit.api.high import HiInterface

"""fns for clipt and evaluation in retrieval style with mAP and cosine sim between query and gallery"""

def get_test_transforms(image_size):
    return albumentations.Compose(
        [
            albumentations.Resize(image_size[0], image_size[1], always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )

def convert_name_to_id(names, le):
    names_id = le.transform(names)
    return names_id

def preproccess_df(csv_path, specie, le, all_names):
    df = pd.read_csv(csv_path)
    if not specie is None:
        df = df[df['species'] == specie]
    if le.classes_.size == 0:
        le.fit(all_names)
        
    names = df['name'].values
    names_id = convert_name_to_id(names,le)
    df['name'] = names_id
    df = df.reset_index(drop=True)
    return df

class EvalDataset(Dataset):
    def __init__(self, csv, transforms, proccessor, bioclip_proccessor=None, car_transforms=None, hi_interface=None):
        self.csv = csv
        self.augmentations = transforms
        self.proccessor = proccessor
        self.bioclip_proccessor =bioclip_proccessor
        self.car_transforms = car_transforms
        self.hi_interface = hi_interface
        
    def __len__(self):
        return len(self.csv)
    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image_path = row['file_path']
        image = Image.open(image_path).convert('RGB')
        if self.hi_interface:
            image = self.hi_interface([image])[0]
            image = image.convert('RGB')


        #depending on model proccess image differently
        if not self.car_transforms is None:
            image = self.car_transforms(image)
        elif not(self.bioclip_proccessor is None):
            image = self.bioclip_proccessor(image)
        elif(self.proccessor is None):
            image = np.array(image)
            image = self.augmentations(image=image)['image']
        else:
            image = np.array(image)
            processed_input = self.proccessor(images=image, return_tensors="pt")
            image = processed_input['pixel_values'].squeeze(0)

        return {"image":image, "label":torch.tensor(row['name']),
                "image_idx": self.csv.index[index]}

def eval_fn(query_loader, gallery_loader, model_name, device):
    #extract query and gallery embeds
    if(model_name == 'clip'):
        embeds_query, labels_query = extract_clipI_embeds(query_loader, device)
        embeds_gallery, labels_gallery = extract_clipI_embeds(gallery_loader, device)
    elif(model_name == 'dino'):
        embeds_query, labels_query = extract_dino_embeds(query_loader, device)
        embeds_gallery, labels_gallery = extract_dino_embeds(gallery_loader, device)
    elif (model_name == 'miewID'):
        embeds_query, labels_query = extract_miewID_embeds(query_loader, device)
        embeds_gallery, labels_gallery = extract_miewID_embeds(gallery_loader, device)
    elif(model_name == 'bioclip'):
        embeds_query, labels_query = extract_bioclip2_embeds(query_loader, device)
        embeds_gallery, labels_gallery = extract_bioclip2_embeds(gallery_loader, device)
    elif(model_name =='car'):
        embeds_query, labels_query = extract_cars_embeds(query_loader, device)
        embeds_gallery, labels_gallery = extract_cars_embeds(gallery_loader, device)
    else:
        raise Exception("invalid model name")
    
        
    #calc cosine dist
    q_pids = np.array(labels_query)
    g_pids = np.array(labels_gallery)
    
    qf = torch.Tensor(embeds_query)
    gf = torch.Tensor(embeds_gallery)
    

    distmat = cosine_distance(qf, gf)

    distmat = distmat.numpy()
    #calc mAP
    mAP = mean_average_precision(distmat,q_pids, g_pids)
    #calc avg cosine sim between same query and gallery instances
    cos_sim = avg_cos_sim(distmat,q_pids,g_pids)

    return mAP, cos_sim



def evaluate(query_csv, gallery_csv,model_name, batch_size, device='cuda', image_size = (440,440), species=None, num_workers=0, ablation_num_subj=None,ablation_num_per_subj=None, remove_background=False):
    """
    Evaluate retrieval performance between query and gallery datasets. 
    Args:
        query_csv (str): Path to the CSV file containing the query dataset.
        gallery_csv (str): Path to the CSV file containing the gallery dataset.
        model_name (str): Name of the model to use for feature extraction. Options: 'clip', 'dino', 'miewID', 'bioclip', 'car'.
        batch_size (int): Batch size for data loading.
        device (str): Device to run the evaluation on. Default is 'cuda'.
        image_size (tuple): Size to which images will be resized. Default is (440, 440).
        species (str, optional): Specific species to filter the datasets. Default is None.
        num_workers (int): Number of workers for data loading. Default is 0.
        ablation_num_subj (int, optional): Number of subjects to sample for ablation. Default is None.
        ablation_num_per_subj (int, optional): Number of images per subject to sample for ablation. Default is None.
        remove_background (bool): Whether to remove background from images using HiInterface. Default is False.
    Returns:
        tuple: Mean Average Precision (mAP) and average cosine similarity between query and gallery embeddings
    """
    df_query_all = pd.read_csv(query_csv)
    df_gallery_all = pd.read_csv(gallery_csv)

    if not species is None:
        df_query_all = df_query_all[df_query_all['species'] == species]
        df_gallery_all = df_gallery_all[df_gallery_all['species'] == species]

    all_names = np.unique(np.concatenate((df_query_all['name'].values, df_gallery_all['name'].values)))
    
    le = LabelEncoder()
    le.fit(all_names)
    
    df_query = preproccess_df(query_csv, specie=species, le=le, all_names=all_names)
    df_gallery = preproccess_df(gallery_csv, specie=species, le=le, all_names=all_names)

    
    #apply ablations if specified
    if not (ablation_num_subj is None):
        gallery_non_query = df_gallery[~df_gallery['name'].isin(df_query['name'].unique())]['name'].unique().tolist()
        if ablation_num_subj > 10:
            sampled_names = random.sample(gallery_non_query,min(ablation_num_subj-10,len(gallery_non_query)) )
            tnames = df_query['name'].unique().tolist() + sampled_names
        else:
            tnames = df_query['name'].unique()
        df_gallery = df_gallery[df_gallery['name'].isin(tnames)]

    if (not ablation_num_per_subj is None):
        df_gallery = df_gallery.groupby('name', group_keys=False).apply(lambda x: x.sample(n=min(len(x), ablation_num_per_subj), random_state=101))
    


    #set up proccessor/model for dataset
    if model_name =='clip':
        model_tag = "openai/clip-vit-base-patch32"
        proccessor = CLIPProcessor.from_pretrained(model_tag)
    elif model_name =='dino':
        dino_model_tag = "facebook/dinov2-base" 
        proccessor = AutoImageProcessor.from_pretrained(dino_model_tag)
    else:
        proccessor = None
    
    if model_name =='bioclip':
        _,_,bioclip_proccessor = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    else:
        bioclip_proccessor = None

    if model_name == 'car':
        car_trans = get_cars_transform()
    else:
        car_trans = None

    if remove_background:
        interface = HiInterface(
            object_type="object",
            batch_size_seg=1,
            batch_size_matting=1,
            device=str('cuda:1'),
            seg_mask_size=640,
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=False
        )
    else:
        interface = None

    #get embeds
    query_dataset = EvalDataset(df_query, get_test_transforms(image_size), proccessor=proccessor, bioclip_proccessor=bioclip_proccessor, car_transforms=car_trans, hi_interface=interface)
    gallery_dataset = EvalDataset(df_gallery, get_test_transforms(image_size), proccessor=proccessor, bioclip_proccessor=bioclip_proccessor, car_transforms=car_trans, hi_interface=interface)
    query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, drop_last=False)
    gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, drop_last=False)

    
    mAP,cos_sim = eval_fn(query_loader=query_loader, gallery_loader=gallery_loader, model_name=model_name, device=device)
    return mAP, cos_sim

def get_clip_t(prompts, img_folder_path, clip_model, proccessor, device='cuda'):

    sim_sum = 0
    for index,prompt in enumerate(prompts):
        img = Image.open(os.path.join(img_folder_path, f'{index}.png'))
        inputs = proccessor(text=[prompt], images=img, return_tensors="pt", padding=True)
        inputs.to(device)
        with torch.no_grad():
            img_embeds = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            text_embeds = clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        img_embeds = F.normalize(img_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        similarity = torch.cosine_similarity(img_embeds, text_embeds).item()
        sim_sum +=similarity
    t_score = sim_sum / len(prompts)
    return t_score

def get_average_clip_t(prompts, subjects_root, device='cuda'):

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    subjects = os.listdir(subjects_root)
    t_sum = 0
    num_subjs = 0
    for subject in subjects:
        if(subject == 'class_prior_imgs'): 
            continue
        num_subjs +=1
        subject_gen_path = os.path.join(subjects_root,subject,'generated_images')
        t_score = get_clip_t(prompts=prompts, img_folder_path=subject_gen_path, clip_model=model,proccessor=processor, device=device)
        t_sum += t_score
    clip_t = t_sum/num_subjs
    return clip_t
