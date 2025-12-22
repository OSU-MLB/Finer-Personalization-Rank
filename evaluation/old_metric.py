import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel,CLIPModel, CLIPProcessor,AutoImageProcessor
from tqdm.auto import tqdm
from torch.cuda.amp import autocast 
import numpy as np
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics.pairwise import cosine_similarity
from cars_classifier import EfficientNetEmbedder
import open_clip
from cars_classifier import get_cars_transform
from carvekit.api.high import HiInterface


"""script for getting traditional gen-reference similarity metrics such as CLIP, DINO, miewID, CarClassifier, BioCLIP2"""

# Need to run cars_classifier.py first to train model
PATH_TO_CARS_CLASSIFIER = 'path/to/cars/classifier'

def get_test_transforms(image_size):
    return albumentations.Compose(
        [
            albumentations.Resize(image_size[0], image_size[1], always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )
def extract_cars_embeds(data_loader,device):
    model = EfficientNetEmbedder()
    model.load_state_dict(torch.load('PATH_TO_CARS_CLASSIFIER'))
    model.eval()
    model = model.to(device)
    embeddings_gen = []
    embeddings_ref = []

    tk0 = tqdm(data_loader, total=len(data_loader), desc="Extracting Cars Embeds")
    with torch.no_grad():
        for batch in tk0:
            gen_img = batch['gen_img'].to(device)
            ref_img = batch['ref_img'].to(device)
        
            _, batch_embeddings_gen = model(gen_img) 
            _, batch_embeddings_ref = model(ref_img) 
            
            batch_embeddings_gen = batch_embeddings_gen.detach().cpu().numpy()
            batch_embeddings_ref = batch_embeddings_ref.detach().cpu().numpy()
            
            batch_embeddings_gen_df = pd.DataFrame(batch_embeddings_gen)
            batch_embeddings_ref_df = pd.DataFrame(batch_embeddings_ref)
            embeddings_gen.append(batch_embeddings_gen_df)
            embeddings_ref.append(batch_embeddings_ref_df)
            
    embeddings_gen = pd.concat(embeddings_gen)
    embeddings_ref = pd.concat(embeddings_ref)

    embeddings_gen = embeddings_gen.values
    embeddings_ref = embeddings_ref.values

    assert not np.isnan(embeddings_gen).sum(), "NaNs found in extracted DINO embeddings"
    assert not np.isnan(embeddings_ref).sum(), "NaNs found in extracted DINO embeddings"

    return embeddings_gen, embeddings_ref


def extract_miewID_embeds(data_loader,device):
    model_tag = f"conservationxlabs/miewid-msv2"
    model = AutoModel.from_pretrained(model_tag, trust_remote_code=True)
    model.to(device)
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))

    embeddings_gen = []
    embeddings_ref = []
    with torch.no_grad():
        for batch in tk0:
            with autocast():
                pixel_values_gen = batch["gen_img"].to(device)
                pixel_values_ref = batch["ref_img"].to(device)
                
                
                batch_embeddings_gen = model.extract_feat(pixel_values_gen)

                batch_embeddings_ref = model.extract_feat(pixel_values_ref)
            
            batch_embeddings_gen = batch_embeddings_gen.detach().cpu().numpy()
            batch_embeddings_ref = batch_embeddings_ref.detach().cpu().numpy()
            
            batch_embeddings_gen_df = pd.DataFrame(batch_embeddings_gen)
            batch_embeddings_ref_df = pd.DataFrame(batch_embeddings_ref)
            embeddings_gen.append(batch_embeddings_gen_df)
            embeddings_ref.append(batch_embeddings_ref_df)
            
    embeddings_gen = pd.concat(embeddings_gen)
    embeddings_ref = pd.concat(embeddings_ref)

    embeddings_gen = embeddings_gen.values
    embeddings_ref = embeddings_ref.values

    assert not np.isnan(embeddings_gen).sum(), "NaNs found in extracted DINO embeddings"
    assert not np.isnan(embeddings_ref).sum(), "NaNs found in extracted DINO embeddings"

    return embeddings_gen, embeddings_ref
    

def extract_bioclip2_embeds(data_loader,device='cuda'):
    import open_clip
    model, _,_ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    model.to(device)
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Extracting BioCLIP2 Embeds")
    embeddings_gen = []
    embeddings_ref = []

    with torch.no_grad():
        for batch in tk0:
            with autocast():
                images_gen = batch["gen_img"].to(device)
                images_ref = batch["ref_img"].to(device)
                image_features_gen = model(images_gen)[0]
                image_features_ref = model(images_ref)[0]

            image_features_gen = image_features_gen.detach().cpu().numpy()
            image_features_ref = image_features_ref.detach().cpu().numpy()
            batch_embeddings_gen_df = pd.DataFrame(image_features_gen)
            batch_embeddings_ref_df = pd.DataFrame(image_features_ref)

            embeddings_gen.append(batch_embeddings_gen_df)
            embeddings_ref.append(batch_embeddings_ref_df)
            
    embeddings_gen = pd.concat(embeddings_gen)
    embeddings_ref = pd.concat(embeddings_ref)

    embeddings_gen = embeddings_gen.values
    embeddings_ref = embeddings_ref.values

    assert not np.isnan(embeddings_gen).sum(), "NaNs found in extracted DINO embeddings"
    assert not np.isnan(embeddings_ref).sum(), "NaNs found in extracted DINO embeddings"

    return embeddings_gen, embeddings_ref

def extract_dino_embeds(data_loader,device='cuda'):
    model_tag = "facebook/dinov2-base"
    model = AutoModel.from_pretrained(model_tag)
    model.to(device)
    model.eval()

    tk0 = tqdm(data_loader, total=len(data_loader), desc="Extracting DINO Embeds")
    embeddings_gen = []
    embeddings_ref = []

    with torch.no_grad():
        for batch in tk0:
            with autocast():
                pixel_values_gen = batch["gen_img"].to(device)
                pixel_values_ref = batch["ref_img"].to(device)
                
                
                outputs_gen = model(pixel_values=pixel_values_gen)
                last_hidden_state_gen = outputs_gen.last_hidden_state

                outputs_ref = model(pixel_values=pixel_values_ref)
                last_hidden_state_ref = outputs_ref.last_hidden_state

                
                batch_embeddings_gen = last_hidden_state_gen[:, 0, :] 
                batch_embeddings_ref = last_hidden_state_ref[:, 0, :] 
            
            batch_embeddings_gen = batch_embeddings_gen.detach().cpu().numpy()
            batch_embeddings_ref = batch_embeddings_ref.detach().cpu().numpy()
            
            batch_embeddings_gen_df = pd.DataFrame(batch_embeddings_gen)
            batch_embeddings_ref_df = pd.DataFrame(batch_embeddings_ref)
            embeddings_gen.append(batch_embeddings_gen_df)
            embeddings_ref.append(batch_embeddings_ref_df)
            
    embeddings_gen = pd.concat(embeddings_gen)
    embeddings_ref = pd.concat(embeddings_ref)

    embeddings_gen = embeddings_gen.values
    embeddings_ref = embeddings_ref.values

    assert not np.isnan(embeddings_gen).sum(), "NaNs found in extracted DINO embeddings"
    assert not np.isnan(embeddings_ref).sum(), "NaNs found in extracted DINO embeddings"

    return embeddings_gen, embeddings_ref

def extract_clip_embeds(data_loader,device='cuda'):
    model_tag = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_tag)
    model.to(device)
    model.eval()

    tk0 = tqdm(data_loader, total=len(data_loader), desc="Extracting CLIP Embeds")
    embeddings_gen = []
    embeddings_ref = []

    with torch.no_grad():
        for batch in tk0:
            with autocast():
                pixel_values_gen = batch["gen_img"].to(device)
                pixel_values_ref = batch["ref_img"].to(device)
                
                batch_embeddings_gen = model.get_image_features(pixel_values=pixel_values_gen)
                batch_embeddings_ref = model.get_image_features(pixel_values=pixel_values_ref)
                
            batch_embeddings_gen = batch_embeddings_gen.detach().cpu().numpy()
            batch_embeddings_ref = batch_embeddings_ref.detach().cpu().numpy()
            
            batch_embeddings_gen_df = pd.DataFrame(batch_embeddings_gen)
            batch_embeddings_ref_df = pd.DataFrame(batch_embeddings_ref)
            embeddings_gen.append(batch_embeddings_gen_df)
            embeddings_ref.append(batch_embeddings_ref_df)
            
    embeddings_gen = pd.concat(embeddings_gen)
    embeddings_ref = pd.concat(embeddings_ref)

    embeddings_gen = embeddings_gen.values
    embeddings_ref = embeddings_ref.values

    assert not np.isnan(embeddings_gen).sum(), "NaNs found in extracted CLIP embeddings"
    assert not np.isnan(embeddings_ref).sum(), "NaNs found in extracted CLIP embeddings"

    return embeddings_gen, embeddings_ref

def get_sim(metric, dataloader,device='cuda'):
    if metric == 'dino':
        embeddings_gen,embeddings_ref = extract_dino_embeds(data_loader=dataloader,device=device)
    elif metric =='clip':
        embeddings_gen,embeddings_ref = extract_clip_embeds(data_loader=dataloader,device=device)
    elif metric =='bioclip':
         embeddings_gen,embeddings_ref = extract_bioclip2_embeds(data_loader=dataloader,device=device)
    elif metric =='miewID':
         embeddings_gen,embeddings_ref = extract_miewID_embeds(data_loader=dataloader,device=device)
    elif metric =='car':
         embeddings_gen,embeddings_ref = extract_cars_embeds(data_loader=dataloader,device=device)
    else:
        raise ValueError(f"Unsupported metric.")
    
    assert len(embeddings_gen) == len(embeddings_ref), "Embeddings arrays must have the same length"
    
    embeddings_gen_t = torch.from_numpy(embeddings_gen)
    embeddings_ref_t = torch.from_numpy(embeddings_ref)
    
    similarity_scores = torch.nn.functional.cosine_similarity(embeddings_gen_t, embeddings_ref_t, dim=1)
    
    return similarity_scores.cpu().numpy().mean()


class EvalDataset(Dataset):
    def __init__(self,csv, transforms,proccessor=None,bioclip_proccessor=None,car_trans=None, hi_interface=None):
        self.csv = csv
        self.proccessor = proccessor
        self.bioclip_proccessor = bioclip_proccessor
        self.augmentations = transforms
        self.car_trans = car_trans
        self.hi_interface = hi_interface
    
    def __len__(self):
        return len(self.csv)
        
    def __getitem__(self, index):
        row = self.csv.iloc[index]
        gen_path = row['file_path']
        ref_path = row['ref_path']
        gen_img = Image.open(gen_path).convert('RGB')
        ref_img = Image.open(ref_path).convert('RGB')

        if self.hi_interface:
            imgs = self.hi_interface([gen_img,ref_img])
            gen_img = imgs[0].convert("RGB")
            ref_img = imgs[0].convert("RGB")

        if self.proccessor:
            processed_input_gen = self.proccessor(images=gen_img, return_tensors="pt")
            processed_input_ref = self.proccessor(images=ref_img, return_tensors="pt")
            
            gen_img = processed_input_gen['pixel_values'].squeeze(0)
            ref_img = processed_input_ref['pixel_values'].squeeze(0)
        elif self.bioclip_proccessor:
            gen_img = self.bioclip_proccessor(gen_img)
            ref_img = self.bioclip_proccessor(ref_img)
        elif self.car_trans:
            gen_img = self.car_trans(gen_img)
            ref_img = self.car_trans(ref_img)
        else:
            gen_img = np.array(gen_img)
            gen_img = self.augmentations(image=gen_img)['image']
            ref_img = np.array(ref_img)
            ref_img = self.augmentations(image=ref_img)['image']
        return {"gen_img":gen_img,
                "ref_img":ref_img
                }

def evaluate_sim(gen_csv_path, metric, species, device='cuda', image_size=(440, 440), num_workers=12, batch_size=64, remove_background=False):
    df = pd.read_csv(gen_csv_path)
    if not species is None:
        df = df[df['species'] ==species]
    
    if metric == 'dino':
        processor_tag = "facebook/dinov2-base"
        processor = AutoImageProcessor.from_pretrained(processor_tag)
    elif metric == 'clip':
        processor_tag = "openai/clip-vit-base-patch32"
        processor = AutoImageProcessor.from_pretrained(processor_tag) 
    else:
        processor = None
    
    if metric =='bioclip':
        _,_,bioclip_proccessor = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    else:
        bioclip_proccessor = None

    if metric == 'car':
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

    eval_dataset = EvalDataset(csv=df, transforms=get_test_transforms(image_size=image_size),proccessor=processor, bioclip_proccessor=bioclip_proccessor,car_trans=car_trans, hi_interface=interface)
    
    
    data_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if interface else num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    similarity = get_sim(metric=metric, dataloader=data_loader, device=device)
    
    return similarity