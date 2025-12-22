from tqdm.auto import tqdm
import torch
from transformers import AutoModel
from torch.cuda.amp import autocast 
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from cars_classifier import EfficientNetEmbedder

PATH_TO_CARS_CLASSIFIER = "path"

def avg_cos_sim(distmat:np.ndarray,q_pids:np.ndarray,g_pids:np.ndarray):
    simmat = 1.0-distmat
    num_q = simmat.shape[0]
    total_cos_sim = 0
    for i in range(num_q):
        q_pid = q_pids[i]
        is_relevant = (g_pids == q_pid)
        num_relevant = is_relevant.sum()
        sims = simmat[i,:] * is_relevant
        total_cos_sim += sims.sum() / num_relevant
    avg_cos_sim = total_cos_sim / num_q
    return avg_cos_sim

def mean_average_precision(distmat: np.ndarray, q_pids: np.ndarray, g_pids: np.ndarray) -> float:
    
    num_q = distmat.shape[0]
    aps = []

    for i in range(num_q):
        q_pid = q_pids[i]
        is_relevant = (g_pids == q_pid)
        indices = np.argsort(distmat[i])
        valid_indices = indices[distmat[i, indices] < np.inf]
        ranked_is_relevant = is_relevant[valid_indices]
        num_gt = np.sum(ranked_is_relevant)

        if num_gt == 0:
            # If no ground truth matches exist for this query, AP is 0.
            aps.append(0.)
            continue

        tp_cumsum = np.cumsum(ranked_is_relevant).astype(float)
        k = np.arange(1, len(ranked_is_relevant) + 1)
        precision_at_k = tp_cumsum / k
        ap = np.sum(precision_at_k * ranked_is_relevant) / num_gt
        
        aps.append(ap)
    if not aps:
        return 0.0
        
    mAP = np.mean(aps)
    return mAP


def extract_cars_embeds(data_loader, device):
    model = EfficientNetEmbedder()
    model.load_state_dict(torch.load(PATH_TO_CARS_CLASSIFIER))

    model.eval()

    model = model.to(device)
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Extracting Cars Embeds")
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in tk0:
            inputs = batch['image'].to(device)
            batch_labels = batch['label'] 
        
            _, batch_embeddings = model(inputs) 
            
            embeddings.append(batch_embeddings.cpu())
            labels.append(batch_labels)
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    
    return embeddings, labels

def cosine_distance(input1, input2):
    """Computes cosine distance. adapted from MiewID code

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat

def extract_bioclip2_embeds(data_loader, device):
    import open_clip

    model, _,_ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')

    model.to(device)
    model.eval()

    tk0 = tqdm(data_loader, total=len(data_loader), desc="Extracting BioCLIP2 Embeds")
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in tk0:
            with autocast():
                images = batch["image"].to(device)
                image_features = model(images)[0]

            image_features = image_features.detach().cpu().numpy()
            image_idx = batch["image_idx"].tolist()
            batch_embeddings_df = pd.DataFrame(image_features, index=image_idx)
            embeddings.append(batch_embeddings_df)

            batch_labels = batch['label'].tolist()
            labels.extend(batch_labels)

    embeddings = pd.concat(embeddings)
    embeddings = embeddings.values

    assert not np.isnan(embeddings).sum(), "NaNs found in extracted embeddings"

    return embeddings, labels



def extract_dino_embeds(data_loader, device):

    model_tag = "facebook/dinov2-base"
    model = AutoModel.from_pretrained(model_tag)
        
    model.to(device)
    model.eval()

    tk0 = tqdm(data_loader, total=len(data_loader), desc="Extracting DINO Embeds")
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in tk0:
            with autocast():
                pixel_values = batch["image"].to(device)
                
                outputs = model(pixel_values=pixel_values)
                last_hidden_state = outputs.last_hidden_state
                
                batch_embeddings = last_hidden_state[:, 0, :] 
            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            
            image_idx = batch["image_idx"].tolist()
            batch_embeddings_df = pd.DataFrame(batch_embeddings, index=image_idx)
            embeddings.append(batch_embeddings_df)

            batch_labels = batch['label'].tolist()
            labels.extend(batch_labels)
            
    embeddings = pd.concat(embeddings)
    embeddings = embeddings.values

    assert not np.isnan(embeddings).sum(), "NaNs found in extracted embeddings"

    return embeddings, labels

def extract_clipI_embeds(data_loader, device):
    """Extracts CLIP image embeddings.
    """
    model_tag = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_tag)
    model.to(device)
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader), desc="Extracting CLIP Embeds")
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in tk0:
            with autocast():
                pixel_values = batch["image"].to(device)
                batch_embeddings = model.get_image_features(pixel_values=pixel_values)
            
            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            
            image_idx = batch["image_idx"].tolist()
            batch_embeddings_df = pd.DataFrame(batch_embeddings, index=image_idx)
            embeddings.append(batch_embeddings_df)

            batch_labels = batch['label'].tolist()
            labels.extend(batch_labels)
            
    embeddings = pd.concat(embeddings)
    embeddings = embeddings.values

    assert not np.isnan(embeddings).sum(), "NaNs found in extracted embeddings"

    return embeddings, labels


def extract_miewID_embeds(data_loader, device):
    """extract miewID embeddings
    """

    model_tag = f"conservationxlabs/miewid-msv2"
    model = AutoModel.from_pretrained(model_tag, trust_remote_code=True)
    model.to(device)
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in tk0:
            with autocast():
                batch_embeddings = model.extract_feat(batch["image"].to(device))
            
            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            
            image_idx = batch["image_idx"].tolist()
            batch_embeddings_df = pd.DataFrame(batch_embeddings, index=image_idx)
            embeddings.append(batch_embeddings_df)

            batch_labels = batch['label'].tolist()
            labels.extend(batch_labels)
            
    embeddings = pd.concat(embeddings)
    embeddings = embeddings.values

    assert not np.isnan(embeddings).sum(), "NaNs found in extracted embeddings"

    return embeddings, labels