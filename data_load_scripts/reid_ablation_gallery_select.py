from argparse import ArgumentParser
import os
import pandas as pd
import shutil
import albumentations
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from transformers import AutoModel
from tqdm.auto import tqdm
from torch.cuda.amp import autocast 
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import random

"""script that creates new gallery from a loaded data folder, then creates either kmeans or randomly sampled gallery"""

def get_test_transforms(image_size = (440,440)):
    return albumentations.Compose(
        [
            albumentations.Resize(image_size[0], image_size[1], always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )

class EvalDataset(Dataset):
    def __init__(self,df):
        self.df = df
        self.augmentations = get_test_transforms()
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = row['file_path']
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = self.augmentations(image=image)['image']
        return {"image":image}
    

def extract_miewID_embeds(data_loader,device='cuda'):
    model_tag = f"conservationxlabs/miewid-msv2"
    model = AutoModel.from_pretrained(model_tag, trust_remote_code=True)
    model.to(device)
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    embeddings = []

    
    with torch.no_grad():
        for batch in tk0:
            with autocast():
                batch_embeddings = model.extract_feat(batch["image"].to(device))
            
            batch_embeddings = batch_embeddings.detach().cpu().numpy()
            
            batch_embeddings_df = pd.DataFrame(batch_embeddings)
            embeddings.append(batch_embeddings_df)
            
    embeddings = pd.concat(embeddings)
    embeddings = embeddings.values

    assert not np.isnan(embeddings).sum(), "NaNs found in extracted embeddings"

    return embeddings

def kmeans_kchoice(df, K=5):

    dataset = EvalDataset(df)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        num_workers=8, 
        shuffle=False, 
        pin_memory=True, 
        drop_last=False
    )
    embeds = extract_miewID_embeds(data_loader=dataloader)
    file_paths = df['file_path'].tolist()

    kmeans_model = KMeans(
        n_clusters=K, 
        random_state=42, 
        n_init='auto', 
        max_iter=300
    )
    
    cluster_labels = kmeans_model.fit_predict(embeds)
    centroids = kmeans_model.cluster_centers_
    
    results_df = pd.DataFrame({
        'embed_index': range(len(embeds)),
        'file_path': file_paths,
        'cluster': cluster_labels
    })

    representative_paths = []

    for k_id in range(K):
        cluster_data = results_df[results_df['cluster'] == k_id]
        
        if cluster_data.empty:
            representative_paths[k_id] = None
            continue
            
        current_cluster_embeds = embeds[cluster_data['embed_index'].values]
        
        centroid = centroids[k_id].reshape(1, -1)
        distances = euclidean_distances(current_cluster_embeds, centroid)
        
        min_distance_index = np.argmin(distances)
        representative_path = cluster_data.iloc[min_distance_index]['file_path']
        
        representative_paths.append(representative_path)
    return representative_paths



def sample(df, K=5):
    file_paths = df['file_path'].tolist()
    return random.sample(file_paths,K)
    


def get_args()-> ArgumentParser:
    parser = ArgumentParser(description="Load and format data")
    parser.add_argument('--root_folder',type=str,required=True)
    parser.add_argument('--output_folder',type=str,required=True)
    parser.add_argument("--random",action='store_true',default=False,help="if not random, then k-means")
    parser.add_argument("--K", type=int, default=10)
    return parser.parse_args()

def main():
    args =get_args()
    random.seed(101)
    root = args.root_folder
    output_folder = args.output_folder
    os.makedirs(output_folder,exist_ok=True)
    tpath = os.path.join(root,'train_data.csv')
    gpath = os.path.join(root,'gallery_data.csv')
    train_df = pd.read_csv(tpath)
    names = train_df['name'].unique().tolist()
    df_gallery = pd.read_csv(gpath)
    total_paths = []
    for name in names:
            df_name = df_gallery[df_gallery['name'] == name]
            k = min(len(df_name),args.K)
            if args.random:
                gallery_paths = sample(df_name,K=k)
            else:
                gallery_paths = kmeans_kchoice(df_name,K=k)
            total_paths += gallery_paths

    df_save = df_gallery[df_gallery['file_path'].isin(total_paths)]
    save_path = os.path.join(output_folder,'gallery_data.csv')
    df_save.to_csv(save_path)

        

if __name__ =='__main__':
    main()

