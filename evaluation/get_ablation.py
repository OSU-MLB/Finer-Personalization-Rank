import pandas as pd
import os
import sys
from argparse import ArgumentParser
import json
from Evaluate import evaluate
from dreambench_plus_eval import get_gpt_concept_score
"""script to ablate gallery, num subjs and imgs"""
def getargs()-> ArgumentParser: 
    args = ArgumentParser()
    args.add_argument("--config", type=str,required=True)
    args.add_argument("--save_path", type=str, required=True)
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument("--device", default='cuda')
    return args.parse_args()


def results(gpath, metric,save_path, args, sources, gallery_subjs, per_subj):
    rows = []
    device = args.device
    for src_name,src in sources:
        qpath = os.path.join(src, 'generated_data.csv')
        classess = pd.read_csv(qpath)['species'].unique()
        for clss in classess:
                metric_mAP,metric_sim = evaluate(qpath,gpath,model_name=metric,batch_size=args.batch_size,species=clss, device=device, ablation_num_subj=gallery_subjs, ablation_num_per_subj=per_subj)
                clipI_mAP,clipI_sim = evaluate(qpath,gpath,model_name='clip',batch_size=args.batch_size,species=clss, device=device,ablation_num_subj=gallery_subjs, ablation_num_per_subj=per_subj)
                dino_mAP,dino_sim = evaluate(qpath,gpath,model_name='dino',batch_size=args.batch_size,species=clss, device=device,ablation_num_subj=gallery_subjs, ablation_num_per_subj=per_subj)

                rows.append({
                    'model':src_name,
                    'class':clss,
                    f'{metric}_mAP':metric_mAP,
                    f'{metric}_sim':metric_sim,
                    'clipI_mAP':clipI_mAP,
                    'clipI_sim':clipI_sim,
                    'dino_mAP':dino_mAP,
                    'dino_sim':dino_sim

                })

    df = pd.DataFrame(rows)
    df.to_csv(save_path)


def baseline_results(gpath,qpath,save_path, metric, args, gallery_subjs, per_subj):
    rows = []
    device = args.device
    classess = pd.read_csv(qpath)['species'].unique()
    for clss in classess:
            metric_mAP,metric_sim = evaluate(qpath,gpath,model_name=metric,batch_size=args.batch_size,species=clss, device=device,ablation_num_subj=gallery_subjs, ablation_num_per_subj=per_subj)
            clipI_mAP,clipI_sim = evaluate(qpath,gpath,model_name='clip',batch_size=args.batch_size,species=clss, device=device,ablation_num_subj=gallery_subjs, ablation_num_per_subj=per_subj)
            dino_mAP,dino_sim = evaluate(qpath,gpath,model_name='dino',batch_size=args.batch_size,species=clss, device=device,ablation_num_subj=gallery_subjs, ablation_num_per_subj=per_subj)

            rows.append({
                'class':clss,
                f'{metric}_mAP':metric_mAP,
                f'{metric}_sim':metric_sim,
                'clipI_mAP':clipI_mAP,
                'clipI_sim':clipI_sim,
                'dino_mAP':dino_mAP,
                'dino_sim':dino_sim

            })
    df = pd.DataFrame(rows)
    df.to_csv(save_path)


def main():
    print("getting ablation baseline")

    args = getargs()
    with open(args.config,"r") as f:
        config = json.load(f)
    
    metric = config['metric']
    ablation_settings = config['ablation_settings']

    gpath = config['gpath']
    tpath = config['tpath']

    sources = config['sources']

    save_path = args.save_path
    os.makedirs(save_path,exist_ok=True)

    for gallery_subjs, gallery_per_subjs in ablation_settings:
        ablation_path = os.path.join(save_path, f'{gallery_subjs}_{gallery_per_subjs}')
        os.makedirs(ablation_path,exist_ok=True)
        ablation_baseline = os.path.join(ablation_path,'baseline.csv')
        if(os.path.exists(ablation_baseline)):
            print(f'skipping {gallery_subjs}_{gallery_per_subjs} baseline')
        else:
            baseline_results(gpath=gpath,qpath=tpath, save_path=ablation_baseline, metric=metric, args=args, gallery_subjs=gallery_subjs, per_subj=gallery_per_subjs)
        
        generated_baseline = os.path.join(ablation_path,'generated.csv')
        if(os.path.exists(generated_baseline)):
            print(f'skipping {gallery_subjs}_{gallery_per_subjs} generated')
        else:
            results(gpath=gpath, metric=metric,save_path=generated_baseline,args=args,sources=sources, gallery_subjs=gallery_subjs, per_subj=gallery_per_subjs)

    print("DONE")

if __name__ == '__main__':
     main()



    


