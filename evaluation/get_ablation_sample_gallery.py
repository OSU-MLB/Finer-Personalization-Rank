import pandas as pd
import os
import sys
from argparse import ArgumentParser
import json
from Evaluate import evaluate
from dreambench_plus_eval import get_gpt_concept_score


"""script to eval ablation gallery, kmeans or sample"""
def getargs()-> ArgumentParser: 
    args = ArgumentParser()
    args.add_argument("--config", type=str,required=True)
    args.add_argument("--save_path", type=str, required=True)
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument("--device", default='cuda')
    return args.parse_args()


def results(gpath, metric,save_path, args, sources):
    rows = []
    device = args.device
    for src_name,src in sources:
        qpath = os.path.join(src, 'generated_data.csv')
        classess = pd.read_csv(qpath)['species'].unique()
        for clss in classess:
                metric_mAP,metric_sim = evaluate(qpath,gpath,model_name=metric,batch_size=args.batch_size,species=clss, device=device)
                clipI_mAP,clipI_sim = evaluate(qpath,gpath,model_name='clip',batch_size=args.batch_size,species=clss, device=device)
                dino_mAP,dino_sim = evaluate(qpath,gpath,model_name='dino',batch_size=args.batch_size,species=clss, device=device)

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


def baseline_results(gpath,qpath,save_path, metric, args):
    rows = []
    device = args.device
    classess = pd.read_csv(qpath)['species'].unique()
    for clss in classess:
            metric_mAP,metric_sim = evaluate(qpath,gpath,model_name=metric,batch_size=args.batch_size,species=clss, device=device)
            clipI_mAP,clipI_sim = evaluate(qpath,gpath,model_name='clip',batch_size=args.batch_size,species=clss, device=device)
            dino_mAP,dino_sim = evaluate(qpath,gpath,model_name='dino',batch_size=args.batch_size,species=clss, device=device)

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

    tpath = config['tpath']

    sources = config['sources']

    save_path = args.save_path
    os.makedirs(save_path,exist_ok=True)

    gpaths = config['gpaths']
    for name,gpath in gpaths:
        baseline_path = os.path.join(save_path,f'{name}_baseline')
        gen_path = os.path.join(save_path,f'{name}')
        baseline_results(gpath,tpath,save_path=baseline_path,metric=metric,args=args)

        results(gpath,metric=metric,save_path=gen_path,args=args,sources=sources)


    
    print("DONE")

if __name__ == '__main__':
     main()



    


