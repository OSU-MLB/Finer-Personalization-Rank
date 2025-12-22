import pandas as pd
import os
import sys
from argparse import ArgumentParser
import json
from Evaluate import evaluate
from dreambench_plus_eval import get_gpt_concept_score

"""script to get baseline results"""

def getargs()-> ArgumentParser: 
    args = ArgumentParser()
    args.add_argument("--config", type=str,required=True)
    args.add_argument("--save_path", type=str, required=True)
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--gpt_save",type=str)
    args.add_argument("--device", default='cuda')
    args.add_argument("--remove_background",action='store_true',default=False)
    args.add_argument("--skip_gpt",action='store_true',default=False)

    return args.parse_args()
def main():
    print("getting baseline results")

    args = getargs()
    with open(args.config,"r") as f:
        config = json.load(f)

    metric = config['metric']
    rows = []
    device = args.device
    if not args.skip_gpt:
        gpt_save = args.gpt_save
        os.makedirs(gpt_save,exist_ok=True)
    src = config['root_path']
    gpath = os.path.join(src,'gallery_data.csv')
    qpath = os.path.join(src,'train_data.csv')
    classess = pd.read_csv(qpath)['species'].unique()
    for clss in classess:
        
        metric_mAP,metric_sim = evaluate(qpath,gpath,model_name=metric,batch_size=args.batch_size,species=clss, device=device, remove_background=args.remove_background)
        clipI_mAP,clipI_sim = evaluate(qpath,gpath,model_name='clip',batch_size=args.batch_size,species=clss, device=device, remove_background=args.remove_background)
        dino_mAP,dino_sim = evaluate(qpath,gpath,model_name='dino',batch_size=args.batch_size,species=clss, device=device, remove_background=args.remove_background)
        

        if not args.skip_gpt:
            gpt_model_save = os.path.join(gpt_save,'baseline.csv')
            gpt = get_gpt_concept_score(query_csv=qpath,gallery_csv=gpath,species=clss,scores_save_csv_path=gpt_model_save)

            rows.append({
                'class':clss,
                'gpt_concept':gpt,
                f'{metric}_mAP':metric_mAP,
                f'{metric}_sim':metric_sim,
                'clipI_mAP':clipI_mAP,
                'clipI_sim':clipI_sim,
                'dino_mAP':dino_mAP,
                'dino_sim':dino_sim

            })
        else:
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
    save_path = args.save_path
    df.to_csv(save_path)




if __name__ == '__main__':
    main()