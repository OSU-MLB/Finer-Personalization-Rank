import pandas as pd
import os
import sys
from argparse import ArgumentParser
import json
from old_metric import evaluate_sim
from Evaluate import evaluate, get_average_clip_t
from dreambench_plus_eval import get_gpt_concept_score
current_dir = os.path.dirname(__file__) 
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)
from generate_scripts.utils import get_prompts

"""script to run main eval"""

def getargs()-> ArgumentParser: 
    args = ArgumentParser()
    args.add_argument("--config", type=str,required=True)
    args.add_argument("--save_path", type=str, required=True)
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--gpt_save",type=str,required=False)
    args.add_argument("--device", default='cuda')
    args.add_argument("--remove_background",action='store_true',default=False)
    args.add_argument("--skip_gpt",action='store_true',default=False)
    args.add_argument("--species", nargs="+", type=str,default=False)
    return args.parse_args()
def main():
    print("getting results")

    args = getargs()
    with open(args.config,"r") as f:
        config = json.load(f)

    metric = config['metric']
    sources = config['sources']
    prompt_type = config['prompt_type']
    rows = []
    device = args.device
    if not args.skip_gpt:
        gpt_save = args.gpt_save
        os.makedirs(gpt_save,exist_ok=True)

    for src_name,src in sources:
        gpath = os.path.join(src,'gallery_data.csv')
        qpath = os.path.join(src, 'generated_data.csv')
        if args.species:
            classess = args.species
        else:
            classess = pd.read_csv(qpath)['species'].unique()
        
        for clss in classess:
            
            clss_root = os.path.join(src,'data',clss)
            clipt = get_average_clip_t(prompts=get_prompts(img_class=clss, prompt_type=prompt_type), subjects_root=clss_root,device=device)

            metric_mAP,_ = evaluate(qpath,gpath,model_name=metric,batch_size=args.batch_size,species=clss, device=device, remove_background=args.remove_background)
            clipI_mAP,_ = evaluate(qpath,gpath,model_name='clip',batch_size=args.batch_size,species=clss, device=device, remove_background=args.remove_background)
            dino_mAP,_ = evaluate(qpath,gpath,model_name='dino',batch_size=args.batch_size,species=clss, device=device, remove_background=args.remove_background)
            if src_name =='dreambooth':
                tpath = os.path.join(src,'train_data.csv')
                _,metric_sim = evaluate(qpath,tpath,model_name=metric,batch_size=args.batch_size,species=clss, device=device, remove_background=args.remove_background)
                _,clipI_sim = evaluate(qpath,tpath,model_name='clip',batch_size=args.batch_size,species=clss, device=device, remove_background=args.remove_background)
                _,dino_sim = evaluate(qpath,tpath,model_name='dino',batch_size=args.batch_size,species=clss, device=device, remove_background=args.remove_background)
                if not args.skip_gpt:
                    gpt_model_save = os.path.join(gpt_save,f'{src_name}.csv')
                    gpt = get_gpt_concept_score(query_csv=qpath,gallery_csv=tpath,species=clss,scores_save_csv_path=gpt_model_save)

                    rows.append({
                        'model':src_name,
                        'class':clss,
                        'clip_t':clipt,
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
                        'model':src_name,
                        'class':clss,
                        'clip_t':clipt,
                        f'{metric}_mAP':metric_mAP,
                        f'{metric}_sim':metric_sim,
                        'clipI_mAP':clipI_mAP,
                        'clipI_sim':clipI_sim,
                        'dino_mAP':dino_mAP,
                        'dino_sim':dino_sim

                    })
            else:
                metric_sim = evaluate_sim(qpath,metric=metric,batch_size=args.batch_size,species=clss, device=device,remove_background=args.remove_background)
                clipI_sim = evaluate_sim(qpath,metric='clip',batch_size=args.batch_size,species=clss, device=device,remove_background=args.remove_background)
                dino_sim = evaluate_sim(qpath,metric='dino',batch_size=args.batch_size,species=clss, device=device,remove_background=args.remove_background)
                if not args.skip_gpt:
                    gpt_model_save = os.path.join(gpt_save,f'{src_name}.csv')
                    gpt = get_gpt_concept_score(query_csv=qpath,gallery_csv=None,species=clss,scores_save_csv_path=gpt_model_save)

                    rows.append({
                        'model':src_name,
                        'class':clss,
                        'clip_t':clipt,
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
                            'model':src_name,
                            'class':clss,
                            'clip_t':clipt,
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