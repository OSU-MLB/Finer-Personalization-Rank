
import pandas as pd
import openai
import os
import time
import numpy
import re
import io
from PIL import Image
import base64
import random
"""based off Dreambench++ code"""
OPENAI_KEY = "your key"
PATTERN = r"(score|Score):\s*[a-zA-Z]*\s*(\d+)"

def encode_image(image_path):
    #similar to dreambench++ paper
    img = Image.open(image_path).resize((512,512))
    buffered = io.BytesIO()
    img.save(buffered,format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def call_gpt(model,messages, **kwargs):
    TRIES = 5
    for i in range(TRIES):
        client = openai.OpenAI(api_key=OPENAI_KEY)
        time.sleep(abs(numpy.random.normal(loc=0.12, scale=0.04)) + 0.1)
        completion = client.chat.completions.create(
            messages=messages, 
            model=model,
            seed=random.randint(0,100),
            **kwargs
        )
        
        content = completion.choices[0].message.content
        score = re.findall(PATTERN, content)
        score = [int(s) for _, s in score]
    
        if len(score) == 1:
            #print(score[0])
            return score[0]
        print("Trying gpt call again")
    
    assert i != TRIES
    

def format_gpt_message(user_prompt,gpt_prompt, src_img_path, target_img_path ):
    src_b64_img = encode_image(src_img_path)
    target_b64_img = encode_image(target_img_path)
    messages = [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": gpt_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{src_b64_img}", "detail": "high"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{target_b64_img}", "detail": "high"}},
                            ],
                        },
                    ]
    return messages




def get_gpt_concept_score(query_csv, gallery_csv, species, scores_save_csv_path, gpt_prompt_file_path='dreambench_plus/gpt_prompt_subject_full.txt',user_prompt_file_path='dreambench_plus/user_prompt_subject_full.txt', openai_model ='gpt-4o-2024-05-13', max_gallery_imgs_used=1):
    with open(gpt_prompt_file_path, "r") as f:
        gpt_prompt = f.read().strip()
    with open(user_prompt_file_path, "r") as f:
        user_prompt = f.read().strip()
    
    df_query = pd.read_csv(query_csv)
    if gallery_csv:
        df_gallery = pd.read_csv(gallery_csv)
        df_gallery = df_gallery[df_gallery['species'] == species]

    random.seed(101)
    
    df_query = df_query[df_query['species'] == species]
    


    if os.path.exists(scores_save_csv_path):
        print("resuming gpt calc")
        score_df = pd.read_csv(scores_save_csv_path)
        done_subjects = score_df['name'].unique()
        for ds in done_subjects:
            print(f"skipping {ds}")
        df_query = df_query[~df_query['name'].isin(done_subjects)]
        if gallery_csv:
            df_gallery = df_gallery[~df_gallery['name'].isin(done_subjects)]
    else:
        print("starting gpt calc")
        score_df = pd.DataFrame(columns=['species','name','gpt_score'])



    subjects = df_query['name'].unique()
    for subject in subjects:
        print(f'getting GPT score for {subject}')
        df_query_subject = df_query[df_query['name'] == subject]
        if gallery_csv:
            df_gallery_subject = df_gallery[df_gallery['name'] == subject]

        query_images_paths = df_query_subject['file_path'].tolist()
        if gallery_csv:
            gallery_images_paths = df_gallery_subject['file_path'].tolist()

            if len(gallery_images_paths) > max_gallery_imgs_used:
                gallery_images_paths = random.sample(gallery_images_paths, max_gallery_imgs_used) 

            total_subject_score = 0
            for gallery_image_path in gallery_images_paths:
                for query_image_path in query_images_paths:

                    messages = format_gpt_message(user_prompt=user_prompt, gpt_prompt=gpt_prompt,src_img_path=gallery_image_path, target_img_path=query_image_path )
                    score = call_gpt(openai_model, messages, temperature=1)
                    total_subject_score += score*.25

            subject_score = total_subject_score/(len(gallery_images_paths) * len(query_images_paths))
        else:
            total_subject_score = 0
            ref_img_paths = df_query_subject['ref_path'].tolist()
            for ref_path,gen_path in zip(ref_img_paths,query_images_paths):
                messages = format_gpt_message(user_prompt=user_prompt, gpt_prompt=gpt_prompt,src_img_path=ref_path, target_img_path=gen_path )
                score = call_gpt(openai_model, messages, temperature=1)
                total_subject_score += score*.25
            subject_score = total_subject_score / len(query_images_paths)
            
        score_df = pd.concat([score_df,pd.DataFrame([{'species':species, 'name':subject, 'gpt_score': subject_score}])],ignore_index=True)
        score_df.to_csv(scores_save_csv_path)

    species_score_df = score_df[score_df['species'] ==species]
    scores = species_score_df['gpt_score'].tolist()
    final_score = sum(scores) / len(scores)
    print(f'final score {final_score}')
    return final_score

        

        
    

    

