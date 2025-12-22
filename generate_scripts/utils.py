"""Holds prompt format fn"""

def get_prompts(img_class, prompt_type, token=None):
    """prompts used for evaluation, based on dataset type
    """
    prompts = []
    
    if(prompt_type=='reid'): #animal reid
        if(token is None):
            prompts = [ f'a {img_class} in a grassy field',
                        f'a photo of a {img_class} on the beach', 
                        f'a {img_class} on a cobblestone street' ,
                        f'a photo of a {img_class} with a mountain in the backgorund' ,
                        f'a {img_class} laying down on the beach',
                        f'a {img_class} in a puddle',
                        f'a {img_class} in the forest' ,
                        f'a photo of a {img_class} sitting down with a mountain in the background' ,
                        f'a {img_class} running with tall grass in the background', 
                        f'a {img_class} walking on a dirt road' ]
        else:
            prompts = [ f'a {token} {img_class} in a grassy field',
                        f'a photo of a {token} {img_class} on the beach', 
                        f'a {token} {img_class} on a cobblestone street' ,
                        f'a photo of a {token} {img_class} with a mountain in the backgorund' ,
                        f'a {token} {img_class} laying down on the beach',
                        f'a {token} {img_class} in a puddle',
                        f'a {token} {img_class} in the forest' ,
                        f'a photo of a {token} {img_class} sitting down with a mountain in the background' ,
                        f'a {token} {img_class} running with tall grass in the background', 
                        f'a {token} {img_class} walking on a dirt road' ]
            
    elif(prompt_type=='cub'): 
        if(token is None):
            prompts = [ f'a photo of a {img_class} on a tree branch',
                        f'a {img_class} on the beach', 
                        f'a photo of a {img_class} on a fence' ,
                        f'a {img_class} with a mountain in the backgorund' ,
                        f'a photo of a {img_class} perched on a rock inside a forest',
                        f'a {img_class} in a puddle',
                        f'a photo of a {img_class} flying in the sky' ,
                        f'a {img_class} perched with a mountain in the background' ,
                        f'a photo of a {img_class} perched with tall grass in the background', 
                        f'a {img_class} flying over water' ]
        else:
            prompts = [ f'a photo of a {token} {img_class} on a tree branch',
                        f'a {token} {img_class} on the beach', 
                        f'a photo of a {token} {img_class} on a fence' ,
                        f'a {token} {img_class} with a mountain in the backgorund' ,
                        f'a photo of a {token} {img_class} perched on a rock inside a forest',
                        f'a {token} {img_class} in a puddle',
                        f'a photo of a {token} {img_class} flying in the sky' ,
                        f'a {token} {img_class} perched with a mountain in the background' ,
                        f'a photo of a {token} {img_class} perched with tall grass in the background', 
                        f'a {token} {img_class} flying over water' ]
            
    elif(prompt_type =='car'):
        if(token is None):
            prompts = [f'a photo of a {img_class} on a dirt road',
                       f'a {img_class} in front of a house',
                       f'a photo of a {img_class} driving down a dirt road with mountains in the distance',
                       f'a {img_class} driving down a mountain road while it is raining',
                       f'a photo of a {img_class} driving on a snowy road in the mountains',
                       f'a {img_class} in front of a mountain',
                       f'a photo of a {img_class} parked on a cliff overlooking the ocean',
                       f'a {img_class} driving fast on a racetrack',
                       f'a photo of a {img_class} parked on the beach in front of the sea',
                       f'a {img_class} parked inside a shipping crate'
                       ]
        else:
            prompts = [f'a photo of a {token} {img_class} on a dirt road sided by cornfields',
                       f'a {token} {img_class} in front of a house',
                       f'a photo of a {token} {img_class} driving down a dirt road with mountains in the distance',
                       f'a {token} {img_class} driving down a mountain road while it is raining',
                       f'a photo of a {token} {img_class} driving on a snowy road in the mountains',
                       f'a {token} {img_class} in front of a mountain',
                       f'a photo of a {token} {img_class} parked on a cliff overlooking the ocean',
                       f'a {token} {img_class} driving fast on a racetrack',
                       f'a photo of a {token} {img_class} parked on the beach in front of the sea',
                       f'a {token} {img_class} parked inside a shipping crate'
                       ]
    return prompts


#dreambooth prompts
def get_class_prompt(img_class):
    return f'a photo of a {img_class}'
def get_instance_prompt(token, img_class):
    return f'a photo of a {token} {img_class}'

