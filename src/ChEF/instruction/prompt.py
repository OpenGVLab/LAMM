# classification
coarse_grained_classification_prompts = [
    "The photo of the ",
    "The image shows",
    "What is the category label for the object in this image?",
    "What is in the image?",
    "Can you identify the object in this image?",
    "Based on the image's features, what could be the potential category label for this image?",
    "What label would you assign to this image based on the object's shape and size?",
    "According to the model's prediction, what is the label assigned to this image?",
    "Can you provide the category label for this image based on the object's color and texture?",
    "What label do you think best describes the image's content?",
    "Based on the image's context, what category label would you assign to it?",
    "Can you suggest any alternate labels for this image based on its content and features?",
    "What is the most suitable category label for this image based on its shape, size, and context?",
    "According to the model's classification, what is the category label assigned to this object?",
    "Based on the image's visual cues, what category label do you think is the most appropriate?",
    "Can you provide any additional labels that could be applied to this image based on its context and features?",
    "What label would you assign to this image based on the object's function or purpose?",
    "According to the image's features and context, what label do you think best represents it?",
    "Can you suggest any potential alternate category labels that might be appropriate for this image based on its attributes?",
    "According to the image's attributes, what label would you assign to it?",
]

fine_grained_classification_prompts = [
    "What is the fine-grained category label for the object in this image?",
]

fine_grained_classification_multiturn_prompts = [
    ["What is the fine-grained category label for the object in this image?", "As the coarse-grained category label for this image is {prefix}, what is the fine-grained category label for this image?"]
]

# LAMM-style classfication prompts
classification_lamm_prompts = ['{question}']

# caption
caption_prompts = [
    "Generate caption of this image:",
    "What is described in the image?",
    "A photo of", 
    "What is the caption of this image?",
    "The caption of the image is",
    "Write a caption for this picture.",
    "What is depicted in this photograph?",
    "A snapshot of what?",
    "Can you create a caption for this image?",
    "The image is captioned as:",
    "Describe what you see in this picture.",
    "Provide a brief explanation of this image.",
    "Write a short description of what's happening in the picture.",
    "What story does this image tell?",
    "The caption for this photograph is:",
    "What would you title this image?",
    "What does this picture represent?",
    "Summarize the image in one sentence.",
    "What is the message conveyed by this image?",
    "Compose a suitable caption for this photo.",
    "An image of", # For kosmos2
]

# LAMM-style standard prompts
caption_lamm_prompts = ['{question}']

# vqa
vqa_prompts = [
    '{question}The answer is',
    '{question}What is the correct option for this question?',
    '{question}',
    '{question}What is the answer?',
    '{question}The answer for the question is',
    '{question}ANSWER:',
    '{question}The answer (option) is',
    '{question}Answer:',
]

# LAMM-style standard prompts
vqa_lamm_prompts = ['{question}']

# Octavius
Classification_octavius3d_prompts = ['{question}']
VQA_octavius3d_prompts = ['{question}']
Caption_octavius3d_prompts = ['{question}']

# multiimage
winoground_prompts = ['{question}']

# counting, question defined in dataset
# example: 'How many {} are there in this image?'
counting_prompts = ['{question}']

# detection
# We provide both LAMM-style standard prompts and multi-turn prompts
detection_lamm_prompts = [
    'Identify all the objects in the image and provide their positions. Your answer needs to give the object name and the bounding box of the object. The bounding box should be represented as [x1, y1, x2, y2] with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.',
    'Detect all the objects in the image.',
]

# Two-turn detection prompts, with the first turn prompt for the category and the second turn prompt for the bounding box.
detection_multiturn_prompts = [
    ['The image shows',
     'Give all the bounding boxes of {prefix} in the image. The bounding box should be represented as [x1, y1, x2, y2] with floating numbers indicating the coordinates of the object in a normalized range of 0-1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.'
    ],
    ['Detect all the objects in the image.', 'Provide coordinates [x0,y0,x1,y1] for {prefix} in the image.'], # For shikra
    ['Detect all the objects in the image.', '<grounding><phrase>{prefix}</phrase>'] # For kosmos2
]

# POPE
# The prompts for POPE are defined in dataset. 
pope_prompts = ['{question}']


    
singleturn_prompt_dict = {
    'coarse_grained_classification_prompts': coarse_grained_classification_prompts,
    'fine_grained_classification_prompts': fine_grained_classification_prompts,
    'caption_prompts': caption_prompts,
    'VQA_prompts': vqa_prompts,
    'counting_prompts': counting_prompts,
    'POPE_prompts':pope_prompts,
    'detection_prompts': detection_lamm_prompts,
    'detection_lamm_prompts': detection_lamm_prompts,
    'VQA_lamm_prompts' : vqa_lamm_prompts,
    'caption_lamm_prompts' : caption_lamm_prompts,
    'Facial_cls_lamm_prompts' : classification_lamm_prompts,
    'classification_lamm_prompts': classification_lamm_prompts,
    'Classification_octavius3d_prompts': Classification_octavius3d_prompts,
    'VQA_octavius3d_prompts': VQA_octavius3d_prompts,
    'Caption_octavius3d_prompts': Caption_octavius3d_prompts,
    'Winoground_prompts': winoground_prompts,
}

multiturn_prompt_dict = {
    'fine_grained_classification_multiturn_prompts': fine_grained_classification_multiturn_prompts,
    'detection_multiturn_prompts': detection_multiturn_prompts,
}

def singleturn_prompt(
        task_name,
        assigned_ids = 0,
        defined_prompt = None,
        **kwargs
        ):
    '''
        return prompt: str
    '''
    print('Using singleturn prompt...')
    if defined_prompt is not None:
        assert isinstance(defined_prompt, str), f'The defined prompt must be string. '
        print(f'Using user defined prompt: {defined_prompt} for task: {task_name}')
        return defined_prompt
    prompt_name = task_name + '_prompts'
    prompt = '{question}'
    if prompt_name in singleturn_prompt_dict:
        prompt = singleturn_prompt_dict[prompt_name][assigned_ids]
        print(f'Using prompt pool prompt: {prompt} for task: {task_name}')
        return prompt
    print(f'No prompt defined for task: {task_name}. Make sure you have the key \'question\' in dataset for prompt.')
    return prompt

def multiturn_prompt(
        task_name, 
        assigned_ids = 0,
        defined_prompt = None,
        **kwargs
    ):
    '''
        return [prompt_1: str, prompt_2: str]
    '''
    print('Using multiturn prompt...')
    if defined_prompt is not None:
        if isinstance(defined_prompt, str):
            defined_prompt = [defined_prompt, defined_prompt]
        print(f'Using user defined prompt: {defined_prompt} for task: {task_name}')
        return defined_prompt
    prompt_name = task_name + '_multiturn_prompts'
    prompt = ['{question}', '{question}']
    if prompt_name in multiturn_prompt_dict:
        prompt = multiturn_prompt_dict[prompt_name][assigned_ids]
        if isinstance(prompt, str):
            prompt = [prompt, prompt]
        print(f'Using prompt pool prompt: {prompt} for task: {task_name}')
        return prompt
    print(f'No prompt defined for task: {task_name}. Make sure you have the key \'question\' in dataset for prompt.')
    return prompt

prompt_func_dict = {
    'singleturn': singleturn_prompt,
    'multiturn': multiturn_prompt,
}

def build_prompt(prompt_type, **kwargs):
    return prompt_func_dict[prompt_type](**kwargs)

if __name__ == "__main__":
    test_dict = {
        'coarse_grained_classification': ['singleturn'],
        'fine_grained_classification': ['singleturn', 'multiturn'],
        'VQA': ['singleturn'],
        'counting': ['singleturn'],
        'caption': ['singleturn'],
        'detection': ['singleturn', 'multiturn']
    }
    for key, value in test_dict.items():
        for prompt_type in value:
            prompt = build_prompt(prompt_type, task_name=key, assigned_ids = -1)
    
    for key, value in test_dict.items():
        for prompt_type in value:
            prompt = build_prompt(prompt_type, task_name=key, defined_prompt='Test question:')


    
