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

fine_grained_classification_multiturn_prompts = [
    'As the coarse-grained category label for this image is {prefix}, what is the fine-grained category label for this image?',
]

# LAMM-style classfication prompts
classification_lamm_prompts = [
    '',
]

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
caption_lamm_prompts = [
    '',
]

# vqa
vqa_prompts = [
    'The answer is',
    'What is the correct option for this question?',
    '',
    'What is the answer?',
    'The answer for the question is',
    'ANSWER:',
    'The answer (option) is',
    'Answer:',
]

# LAMM-style standard prompts
vqa_lamm_prompts = [
    '',
]

Classification_octavius3d_prompts = ['']
VQA_octavius3d_prompts = ['']
Caption_octavius3d_prompts = ['']

# counting, question defined in dataset
# example: 'How many {} are there in this image?'
counting_prompts = [
    "",
]

# detection
# We provide both LAMM-style standard prompts and multi-turn prompts
detection_lamm_prompts = [
    'Identify all the objects in the image and provide their positions. Your answer needs to give the object name and the bounding box of the object. The bounding box should be represented as [x1, y1, x2, y2] with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.'
]

# Two-turn detection prompts, with the first turn query for the category and the second turn query for the bounding box.
detection_multi_turn_prompts = [
    ['The image shows',
     'Give all the bounding boxes of {prefix} in the image. The bounding box should be represented as [x1, y1, x2, y2] with floating numbers indicating the coordinates of the object in a normalized range of 0-1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.'
    ],
    ['Detect all the objects in the image.', 'Provide coordinates [x0,y0,x1,y1] for {prefix} in the image.'], # For shikra
    ['Detect all the objects in the image.', '<grounding><phrase>{prefix}</phrase>'] # For kosmos2
]

# POPE
# The prompts for POPE are defined in dataset. 
pope_prompts = [
    "",
]

singleturn_query_dict = {
    'coarse_grained_classification_prompts': coarse_grained_classification_prompts,
    'caption_prompts': caption_prompts,
    'VQA_prompts': vqa_prompts,
    'counting_prompts': counting_prompts,
    'POPE_prompts':pope_prompts,
    'detection_lamm_prompts': detection_lamm_prompts,
    'VQA_lamm_prompts' : vqa_lamm_prompts,
    'caption_lamm_prompts' : caption_lamm_prompts,
    'Facial_cls_lamm_prompts' : classification_lamm_prompts,
    'classification_lamm_prompts': classification_lamm_prompts,
    'Classification_octavius3d_prompts': Classification_octavius3d_prompts,
    'VQA_octavius3d_prompts': VQA_octavius3d_prompts,
    'Caption_octavius3d_prompts': Caption_octavius3d_prompts,
}

multiturn_query_dict = {
    'fine_grained_classification_multiturn_prompts':
    fine_grained_classification_multiturn_prompts,
    'detection_multiturn_prompts':
    detection_multi_turn_prompts,
}

def singleturn_query(
        task_name,
        assigned_ids = 0,
        defined_query = None,
        **kwargs
        ):
    if defined_query is not None:
        assert isinstance(defined_query, str), f'The defined query must be string. '
    prompt_name = task_name + '_prompts'
    query = ''

    query = singleturn_query_dict.get(prompt_name, )
    if prompt_name in query_pool_dict:
        query_pool = query_pool_dict[prompt_name]
        query = query_pool[assigned_ids]
    return query

def multiturn_query(
        task_name, 
        assigned_ids = 0,
        **kwargs
    ):
    multiturn_name = task_name + '_multiturn_prompts'
    multiturn_query = ['', '']
    if multiturn_name in multiturn_query_dict:
        multiturnppl_pool = multiturn_query_dict[multiturn_name]
        multiturn_query = multiturnppl_pool[assigned_ids]
    return multiturn_query

query_func_dict = {
    'singleturn': singleturn_query,
    'multiturn': multiturn_query,
}

def build_query(query_type, **kwargs):
    return query_func_dict[query_type](**kwargs)

if __name__ == "__main__":
    test_dict = {
        'coarse_grained_classification': ['standard_query', 'query_pool'],
        'fine_grained_classification': ['standard_query', 'query_pool', 'multiturn'],
        'VQA': ['standard_query', 'query_pool'],
        'counting': ['standard_query', 'query_pool'],
        'caption': ['standard_query', 'query_pool'],
        'detection': ['multiturn'],
        'POPE': ['standard_query', 'query_pool'],
    }
    for key, value in test_dict.items():
        for query_type in value:
            query = build_query(query_type, task_name = key, assigned_ids = -1)
            print(f'The query from {query_type} for {key} is "{query}".')

    for key in test_dict.keys():
        template = build_template(task_name = key, assigned_ids = -1)
        print(f'The template for {key} is "{template}".')
