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
    'What is the fine-grained category label for this image?',
]
fine_grained_classification_multiturn_prompts = [
    ['What is the fine-grained category label for this image?',
     'As the coarse-grained category label for this image is {}, what is the fine-grained category label for this image?']
     # replace {} with the fore_label (defined in scenario) 
]

# classification answer templates for ppl inference
coarse_grained_classification_templates = [
    'The object in the image is {}',
    '{}',
]
fine_grained_classification_templates = [
    'The fine-grained category label for this image is {}',
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

# caption answer templates for ppl inference
caption_templates = [
    'The caption for this image is \" {}',
    '{}',
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

# vqa answer templates for ppl inference
vqa_templates = [
    'The answer is {}',
    'The correct option for the question is {}',
    '{}',
]

# counting, question defined in dataset
# example: 'How many {} are there in this image?'
counting_prompts = [
    "",
]

# counting answer templates for ppl inference
counting_templates = [
    "{}",
]

# detection
# We only provide the multi-turn inference for detection tasks.
# Two-turn prompts, with the first turn query for the category and the second turn query for the bounding box.
detection_multi_turn_prompts = [
    ['The image shows',
     'Give all the bounding boxes of {} in the image. The bounding box should be represented as [x1, y1, x2, y2] with floating numbers indicating the coordinates of the object in a normalized range of 0-1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.' # replace {} with the fore_label
    ],
    ['Detect all the objects in the image.', 'Provide coordinates [x0,y0,x1,y1] for {} in the image.'], # For shikra
    ['Detect all the objects in the image.', '<grounding><phrase>{}</phrase>'] # For kosmos2
]

# templates for two-turn ppl inference
detection_templates = [
    ['The object in the image is {}', 'It is located at the bbox {}'], # default
    ['The object in the image is {}', 'The {}'], # For shikra
    ['The object in the image is {}', '{}'] # For kosmos2
]


# POPE TODO: VQA
# The prompts for POPE are defined in dataset. 
pope_prompts = [
    "",
]
# POPE answer templates for ppl inference
pope_templates=[ 
    "{}",
    'The answer is {}'
]


query_pool_dict = {
    'coarse_grained_classification_prompts': coarse_grained_classification_prompts,
    'fine_grained_classification_prompts': fine_grained_classification_prompts,
    'caption_prompts': caption_prompts,
    'VQA_prompts': vqa_prompts,
    'counting_prompts': counting_prompts,
    'POPE_prompts':pope_prompts
}
ppl_template_dict = {
    'coarse_grained_classification_templates': coarse_grained_classification_templates,
    'fine_grained_classification_templates': fine_grained_classification_templates,
    'VQA_templates': vqa_templates,
    'counting_templates': counting_templates,
    'caption_templates': caption_templates,
    'detection_templates': detection_templates,
    'POPE_templates':pope_templates,
}

multiturn_query_dict = {
    'fine_grained_classification_multiturn_prompts':
    fine_grained_classification_multiturn_prompts,
    'detection_multiturn_prompts':
    detection_multi_turn_prompts,
}

def query_from_query_pool(
        task_name,
        assigned_ids = 0,
        **kwargs
        ):
    prompt_name = task_name + '_prompts'
    query_pool = query_pool_dict[prompt_name]
    query = query_pool[assigned_ids]
    return query

def query_from_standard_query(
        task_name,
        **kwargs
        ):
    return query_from_query_pool(task_name, assigned_ids=0)

def multiturn_query_from_query_pool(
        task_name, 
        assigned_ids = 0,
        **kwargs
    ):
    """
        return : list of tuple (prompt, multiturn_prompt, answer_template)
    """
    multiturn_name = task_name + '_multiturn_prompts'
    multiturnppl_pool = multiturn_query_dict[multiturn_name]
    multiturn_query = multiturnppl_pool[assigned_ids]
    return multiturn_query

def ppl_template(
        task_name, 
        assigned_ids = 0,
        query_type = '', 
        **kwargs
    ):
    template_name = task_name + '_templates'
    template_pool = ppl_template_dict[template_name]
    template = template_pool[assigned_ids]
    if query_type == 'multiturn' and isinstance(template, str): # each turn uses the same template
        template = [template, template]
    return template

query_func_dict = {
    'standard_query': query_from_standard_query,
    'query_pool': query_from_query_pool,
    'multiturn': multiturn_query_from_query_pool,
}


def build_query(query_type, **kwargs):
    build_fuc = query_func_dict[query_type]
    return build_fuc(**kwargs)

def build_template(**kwargs):
    return ppl_template(**kwargs)




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