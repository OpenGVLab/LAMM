# classification
coarse_grained_classification_answer_templates = [
    'The object in the image is {option}',
    '{option}',
]
fine_grained_classification_answer_templates = [
    'The fine-grained category label for this image is {option}',
]

# caption
caption_answer_templates = [
    'The caption for this image is \" {option}',
    '{option}',
]

# vqa
vqa_answer_templates = [
    'The answer is {option}',
    'The correct option for the question is {option}',
    '{option}',
]

# counting
counting_answer_templates = [
    "{option}",
]

# detection
detection_answer_templates = [
    ['The object in the image is {option}', 'It is located at the bbox {option}'], # default
    ['The object in the image is {option}', 'The {option}'], # For shikra
    ['The object in the image is {option}', '{option}'] # For kosmos2
]

# POPE
pope_answer_templates=[ 
    "{option}",
    'The answer is {option}'
]

# octavius3d
octavius3d_answer_templates = [
    "{option}",
]


answer_template_dict = {
    'coarse_grained_classification_templates': coarse_grained_classification_answer_templates,
    'fine_grained_classification_templates': fine_grained_classification_answer_templates,
    'VQA_templates': vqa_answer_templates,
    'counting_templates': counting_answer_templates,
    'caption_templates': caption_answer_templates,
    'detection_templates': detection_answer_templates,
    'POPE_templates': pope_answer_templates,
    'Classification_octavius3d_templates': octavius3d_answer_templates,
    'Caption_octavius3d_templates': octavius3d_answer_templates,
    'VQA_octavius3d_templates': octavius3d_answer_templates,
}

def ppl_template(
        task_name, 
        assigned_ids = 0,
        defined_template = None,
        **kwargs
    ):
    if defined_template is not None:
        pass
    template_name = task_name + '_templates'
    template = '{}'
    if template_name in ppl_template_dict:
        template_pool = ppl_template_dict[template_name]
        template = template_pool[assigned_ids]
    if query_type == 'multiturn' and isinstance(template, str): # each turn uses the same template
        template = [template, template]
    return template

def build_template(**kwargs):
    # LAMM-style inference does not require template
    if kwargs['task_name'].endswith('lamm'):
        return None

    return ppl_template(**kwargs)