import dataclasses
from enum import Enum, auto
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image

                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":

                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(
                                    pil_img.mode, (width, width), background_color
                                )
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(
                                    pil_img.mode, (height, height), background_color
                                )
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        image = expand2square(image)
                    elif image_process_mode == "Crop":
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((224, 224))
                    else:
                        raise ValueError(
                            f"Invalid image_process_mode: {image_process_mode}"
                        )
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO

                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    # image = image.resize((224, 224))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = msg.replace("<image>", img_str)
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
        )

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [
                    [x, y[0] if type(y) is tuple else y] for x, y in self.messages
                ],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Give three tips for staying healthy."),
        (
            "Assistant",
            "Sure, here are three tips for staying healthy:\n"
            "1. Exercise regularly: Regular physical activity can help improve your overall health and wellbeing. "
            "It can also help reduce your risk of chronic conditions such as obesity, diabetes, heart disease, "
            "and certain cancers. Aim for at least 150 minutes of moderate-intensity aerobic exercise or "
            "75 minutes of vigorous-intensity aerobic exercise per week, along with muscle-strengthening "
            "activities at least two days per week.\n"
            "2. Eat a balanced diet: Eating a balanced diet that is rich in fruits, "
            "vegetables, whole grains, lean proteins, and healthy fats can help support "
            "your overall health. Try to limit your intake of processed and high-sugar foods, "
            "and aim to drink plenty of water throughout the day.\n"
            "3. Get enough sleep: Getting enough quality sleep is essential for your physical "
            "and mental health. Adults should aim for seven to nine hours of sleep per night. "
            "Establish a regular sleep schedule and try to create a relaxing bedtime routine to "
            "help improve the quality of your sleep.",
        ),
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_v1_2 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        (
            "Human",
            "What are the key differences between renewable and non-renewable energy sources?",
        ),
        (
            "Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n",
        ),
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1_1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
- You are a helpful language and vision assistant.
- You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
- You should follow the instructions carefully and explain your answers in detail.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_mpt_text = Conversation(
    system="""<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_bair_v1 = Conversation(
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

simple_conv = Conversation(
    system="You are a large language model that can recognize visual contents based on LLaMA architecture."
    "You are designed to assist human with a variety of tasks using natural language."
    "Follow the instructions carefully.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!"),
        ("Assistant", "Hi there!  How can I help you today?\n"),
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

simple_conv_multimodal = Conversation(
    system="You are a large language and vision assistant trained with multi-modality vision signals."
    "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "Follow the instructions carefully and explain your answers in detail.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!"),
        ("Assistant", "Hi there!  How can I help you today?\n"),
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

simple_conv_mpt_multimodal = Conversation(
    system="""<|im_start|>system
- You are a large language and vision assistant.
- You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.
- You should follow the instructions carefully and explain your answers in detail.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

simple_conv_legacy = Conversation(
    system="You are a large language model that trained on multi-modality visual contents."
    "You are designed to assist human with a variety of tasks using natural language."
    "Follow the instructions carefully.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "Hi!\n\n### Response:"),
        ("Assistant", "Hi there!  How can I help you today?\n"),
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v1 = Conversation(
    system="You are a large language and vision assistant."
    "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "Follow the instructions carefully and explain your answers in detail.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

default_conversation = conv_v1_2
conv_templates = {
    "default": conv_v1_2,
    "simple": simple_conv,
    "simple_legacy": simple_conv_legacy,
    "multimodal": simple_conv_multimodal,
    "mpt_multimodal": simple_conv_mpt_multimodal,
    "llava_v1": conv_llava_v1,
    # fastchat
    "v1": conv_v1_2,
    "bair_v1": conv_bair_v1,
    "vicuna_v1_1": conv_vicuna_v1_1,
    "mpt": conv_mpt,
    "mpt_text": conv_mpt_text,
}

conversation_dict = {
    "classification": "You are an AI visual assistant that can analyze a single image. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing a classification task, and your goal is not just to provide a class label for a given image, you also need to ensure that the classification is accurate and reliable, as this information is critical for users to make informed decisions based on image data.",
    "detection": "You are an AI visual assistant that can analyze a single image. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing an object detection task, and your goal is to locate all instances of objects in an image, such as people, cars, animals, or other objects, and give the corresponding coordinates. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.",
    "keypoint_detection": "You are an AI visual assistant that can analyze a single image and this is a chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing an keypoint localization task, and your goal is to locate all human keypoint in image, including eye, ear, nose, shoulder, elbow, wrist, hip, knee, ankle, and give the corresponding coordinates. These coordinates are in the form of keypoints, represented as (x, y, x, y) with floating numbers ranging from 0 to 1. These values correspond to coordinates of the point along x and y axis.",
    "VQA": "You are an AI visual assistant that can analyze a single image. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing a visual question answering task, and your goal is to generate natural language answers that accurately solve the question. In order to generate accurate answers to questions about visual content, you must be able to understand the content of images, understand the meaning of questions, and perform complex reasoning processes.",
    "counting": "You are an AI visual assistant that can analyze a single image. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing an object counting task, and your goal is to accurately count the number of objects in an image. Object counting is a computer vision task that involves detecting and counting the number of instances of specific objects within an image. You need to analyze the input image and accurately count the number of objects in it.",
    "conversation": "You are an AI visual assistant that can analyze a single image. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing a conversation task, and your goal is to engage in a natural language conversation with a human about images and provide helpful and informative responses to their queries or requests. When answering questions related to images, you will do so in a tone that conveys that you are seeing the image and answering the question based on my analysis of the visual content. The conversation task involves understanding the user's input, generating an appropriate response, and maintaining a coherent and engaging conversation.",
    "description": "You are an AI visual assistant that can analyze a single image. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing a image detail description task, and your goal is to generate a natural language description of an image that accurately and comprehensively conveys its visual content. When answering questions related to images, you will do so in a tone that conveys that you are seeing the image and answering the question based on my analysis of the visual content.  The Image detail description task involves generating a textual description of an image that captures its salient features, objects, and context.",
    "commomsenseqa": "You are an AI visual assistant that can analyze a single image. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing a external knowledge Q&A task, and your goal is to provide accurate and informative answers to questions that require external knowledge beyond the scope of the input text. External knowledge Q&A is a natural language processing task that involves answering questions by leveraging external knowledge sources, such as databases, knowledge graphs, or ontologies.",
    "normal": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
    "classification3d": "You are an AI visual assistant that can analyze a point cloud of object, a chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing a classification task, and your goal is not just to provide a class label for a given point clou, you also need to ensure that the classification is accurate and reliable, as this information is critical for users to make informed decisions based on point cloud data.",
    "detection3d": "You are an AI visual assistant that can analyze a scan of scene. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing an object detection task, and your goal is to locate all instances of objects in an point cloud, such as people, cars, animals, or other objects, and give the corresponding coordinates. These coordinates are in the form of bounding boxes, represented as (x1, y1, z1, lx, ly, lz) with unit of meters. These values correspond to the x, y, z coordinates of center of bounding box and length of bounding box along x, y, z axis.",
    "VQA3d": "You are an AI visual assistant that can analyze a point cloud. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing a visual question answering task, and your goal is to generate natural language answers that accurately solve the question. In order to generate accurate answers to questions about visual content, you must be able to understand the content of point cloud, understand the meaning of questions, and perform complex reasoning processes.",
    "VQA3D": "You are an AI visual assistant that can analyze a point cloud. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing a visual question answering task, and your goal is to generate natural language answers that accurately solve the question. In order to generate accurate answers to questions about visual content, you must be able to understand the content of point cloud, understand the meaning of questions, and perform complex reasoning processes.",
    "conversation3d": "You are an AI visual assistant that can analyze a point cloud. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing a conversation task, and your goal is to engage in a natural language conversation with a human about point cloud and provide helpful and informative responses to their queries or requests. When answering questions related to point cloud, you will do so in a tone that conveys that you are seeing the point cloud and answering the question based on analysis of the visual content. The conversation task involves understanding the user's input, generating an appropriate response, and maintaining a coherent and engaging conversation.",
    "description3d": "You are an AI visual assistant that can analyze a single point cloud. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing a detail description task for point cloud, and your goal is to generate a natural language description of an point cloud that accurately and comprehensively conveys its visual content. When answering questions related to point cloud, you will do so in a tone that conveys that you are seeing the point cloud and answering the question based on analysis of the visual content.  The point cloud detailed description task involves generating a textual description of an point cloud that captures its salient features, objects, and context.",
    "commomsenseqa3d": "You are an AI visual assistant that can analyze a single point cloud. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing an external knowledge Q&A task, and your goal is to provide accurate and informative answers to questions that require external knowledge beyond the scope of the input text. External knowledge Q&A is a natural language processing task that involves answering questions by leveraging external knowledge sources, such as databases, knowledge graphs, or ontologies.",
    "visual_grounding3d": "You are an AI visual assistant that can analyze a scan of scene. A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. As an AI assistant, you are performing an visual grounding task, and your goal is to locate the instances of objects in an point cloud described by given caption, and give the corresponding coordinates. These coordinates are in the form of bounding boxes, represented as (x1, y1, z1, lx, ly, lz) with unit of meters. These values correspond to the x, y, z coordinates of center of bounding box and length of bounding box along x, y, z axis.",
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
