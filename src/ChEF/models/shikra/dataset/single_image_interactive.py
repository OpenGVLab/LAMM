import copy
from typing import Optional

from PIL import Image

from .single_image_convsation import SingleImageConvDatasetMixin


class SingleImageInteractive(SingleImageConvDatasetMixin):
    _printed_sample = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image: Optional[Image.Image] = None
        self.roles = ('human', 'gpt')
        self.boxes = []
        self.points = []
        self.raw_conv = []
        self.conversations = []

    def set_image(self, image: Image.Image):
        assert self.image is None, f"{image}"
        self.image = image

    def append_message(self, role: str, message: str, *, boxes=None, points=None, boxes_seq=None, points_seq=None):
        """Append a new message."""
        assert role in self.roles

        def convert_idx(objs_seq, objs_value, get_obj_idx_func):
            if objs_seq is None:
                return None
            ret = []
            for objs_idx in objs_seq:
                new_objs_idx = []
                for idx in objs_idx:
                    new_idx = get_obj_idx_func(objs_value[idx])
                    new_objs_idx.append(new_idx)
                ret.append(tuple(new_objs_idx))
            return tuple(ret)

        boxes_seq = convert_idx(boxes_seq, boxes, self._get_box_idx)
        points_seq = convert_idx(points_seq, points, self._get_point_idx)

        if self.image is not None:
            previous_message_has_image_placeholder = any(
                '<image>' in item['value'] for item in self.conversations
            )
            if not previous_message_has_image_placeholder and '<image>' not in message:
                message = '<image> ' + message
            if previous_message_has_image_placeholder and '<image>' in message:
                message = message.replace('<image>', '')

        self.conversations.append(
            {
                'from': role,
                'value': message,
                'boxes_seq': copy.deepcopy(boxes_seq),
                'points_seq': copy.deepcopy(points_seq),
            }
        )

    def get_raw_item(self, index=None):
        ret = copy.deepcopy({
            'image': self.image,
            'target': {
                'boxes': self.boxes,
                'points': self.points,
            },
            'conversations': self.conversations,
        })
        assert ret['conversations'][0]['from'] == self.roles[0]
        if ret['conversations'][-1]['from'] == self.roles[0]:
            ret['conversations'].append(
                {
                    'from': self.roles[1],
                    'value': '',
                }
            )
        return ret

    def __getitem__(self, index, incontext_cfg=None):
        item = self.get_raw_item(index)
        image: Image.Image = item.get('image', None)
        target: Dict[str, Any] = item.get('target', None)
        raw_conv: List[Dict[str, Any]] = item['conversations']
        # transform 
        assert isinstance(image, list) == isinstance(target, list)
        multimage_mode = isinstance(image, list)
        if isinstance(image, list):
            # TODO: validate raw item
            transformed_image, transformed_target = [], []
            for img, tgt in zip(image, target):
                if self.transforms is not None and image is not None:
                    img, tgt = self.transforms(img, tgt)
                if tgt is not None:
                    tgt['width'], tgt['height'] = img.width, img.height
                transformed_image.append(img)
                transformed_target.append(tgt)
            image, target = transformed_image, transformed_target
        else:
            self.validate_raw_item(item)  # only validate for single image.
            if self.transforms is not None and image is not None:
                image, target = self.transforms(image, target)
            has_image = 'image' in item and bool(item['image'])
            has_target = 'target' in item and bool(item['target']) and any(bool(elem) for elem in item['target'].values())
            if has_target and has_image:
                target['width'], target['height'] = image.width, image.height

        # preprocess
        raw_conv = self.process_conv(raw_conv)
        raw_conv, image = self.process_conv_multimage(raw_conv, image)
        raw_conv, _ = self.process_target(raw_conv, target, multimage_mode=multimage_mode)
        conv = self.build_conv(raw_conv)
        if incontext_cfg is not None:
            if incontext_cfg['add_sysmsg']:
                conv.system += incontext_cfg['sysmsg']
        image_dict = self.process_image(image)
        input_text = conv.get_prompt()

        # return
        ret_dict = {}
        ret_dict.update(dict(
            input_text = input_text
        ))
        ret_dict.update(image_dict)
        return ret_dict


    def to_model_input(self, incontext_cfg=None):
        item = self.__getitem__(0, incontext_cfg)
        ret = {'input_text': item['input_text']}
        if 'image' in item and item['image'] is not None:
            ret['images'] = item['image'].unsqueeze(0).cuda()
        else:
            ret['images'] = None
        return ret

    def to_gradio_chatbot_new_messages(self):
        conv = self.__getitem__(0, return_conv=True)
        new_messages = conv.messages[-2:]
        ret_messages = []
        for r, m in new_messages:
            nm = m.replace('<im_patch>', '').replace('<im_end>', '').replace('<im_start>', '<image>')
            ret_messages.append((r, nm))
        return ret_messages

    def _get_box_idx(self, box):
        assert isinstance(box, (tuple, list)), f"{type(box)}"
        assert isinstance(box[0], (int, float)), f"{type(box[0])}"
        assert len(box) == 4
        box = tuple(box)
        if box not in self.boxes:
            self.boxes.append(box)
            return len(self.boxes) - 1
        else:
            return self.boxes.index(box)

    def _get_point_idx(self, point):
        assert isinstance(point, (tuple, list))
        assert isinstance(point[0], (int, float))
        assert len(point) == 2
        point = tuple(point)
        if point not in self.points:
            self.points.append(tuple(point))
            return len(self.points) - 1
        else:
            return self.points.index(point)

    def __len__(self):
        return 1
