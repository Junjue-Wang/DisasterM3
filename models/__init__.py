import warnings
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    pass

from qwen_vl_utils import process_vision_info
from torchvision.transforms import InterpolationMode
from transformers import AutoProcessor, AutoTokenizer
import torchvision.transforms as T


def messages_contain_video(messages: List[Dict]) -> bool:
    contain_video = False
    for msg in messages:
        if not isinstance(msg["content"], list):
            continue
        for content in msg["content"]:
            if content["type"] == "video":
                contain_video = True
                break
        if contain_video:
            break
    return contain_video


###################################################################################################
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def load_video_frame_np(video_path, input_size=448, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
    ])
    np_img_list = []
    frame_indices = get_index(None, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        np_img_list.append(
            transform(Image.fromarray(vr[frame_index].asnumpy()).convert('RGB'))
        )
    np_video_frames = np.stack(np_img_list, axis=0)
    return np_video_frames
###################################################################################################


class ModelConfig(ABC):
    default_engine_args: Dict = dict(generation_config="auto")

    @abstractmethod
    def get_prompt_from_question(self, messages: List[Dict]):
        raise NotImplementedError


class QwenVL(ModelConfig):
    def __init__(
            self, model_id: str, max_model_len: int = None, max_tokens: int = None,
            max_num_frames: int = 32, video_min_pixels: int = 256 * 28 * 28,
            video_max_pixels: int = 2048 * 28 * 28,
    ):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.default_engine_args["model"] = model_id
        self.default_engine_args["config_format"] = "hf"

        if max_tokens is not None and max_tokens > 32768:
            warnings.warn(f"maximum context length of qwen2.5 is 32768 tokens smaller than your input {max_tokens}! "
                           f"Clip your max_tokens to 32768.")
            max_tokens = 32768
        if max_tokens is not None:
            self.default_engine_args["override_generation_config"] = {"max_tokens": max_tokens}

        if "3b" in model_id.lower() or "7b" in model_id.lower() or "14b" in model_id.lower():
            self.default_engine_args["tensor_parallel_size"] = 1
        elif "32b" in model_id.lower():
            self.default_engine_args["tensor_parallel_size"] = 2
        elif "72b" in model_id.lower():
            self.default_engine_args["tensor_parallel_size"] = 4
            if max_model_len is None:
                if "qwen2.5" in model_id.lower():
                    print("If not given, max_model_len is default to be 50000 for qwen2.5 72B model to feed in 4 H100 GPUs")
                    max_model_len = 50000
                elif "qwen2" in model_id.lower():
                    print("If not given, max_model_len is default to be 32768 for qwen2 72B model to feed in 4 H100 GPUs")
                    max_model_len = 32768
        else:
            raise ValueError("Unknown model_id: {}".format(model_id))

        if max_model_len is not None:
            self.default_engine_args.update(
                max_model_len=max_model_len,
                max_num_batched_tokens=max_model_len,
            )

        self.max_num_frames = max_num_frames
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels

    def get_prompt_from_question(self, messages: List[Dict]):
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if messages_contain_video(messages):
            for msg in messages:
                if not isinstance(msg["content"], list):
                    continue
                for content in msg["content"]:
                    if content["type"] == "video":
                        content["min_pixel"] = self.video_min_pixels
                        content["max_pixel"] = self.video_max_pixels
                        content["fps"] = 1.0

            _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            total_frames = video_inputs[0].shape[0]
            if total_frames > self.max_num_frames:
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]

            multimodal_inputs = {
                "multi_modal_data": {"video": video_inputs},
            }
            return prompt_text, multimodal_inputs
        else:
            return prompt_text


# System prompt of LLaMA-Factory's intern_vl template. The -hf chat template has none.
INTERNVL_SYSTEM_PROMPT = "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"


class InternVL(ModelConfig):

    def __init__(self, model_id: str, max_model_len: int = None, max_tokens: int = None, max_num_frames: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Only HF-format checkpoints (InternVLForConditionalGeneration) define this token.
        self.hf_image_token = getattr(self.tokenizer, "context_image_token", None)
        self.default_engine_args["model"] = model_id
        self.default_engine_args["trust_remote_code"] = True
        if self.hf_image_token is not None:
            # Training uses one 448x448 tile per image; InternVLProcessor defaults to crop_to_patches=True.
            self.default_engine_args["mm_processor_kwargs"] = {"crop_to_patches": False}
        if max_tokens is not None:
            self.default_engine_args["override_generation_config"] = {"max_tokens": max_tokens}

        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        if "override_generation_config" not in self.default_engine_args:
            self.default_engine_args["override_generation_config"] = {}
        self.default_engine_args["override_generation_config"]["stop_token_ids"] = stop_token_ids

        if "-8b" in model_id.lower() or "-14b" in model_id.lower():
            self.default_engine_args["tensor_parallel_size"] = 1
        elif "-78b" in model_id.lower():
            self.default_engine_args["tensor_parallel_size"] = 4
            if max_model_len is None:
                print("If not given, max_model_len is default to be 32768 for 78B model to feed in 4 H100 GPUs")
                max_model_len = 32768
        else:
            raise ValueError("Unknown model_id: {}".format(model_id))

        if max_model_len is not None:
            self.default_engine_args.update(
                max_model_len=max_model_len,
                max_num_batched_tokens=max_model_len,
            )

        self.max_num_frames = max_num_frames

    def get_prompt_from_question(self, messages: List[Dict]):
        if messages_contain_video(messages):
            multimodal_inputs = {}

            for msg_idx, msg_dict in enumerate(messages):
                content_str = ""
                if isinstance(msg_dict["content"], list):
                    for content_dict in msg_dict["content"]:
                        if content_dict["type"] == "text":
                            content_str += content_dict["text"]
                        elif content_dict["type"] == "video":
                            pixel_values, num_patches_list = load_video(content_dict["video"], num_segments=self.max_num_frames)
                            pixel_values = pixel_values.numpy()
                            video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
                            content_str += video_prefix
                            multimodal_inputs["multi_modal_data"] = {"image": [Image.fromarray(pix_val.transpose(1, 2, 0).astype(np.uint8)) for pix_val in pixel_values]}
                        else:
                            raise ValueError("Unknown content type: {}".format(content_dict["type"]))

                elif isinstance(msg_dict["content"], str):
                    content_str += msg_dict["content"]

                else:
                    raise ValueError("Unknown content type: {}".format(msg_dict["content"]))

                msg_dict["content"] = content_str

            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return prompt, multimodal_inputs

        else:
            for msg_idx, msg_dict in enumerate(messages):
                content_str = ""
                if isinstance(msg_dict["content"], list):
                    for content_dict in msg_dict["content"]:
                        if content_dict["type"] == "text":
                            content_str += content_dict["text"]
                        elif content_dict["type"] == "image":
                            content_str += (self.hf_image_token or "<image>") + "\n"
                        else:
                            raise ValueError("Unknown content type: {}".format(content_dict["type"]))

                elif isinstance(msg_dict["content"], str):
                    content_str += msg_dict["content"]

                else:
                    raise ValueError("Unknown content type: {}".format(msg_dict["content"]))

                # Training has no extra newline after an image when the next text starts with a blank line.
                if self.hf_image_token is not None:
                    content_str = content_str.replace(self.hf_image_token + "\n\n\n", self.hf_image_token + "\n\n")
                msg_dict["content"] = content_str

            if self.hf_image_token is not None:
                messages = [{"role": "system", "content": INTERNVL_SYSTEM_PROMPT}] + messages
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return prompt


class Llava(ModelConfig):
    def __init__(self, model_id: str, max_model_len: int = None, max_tokens: int = None, max_num_frames: int = 32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.chat_template is None:
            assert model_id == "llava-hf/llava-1.5-7b-hf", model_id

        self.default_engine_args["model"] = model_id
        if max_tokens is not None:
            self.default_engine_args["override_generation_config"] = {"max_tokens": max_tokens}

        if "-7b" in model_id.lower():
            self.default_engine_args["tensor_parallel_size"] = 1
        elif "-72b" in model_id.lower():
            self.default_engine_args["tensor_parallel_size"] = 4
        else:
            raise ValueError("Unknown model_id: {}".format(model_id))

        if max_model_len is not None:
            self.default_engine_args.update(
                max_model_len=max_model_len,
                max_num_batched_tokens=max_model_len,
            )

        self.max_num_frames = max_num_frames

    def get_prompt_from_question(self, messages: List[Dict]):
        if getattr(self.tokenizer, "chat_template", None) is not None:
            if messages_contain_video(messages):
                multimodal_inputs = {}

                for msg_idx, msg_dict in enumerate(messages):
                    content_str = ""
                    for content_dict in msg_dict["content"]:
                        if content_dict["type"] == "text":
                            content_str += content_dict["text"]
                        elif content_dict["type"] == "video":
                            if isinstance(content_dict["video"], str):
                                pixel_values = load_video_frame_np(content_dict["video"], num_segments=self.max_num_frames)
                            elif isinstance(content_dict["video"], list):
                                pixel_values = [Image.open(frame_path).convert("RGB").resize((448, 448), resample=Resampling.BILINEAR) for frame_path in content_dict["video"]]
                                pixel_values = np.array(pixel_values)
                            else:
                                raise ValueError("Unknown content type: {}".format(content_dict["type"]))

                            content_str += "<video>"
                            multimodal_inputs["multi_modal_data"] = {"video": pixel_values}
                        else:
                            raise ValueError("Unknown content type: {}".format(content_dict["type"]))

                    msg_dict["content"] = content_str

                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                return prompt, multimodal_inputs

            else:
                for msg_idx, msg_dict in enumerate(messages):
                    content_str = ""
                    for content_dict in msg_dict["content"]:
                        if content_dict["type"] == "text":
                            content_str += content_dict["text"]
                        elif content_dict["type"] == "image":
                            content_str += "<image>\n"
                        else:
                            raise ValueError("Unknown content type: {}".format(content_dict["type"]))

                    msg_dict["content"] = content_str

                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                return prompt

        # llava v1.5 case
        else:
            assert not messages_contain_video(messages), messages
            prompt = ""
            for msg_idx, msg_dict in enumerate(messages):
                if msg_dict["role"] == "user":
                    prompt += "USER: "
                elif msg_dict["role"] == "assistant":
                    prompt += "\nASSISTANT:"
                else:
                    raise ValueError("Unknown message type: {}".format(msg_dict["type"]))

                for content_dict in msg_dict["content"]:
                    if content_dict["type"] == "text":
                        prompt += content_dict["text"]
                    elif content_dict["type"] == "image":
                        prompt += "<image>\n"
                    else:
                        raise ValueError("Unknown content type: {}".format(content_dict["type"]))

            if messages[-1]["role"] != "assistant":
                prompt += "\nASSISTANT:"

            return prompt


def build_model_config(model_id: str, **kwargs) -> ModelConfig:
    if "llava" in model_id.lower():
        model_config = Llava(model_id=model_id, **kwargs)
    elif "qwen" in model_id.lower() and "vl" in model_id.lower():
        model_config =  QwenVL(model_id=model_id, **kwargs)
    elif "intern" in model_id.lower() and "vl" in model_id.lower():
        model_config =  InternVL(model_id=model_id, **kwargs)
    else:
        raise NotImplementedError(model_id)

    engine_args = model_config.default_engine_args
    if "model" not in engine_args:
        engine_args["model"] = model_id
    return model_config
