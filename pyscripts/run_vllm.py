import argparse
import json
import os
from dataclasses import asdict
from os.path import dirname, abspath, join
from typing import Dict, List

from PIL import Image
from PIL.Image import Resampling
from tqdm import tqdm
from transformers import GenerationConfig
from vllm import EngineArgs, LLM, SamplingParams

from models import build_model_config, ModelConfig


PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))



prompt_libs = dict(
    bearing_body="""Analyze both the pre-disaster and post-disaster images to answer the following question. Choose the best option(s) from the candidate options provided.

pre-disaster image:
<image>

post-disaster image:
<image>

Question: {prompt}
Options: {options_str}

Your task is to respond with ONLY the capital letters of the correct options, separated by a comma and a space (e.g., C, D, H). Do not include any explanation or other text.""",
    landuse="""Analyze the image to answer the following question. Choose the best option(s) from the candidate options provided.

Question: {prompt}
Options: {options_str}

Your task is to respond with ONLY the capital letters of the correct options, separated by a comma and a space (e.g., C, D, H). Do not include any explanation or other text.""",
    single_choice="""Analyze both the pre-disaster and post-disaster images to answer the following question. Choose the best option from the candidate options provided.

pre-disaster image:
<image>

post-disaster image:
<image>

Question: {prompt}
Options: {options_str}

Your task is to respond with ONLY the capital letter of the correct option (e.g., C). Do not include any explanation or other text.""",
    relational_reasoning_qa="""Analyze the image to answer the following question. Choose the best option from the candidate options provided.

Question: {prompt}
Options: {options_str}

Your task is to respond with ONLY the capital letter of the correct option (e.g., C). Do not include any explanation or other text.""",
    caption="""Your TASK is to analyze the provided pair of pre-disaster and post-disaster remote sensing images. 
You will act as a remote sensing analyst to identify the type of disaster and assess its impact on both built and natural environments across five specific categories.

pre-disaster image:
<image>

post-disaster image:
<image>

Your analysis must be formatted as follows:
DISASTER: [the name of the disaster]
BUILDING: [describe impacts on buildings]
ROAD: [describe impacts on road networks]
VEGETATION: [describe impacts on natural, unmanaged vegetation cover]
WATER_BODY: [describe changes to water bodies]
AGRICULTURE: [describe impacts on managed agricultural land]
CONCLUSION: [provide a concise 1-2 sentence summary synthesizing the overall disaster impacts observed across the categories.]""",
    recovery="""Your TASK is to generate concise and integrated recovery recommendations for the affected area based on the provided pre-disaster and post-disaster remote sensing images. Aspects to focus on include infrastructure restoration, housing reconstruction, and ecological and geological environment restoration.

pre-disaster image: 
<image>

post-disaster image: 
<image>

Based on your analysis of the images:
1. First determine if recovery actions are necessary. If no significant damage or impact is observed, clearly state no recovery recommendations due to no discernible impact.
2. If recovery is needed, provide recommendations in the following format:
IMMEDIATE_RECOVERY: [Provide an integrated paragraph within 50 words describing immediate recovery actions. Create a flowing narrative.]
LONG_TERM_RECOVERY: [Provide an integrated paragraph within 50 words describing long-term recovery strategies. Create a flowing narrative.]

Ensure your recommendations are realistic, feasible, and properly prioritized based on the visible damage in the images."""
)


def get_messages_from_data(data_dict: Dict, subset: str):
    if subset in ["bearing_body", "building_damage_counting", "disaster_type", "road_damage_counting"]:
        prompt_key = subset if subset == "bearing_body" else "single_choice"
        prompt_text = prompt_libs[prompt_key].format(prompt=data_dict["prompts"], options_str=data_dict["options_str"])
        prompt_splits = prompt_text.split("<image>")
        assert len(prompt_splits) == 3, len(prompt_splits)

        pre_disaster_image_path = join(f"{PROJECT_ROOT}/data", "images", data_dict["pre_image_path"])
        post_disaster_image_path = join(f"{PROJECT_ROOT}/data", "images", data_dict["post_image_path"])
        images = [pre_disaster_image_path, post_disaster_image_path]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_splits[0]},
                    {"type": "image", "image": pre_disaster_image_path},
                    {"type": "text", "text": prompt_splits[1]},
                    {"type": "image", "image": post_disaster_image_path},
                    {"type": "text", "text": prompt_splits[2]},
                ]
            }
        ]

    elif subset in ["landuse", "relational_reasoning_qa"]:
        if subset == "landuse":
            image_path = join(f"{PROJECT_ROOT}/data", "images", data_dict["pre_image_path"])
            prompt_text = prompt_libs[subset].format(prompt=data_dict["prompts"], options_str=data_dict["options_str"])
        else:
            image_path = join(f"{PROJECT_ROOT}/data", data_dict["image_path"].replace("\\", "/"))
            prompt_text = prompt_libs[subset].format(prompt=data_dict["prompts"], options_str=data_dict["option_str"])

        images = [image_path]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ]
            }
        ]

    elif subset in ["caption", "recovery"]:
        prompt_text = prompt_libs[subset]
        prompt_splits = prompt_text.split("<image>")
        assert len(prompt_splits) == 3, len(prompt_splits)

        pre_disaster_image_path = join(f"{PROJECT_ROOT}/data", "images", data_dict["pre_image_path"])
        post_disaster_image_path = join(f"{PROJECT_ROOT}/data", "images", data_dict["post_image_path"])
        images = [pre_disaster_image_path, post_disaster_image_path]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_splits[0]},
                    {"type": "image", "image": pre_disaster_image_path},
                    {"type": "text", "text": prompt_splits[1]},
                    {"type": "image", "image": post_disaster_image_path},
                    {"type": "text", "text": prompt_splits[2]},
                ]
            }
        ]

    else:
        raise ValueError('Unknown subset {}'.format(subset))

    return messages, images


def create_batch_inputs(data_list: List[Dict], model_config: ModelConfig, args):
    for i in range(0, len(data_list), args.batch_size):
        batch_data = data_list[i:i + args.batch_size]
        batch_inputs = []
        batch_metadata = []

        for data_dict in batch_data:
            messages, images = get_messages_from_data(data_dict, args.subset)
            prompt_text = model_config.get_prompt_from_question(messages)

            images = [Image.open(image_path).convert("RGB") for image_path in images]
            if args.image_size is not None:
                for img_idx, image in enumerate(images):
                    images[img_idx] = image.resize((args.image_size, args.image_size), resample=Resampling.BICUBIC)

            inputs = {
                "prompt": prompt_text,
                "multi_modal_data": {"image": images}
            }

            batch_inputs.append(inputs)
            batch_metadata.append(data_dict)

        yield batch_inputs, batch_metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--subset', type=str, required=True)
    parser.add_argument('--max_model_len', type=int, default=None)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--tensor_parallel_size', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=None)
    args = parser.parse_args()

    reformat_model_id = "/".join(args.model_id.strip("/").split("/")[-2:]).replace('/', '--')
    result_save_path = join(PROJECT_ROOT, "results", args.subset, reformat_model_id, "finished.jsonl")
    os.makedirs(dirname(result_save_path), exist_ok=True)

    finish_ids = []
    if not args.overwrite and os.path.exists(result_save_path):
        with open(result_save_path) as f:
            for line in f.readlines():
                data_dict = json.loads(line.strip())
                finish_ids.append(data_dict["id"])

    if len(finish_ids) > 0:
        print(f"Resume from {len(finish_ids)} finished samples")
    else:
        print(f"Start from scratch")

    subset_json = join(f"{PROJECT_ROOT}/data", f"{args.subset}.json")
    print("Reading data from {}".format(subset_json))
    with open(subset_json, "r") as f:
        ds = json.load(f)
        ds = [dict(id=f"{args.subset}_{data_idx}", **data_dict) for data_idx, data_dict in enumerate(ds)]
        ds = [data_dict for data_dict in ds if data_dict["id"] not in finish_ids]

    if len(ds) > 0:
        model_config = build_model_config(model_id=args.model_id, max_model_len=args.max_model_len, max_tokens=args.max_tokens)
        engine_args = model_config.default_engine_args
        if "model" not in engine_args:
            engine_args["model"] = args.model_id
        if args.subset in ["bearing_body", "building_damage_counting", "disaster_type", "road_damage_counting", "caption", "recovery"]:
            engine_args["limit_mm_per_prompt"] = dict(image=2)
        else:
            engine_args["limit_mm_per_prompt"] = dict(image=1)
        if args.tensor_parallel_size is not None:
            engine_args["tensor_parallel_size"] = args.tensor_parallel_size
        engine_args = asdict(EngineArgs(**engine_args))
        print(f"Engine arguments: {engine_args}")
        vlm_model = LLM(**engine_args)

        sampling_params = SamplingParams(max_tokens=args.max_tokens)
        try:
            default_generation_config = GenerationConfig.from_pretrained(args.model_id).to_diff_dict()
            for key in default_generation_config:
                if hasattr(sampling_params, key):
                    setattr(sampling_params, key, default_generation_config[key])
            sampling_params.update_from_generation_config(default_generation_config)
        except OSError:
            for key in engine_args["override_generation_config"]:
                if hasattr(sampling_params, key):
                    setattr(sampling_params, key, engine_args["override_generation_config"][key])

        if args.temperature is not None:
            sampling_params.temperature = args.temperature
        sampling_params.temperature = max(sampling_params.temperature, 0.01)
        print(f"Sampling params: {sampling_params}")

        with open(result_save_path, "w" if len(finish_ids) == 0 else "a") as f:
            progress_bar = tqdm(total=len(ds))
            for batch_inputs, batch_metadata in create_batch_inputs(ds, model_config, args):
                outputs = vlm_model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)

                for idx, output in enumerate(outputs):
                    generated_text = output.outputs[0].text
                    assert int(output.request_id) % args.batch_size == idx, (output.request_id, idx)

                    dump_dict = {
                        "id": batch_metadata[idx]["id"],
                        "response": generated_text,
                    }
                    f.write(json.dumps(dump_dict, ensure_ascii=False) + "\n")
                    f.flush()
                    progress_bar.update(1)

            progress_bar.close()

    with open(result_save_path) as f:
        finished_data = [json.loads(line.strip()) for line in f.readlines()]

    with open(result_save_path.replace(".jsonl", ".json"), "w") as f:
        json.dump(finished_data, f, indent=4)
