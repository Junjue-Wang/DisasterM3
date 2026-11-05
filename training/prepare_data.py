"""
Build the LLaMA-Factory training set (data/DisasterM3.json) from the DisasterM3 instruct set.

Usage:
    python prepare_data.py --data_root /path/to/DisasterM3_Instruct
"""
import argparse
import json
import os
import sys
from os.path import abspath, dirname, join

from tqdm import tqdm

PACKAGE_DIR = dirname(abspath(__file__))

RELEASE_FILE = "train_release.json"

QA_TASKS = [
    "bearing_body",
    "building_damage_counting",
    "disaster_type",
    "landuse",
    "road_damage_counting",
    "relational_reasoning_qa",
]
REPORT_TASKS = ["caption", "recovery"]

# Task names used in train_release.json.
TASK_NAMES = {
    "Disaster Bearing Bodies Recognition": "bearing_body",
    "Building Damage Counting": "building_damage_counting",
    "Disaster Type Recognition": "disaster_type",
    "Disaster Scene Recognition": "landuse",
    "Road Damage Counting": "road_damage_counting",
    "relational reasoning": "relational_reasoning_qa",
    "disaster caption": "caption",
    "disaster restoration advice": "recovery",
}
# Not trained through LLaMA-Factory.
SKIPPED_TASKS = ["Referring Expression Segmentation"]

# The paper's training data repeats the counting tasks twice (2x oversampling). The paper does not state
# this; set the values to 1 to disable.
TASK_REPEATS = {"building_damage_counting": 2, "road_damage_counting": 2}

# Image pair excluded from the paper's training data.
EXCLUDED_PRE_IMAGE = "socal_fire_00001247_pre_disaster.png"

# The training prompts must stay identical to the paper's training data (including trailing spaces).
prompt_libs = dict(
    double_images="""Analyze both the pre-disaster and post-disaster images to answer the following question.

pre-disaster image:
<image>

post-disaster image:
<image>

Question: {question}""",
    single_image="""<image>
Analyze the image to answer the following question.

Question: {question}""",
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


def load_annotations(data_root):
    """Read train_release.json and group its rows by task, in file order."""
    with open(join(data_root, RELEASE_FILE), "r", encoding="utf-8") as f:
        rows = json.load(f)

    docs_by_task = {task: [] for task in QA_TASKS + REPORT_TASKS}
    for doc in rows:
        if doc["task"] in TASK_NAMES:
            docs_by_task[TASK_NAMES[doc["task"]]].append(doc)
        elif doc["task"] not in SKIPPED_TASKS:
            print(f"Unknown task in {RELEASE_FILE}: {doc['task']!r}", file=sys.stderr)
            sys.exit(1)

    empty_tasks = [task for task, docs in docs_by_task.items() if not docs]
    if empty_tasks:
        print(f"No rows found for: {', '.join(empty_tasks)}", file=sys.stderr)
        sys.exit(1)
    return docs_by_task


def train_image(data_root, rel_path):
    return join(data_root, rel_path.replace("\\", "/"))


def make_sample(prompt, answer, images):
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
        "images": images,
    }


def convert(task, doc, data_root):
    """Convert one annotation into a sharegpt sample, or return None if it is excluded."""
    if task == "landuse":
        prompt = prompt_libs["single_image"].format(question=doc["prompts"])
        return make_sample(prompt, doc["training_answer"], [train_image(data_root, doc["pre_image_path"])])

    if task == "relational_reasoning_qa":
        prompt = prompt_libs["single_image"].format(question=doc["prompts"])
        return make_sample(prompt, doc["ground_truth"], [train_image(data_root, doc["image_path"])])

    if doc["pre_image_path"].endswith(EXCLUDED_PRE_IMAGE):
        return None

    images = [train_image(data_root, doc["pre_image_path"]), train_image(data_root, doc["post_image_path"])]
    if task in REPORT_TASKS:
        return make_sample(prompt_libs[task], doc["ground_truth"], images)
    return make_sample(prompt_libs["double_images"].format(question=doc["prompts"]), doc["training_answer"], images)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True, help=f"directory holding {RELEASE_FILE} and train_images/")
    parser.add_argument("--output", default=join(PACKAGE_DIR, "data", "DisasterM3.json"))
    parser.add_argument("--drop_missing", action="store_true", help="drop samples whose images are missing")
    args = parser.parse_args()

    data_root = abspath(args.data_root)
    if not os.path.isfile(join(data_root, RELEASE_FILE)):
        print(f"{RELEASE_FILE} not found in {data_root}. Check --data_root.", file=sys.stderr)
        sys.exit(1)

    docs_by_task = load_annotations(data_root)
    samples, task_counts = [], {}
    for task in QA_TASKS + REPORT_TASKS:
        docs = docs_by_task[task]
        task_samples = [s for s in (convert(task, doc, data_root) for doc in tqdm(docs, desc=task)) if s is not None]
        repeat = TASK_REPEATS.get(task, 1)
        for _ in range(repeat):
            samples.extend(task_samples)
        task_counts[task] = (len(task_samples), repeat)

    num_before = len(samples)
    image_paths = {p for s in samples for p in s["images"]}
    missing_images = sorted(p for p in tqdm(image_paths, desc="checking images") if not os.path.isfile(p))
    if missing_images:
        print(f"{len(missing_images)} / {len(image_paths)} images not found, e.g.:\n  " + "\n  ".join(missing_images[:10]), file=sys.stderr)
        if not args.drop_missing:
            print("Fix the data layout, or rerun with --drop_missing to skip these samples.", file=sys.stderr)
            sys.exit(1)
        missing_images = set(missing_images)
        samples = [s for s in samples if not any(p in missing_images for p in s["images"])]
        print(f"Dropped {num_before - len(samples)} samples with missing images.", file=sys.stderr)

    os.makedirs(dirname(abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\n{'task':<28}{'samples':>10}{'repeat':>8}{'total':>10}")
    for task, (num, repeat) in task_counts.items():
        print(f"{task:<28}{num:>10}{repeat:>8}{num * repeat:>10}")
    if len(samples) < num_before:
        print(f"\nThese counts are from before dropping {num_before - len(samples)} samples with missing images.")
    print(f"\nSaved {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
