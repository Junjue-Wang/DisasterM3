"""
Convert the DisasterM3 benchmark set into LLaMA-Factory format (data/bench/<subset>.json) for
LLaMA-Factory's scripts/vllm_infer.py. Only needed when the official run_vllm.py cannot run a model.

Prompts follow the official pyscripts/run_vllm.py (InternVL formatting), and every sample keeps the
official id "<subset>_<index>".

Usage:
    python prepare_bench.py --bench_root /path/to/benchmark/data
"""
import argparse
import json
import os
import sys
from os.path import abspath, dirname, join

from prepare_data import prompt_libs as train_prompt_libs

PACKAGE_DIR = dirname(abspath(__file__))

SUBSETS = [
    "bearing_body",
    "building_damage_counting",
    "disaster_type",
    "road_damage_counting",
    "landuse",
    "relational_reasoning_qa",
    "caption",
    "recovery",
]

# Copied from the official pyscripts/run_vllm.py. Caption and recovery prompts are identical to the training ones.
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
    caption=train_prompt_libs["caption"],
    recovery=train_prompt_libs["recovery"],
)


def bench_image(bench_root, rel_path):
    return join(bench_root, "images", rel_path.replace("\\", "/"))


def build_prompt(subset, doc, bench_root):
    """Return (prompt_text, image_paths) as built by the official get_messages_from_data()."""
    if subset in ["bearing_body", "building_damage_counting", "disaster_type", "road_damage_counting"]:
        prompt_key = subset if subset == "bearing_body" else "single_choice"
        text = prompt_libs[prompt_key].format(prompt=doc["prompts"], options_str=doc["options_str"])
        images = [bench_image(bench_root, doc["pre_image_path"]), bench_image(bench_root, doc["post_image_path"])]
    elif subset == "landuse":
        text = prompt_libs[subset].format(prompt=doc["prompts"], options_str=doc["options_str"])
        images = [bench_image(bench_root, doc["pre_image_path"])]
    elif subset == "relational_reasoning_qa":
        text = prompt_libs[subset].format(prompt=doc["prompts"], options_str=doc["option_str"])
        images = [join(bench_root, doc["image_path"].replace("\\", "/"))]
    else:
        text = prompt_libs[subset]
        images = [bench_image(bench_root, doc["pre_image_path"]), bench_image(bench_root, doc["post_image_path"])]

    # The official InternVL path renders every image as "<image>\n"; single-image prompts put it first.
    if "<image>" in text:
        text = text.replace("<image>", "<image>\n")
    else:
        text = "<image>\n" + text
    return text, images


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench_root", required=True, help="directory holding <subset>.json and images/")
    parser.add_argument("--output_dir", default=join(PACKAGE_DIR, "data", "bench"))
    args = parser.parse_args()

    bench_root = abspath(args.bench_root)
    subsets = [s for s in SUBSETS if os.path.isfile(join(bench_root, f"{s}.json"))]
    if not subsets:
        print(f"No <subset>.json found in {bench_root}. Expected some of: {', '.join(SUBSETS)}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    for subset in subsets:
        with open(join(bench_root, f"{subset}.json"), "r") as f:
            docs = json.load(f)

        samples = []
        for idx, doc in enumerate(docs):
            text, images = build_prompt(subset, doc, bench_root)
            label = doc.get("ground_truth", "")
            samples.append(
                {
                    "id": f"{subset}_{idx}",
                    "conversations": [
                        {"from": "human", "value": text},
                        {"from": "gpt", "value": label if isinstance(label, str) else json.dumps(label)},
                    ],
                    "images": images,
                }
            )

        missing = sorted({p for s in samples for p in s["images"] if not os.path.isfile(p)})
        if missing:
            print(f"[{subset}] {len(missing)} images not found, e.g.:\n  " + "\n  ".join(missing[:10]), file=sys.stderr)
            sys.exit(1)

        with open(join(args.output_dir, f"{subset}.json"), "w") as f:
            json.dump(samples, f, indent=2)
        print(f"{subset:<28}{len(samples):>8} samples -> dataset disaster_m3_bench_{subset}")

    skipped = [s for s in SUBSETS if s not in subsets]
    if skipped:
        print(f"Skipped (no json found): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
