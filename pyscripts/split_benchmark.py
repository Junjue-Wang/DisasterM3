"""
Split benchmark_release.json into the <subset>.json files read by pyscripts/run_vllm.py.

Rows keep the release order, so the ids "<subset>_<index>" stay stable. All fields are kept and only
the image paths are rewritten: pre_image_path and post_image_path become relative
to <output_dir>/images, and image_path becomes relative to <output_dir>. <output_dir>/images is a
symlink to test_images.

Usage:
    python pyscripts/split_benchmark.py --bench_root /path/to/DisasterM3_Bench
"""
import argparse
import json
import os
import shlex
import sys
from os.path import abspath, dirname, isdir, isfile, join, realpath

# The same folder run_vllm.py reads: "data" next to the repository folder.
PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))

RELEASE_FILE = "benchmark_release.json"
IMAGE_DIR = "test_images"
IMAGE_LINK = "images"

# Task names used in benchmark_release.json, in the subset order of run_vllm.py.
TASK_NAMES = {
    "Disaster Bearing Bodies Recognition": "bearing_body",
    "Building Damage Counting": "building_damage_counting",
    "Disaster Type Recognition": "disaster_type",
    "Road Damage Counting": "road_damage_counting",
    "Disaster Scene Recognition": "landuse",
    "Relational Reasoning": "relational_reasoning_qa",
    "Disaster Report": "caption",
    "Disaster Restoration Advice": "recovery",
}
SKIPPED_TASKS = ["Referring Expression Segmentation"]


def path_in_image_dir(path):
    """Return a release path such as test_images\\x.png relative to test_images."""
    path = path.replace("\\", "/")
    if not path.startswith(IMAGE_DIR + "/"):
        raise ValueError(f"Expected a path under {IMAGE_DIR}/, got {path}")
    return path[len(IMAGE_DIR) + 1:]


def rewrite_paths(doc):
    doc = dict(doc)
    for key in ("pre_image_path", "post_image_path"):
        if key in doc:
            doc[key] = path_in_image_dir(doc[key])
    if "image_path" in doc:
        doc["image_path"] = f"{IMAGE_LINK}/{path_in_image_dir(doc['image_path'])}"
    return doc


def split_by_subset(rows):
    """Group rows by subset in release order. Exits on an unknown task or an empty subset."""
    docs_by_subset = {subset: [] for subset in TASK_NAMES.values()}
    num_skipped = {task: 0 for task in SKIPPED_TASKS}
    for row in rows:
        if row["task"] in TASK_NAMES:
            docs_by_subset[TASK_NAMES[row["task"]]].append(rewrite_paths(row))
        elif row["task"] in num_skipped:
            num_skipped[row["task"]] += 1
        else:
            print(f"Unknown task in {RELEASE_FILE}: {row['task']!r}", file=sys.stderr)
            sys.exit(1)

    empty_subsets = [subset for subset, docs in docs_by_subset.items() if not docs]
    if empty_subsets:
        print(f"No rows found for: {', '.join(empty_subsets)}", file=sys.stderr)
        sys.exit(1)
    return docs_by_subset, num_skipped


def find_missing_images(docs_by_subset, image_dir):
    """Return the images referenced by the rows that are not in image_dir."""
    paths = set()
    for docs in docs_by_subset.values():
        for doc in docs:
            paths.update(doc[key] for key in ("pre_image_path", "post_image_path") if key in doc)
            if "image_path" in doc:
                paths.add(doc["image_path"][len(IMAGE_LINK) + 1:])
    return sorted(path for path in paths if not isfile(join(image_dir, path)))


def link_images(image_dir, link):
    """Point link at image_dir. An existing correct link is kept."""
    if realpath(link) == realpath(image_dir):
        print(f"Keeping {link} -> {image_dir}")
        return
    if os.path.lexists(link) and not os.path.islink(link):
        print(f"{link} exists and is not a link. Move it away and rerun.", file=sys.stderr)
        sys.exit(1)
    try:
        os.symlink(image_dir, link, target_is_directory=True)
    except OSError as error:
        print(f"Could not create {link}: {error}", file=sys.stderr)
        print(f"Create it manually, then rerun:\n  ln -sfn {shlex.quote(image_dir)} {shlex.quote(link)}", file=sys.stderr)
        sys.exit(1)
    print(f"Linked {link} -> {image_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench_root", required=True, help=f"directory holding {RELEASE_FILE} and {IMAGE_DIR}/")
    parser.add_argument("--output_dir", default=join(PROJECT_ROOT, "data"), help="directory read by run_vllm.py")
    args = parser.parse_args()

    bench_root, output_dir = abspath(args.bench_root), abspath(args.output_dir)
    release_path, image_dir = join(bench_root, RELEASE_FILE), join(bench_root, IMAGE_DIR)
    if not isfile(release_path) or not isdir(image_dir):
        print(f"{RELEASE_FILE} and {IMAGE_DIR}/ must both be in {bench_root}. Check --bench_root.", file=sys.stderr)
        sys.exit(1)

    with open(release_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    docs_by_subset, num_skipped = split_by_subset(rows)

    missing_images = find_missing_images(docs_by_subset, image_dir)
    if missing_images:
        print(f"{len(missing_images)} images not found in {image_dir}, e.g.:\n  " + "\n  ".join(missing_images[:10]), file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    link_images(image_dir, join(output_dir, IMAGE_LINK))
    for subset, docs in docs_by_subset.items():
        with open(join(output_dir, f"{subset}.json"), "w", encoding="utf-8") as f:
            json.dump(docs, f)
        print(f"{subset:<28}{len(docs):>8} rows")
    for task, num in num_skipped.items():
        print(f"Skipped {num} rows of {task}, which run_vllm.py does not support")

    print(f"Saved {sum(len(docs) for docs in docs_by_subset.values())} rows to {output_dir}")


if __name__ == "__main__":
    main()
