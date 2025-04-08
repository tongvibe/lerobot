# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from huggingface_hub import HfApi

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, write_info, write_stats

V20 = "v2.0"
V21 = "v2.1"
CODEBASE_VERSION = V20


def convert_dataset(
    repo_id: str,
    root: str,
    branch: str | None = None,
):
    dataset = LeRobotDataset(repo_id, revision=V21, root=root, force_cache_sync=False)

    if (dataset.root / STATS_PATH).is_file():
        (dataset.root / STATS_PATH).unlink()

    write_stats(dataset.meta.stats, dataset.root)

    dataset.meta.info["codebase_version"] = CODEBASE_VERSION
    write_info(dataset.meta.info, dataset.root)

    # delete episodes_stats.json file
    if (dataset.root / EPISODES_STATS_PATH).is_file:
        (dataset.root / EPISODES_STATS_PATH).unlink()

    dataset.push_to_hub(branch=branch, tag_version=False, allow_patterns="meta/")

    hub_api = HfApi()
    if hub_api.file_exists(repo_id=dataset.repo_id, filename=EPISODES_STATS_PATH, revision=branch, repo_type="dataset"):
        hub_api.delete_file(path_in_repo=EPISODES_STATS_PATH, repo_id=dataset.repo_id, revision=branch, repo_type="dataset")

    hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset "
        "(e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Repo branch to push your dataset. Defaults to the main branch.",
    )

    args = parser.parse_args()
    convert_dataset(**vars(args))
