import os

import yaml
from dotenv import load_dotenv

import supervisely as sly
from supervisely.project.project_settings import LabelingInterface

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

task_id = sly.env.task_id()
team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)

if dataset_id is not None:
    dataset_infos = [api.dataset.get_info_by_id(dataset_id)]
    project_id = dataset_infos[0].project_id
else:
    dataset_infos = api.dataset.get_list(project_id, recursive=True)

project_info = api.project.get_info_by_id(project_id)
project_meta_json = api.project.get_meta(project_id)
project_meta = sly.ProjectMeta.from_json(project_meta_json)

channel_order_yaml = os.environ["modal.state.channelOrder"]  # input yaml string
channel_order_dict = yaml.safe_load(channel_order_yaml)  # dict
channel_order = [
    channel_order_dict[channel] for channel in ["R", "G", "B"]
]  # dict to list

if len(channel_order) < 3:
    raise RuntimeError(
        "Incorrect channel order provided, it should be a YAML string with R, G, B channels, "
        "for example: 'R: _0\\nG: _1\\nB: _2'"
    )

sly.logger.debug(f"Channel order: {channel_order}")

multispectral_tag_meta = project_meta.get_tag_meta(LabelingInterface.MULTISPECTRAL)
if multispectral_tag_meta is None:
    raise RuntimeError(
        "Multispectral tag not found in project meta, this app can work only with Multispectral "
        f"projects. Ensure that {LabelingInterface.MULTISPECTRAL} tag is present in project meta."
    )

datasets_with_new_images = []
