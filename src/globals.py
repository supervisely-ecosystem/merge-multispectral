import os
import supervisely as sly
from dotenv import load_dotenv
from supervisely.project.project_settings import LabelingInterface

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)

if project_id is None:
    dataset_infos = [api.dataset.get_info_by_id(dataset_id)]
    project_id = dataset_infos[0].project_id
else:
    dataset_infos = api.dataset.get_list(project_id, recursive=True)

project_meta_json = api.project.get_meta(project_id)
project_meta = sly.ProjectMeta.from_json(project_meta_json)

channels_input = os.environ.get("modal.state.channelOrder")
channel_order = list(channels_input.split(","))
if len(channel_order) < 3:
    raise RuntimeError(
        "Incorrect channel order provided, it should be a comma-separated list of channel postfixes"
        " with at least 3 elements, for example: '_1,_2,_3', where '_1', '_2', '_3' are postfixes "
        "of the channels image names, that will be considered as R, G, B channels."
    )

multispectral_tag_meta = project_meta.get_tag_meta(LabelingInterface.MULTISPECTRAL)
if multispectral_tag_meta is None:
    raise RuntimeError(
        "Multispectral tag not found in project meta, this app can work only with Multispectral "
        f"projects. Ensure that {LabelingInterface.MULTISPECTRAL} tag is present in project meta."
    )
