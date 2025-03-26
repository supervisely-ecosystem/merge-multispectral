import os
from collections import defaultdict
from typing import Generator, List, Tuple, Union

import supervisely as sly
from dotenv import load_dotenv
from supervisely.project.project_settings import LabelingInterface

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely-dev.env"))
    load_dotenv("local.env")

channel_order = os.environ.get("modal.state.channelOrder")
print(channel_order)

team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)

api: sly.Api = sly.Api.from_env()

sly.logger.debug(
    f"API instance created for team_id={team_id}, workspace_id={workspace_id}, "
    f"project_id={project_id}, dataset_id={dataset_id}"
)

if project_id is None:
    sly.logger.debug(
        "App is launched from context of dataset, retrieving project_id..."
    )
    dataset_info = api.dataset.get_info_by_id(dataset_id)
    project_id = dataset_info.project_id

project_meta_json = api.project.get_meta(project_id)
project_meta = sly.ProjectMeta.from_json(project_meta_json)

multispectral_tag_meta = project_meta.get_tag_meta(LabelingInterface.MULTISPECTRAL)
if multispectral_tag_meta is None:
    raise RuntimeError(
        "Multispectral tag not found in project meta, this app can work only with Multispectral "
        f"projects. Ensure that {LabelingInterface.MULTISPECTRAL} tag is present in project meta."
    )

sly.logger.info(f"Found multispectral tag with ID={multispectral_tag_meta.sly_id}.")


def image_groups(
    dataset_id: int, tag_id: int
) -> Generator[Tuple[Union[str, int], List[sly.ImageInfo]], None, None]:
    """Generator that yields tuples of group name and list of image infos in all image groups
    in the given dataset.

    :param dataset_id: ID of the dataset to search for image groups.
    :type dataset_id: int
    :param tag_id: ID of the tag to use for grouping images.
    :type tag_id: int

    :return: Generator that yields tuples of group name and list of image infos in all image groups.
    :rtype: Generator[Tuple[Union[str, int], List[sly.ImageInfo]], None, None]

    """
    image_infos = api.image.get_list(dataset_id)
    image_groups = defaultdict(list)

    for image_info in image_infos:
        for tag_json in image_info.tags:
            if tag_json.get("tagId") == tag_id:
                group_name = tag_json.get("value")
                if group_name is not None:
                    image_groups[group_name].append(image_info)

    for group_name, image_infos in image_groups.items():
        yield group_name, image_infos


if not dataset_id:
    datasets = api.dataset.get_list(project_id)
else:
    datasets = [api.dataset.get_info_by_id(dataset_id)]
    
sly.logger.info(f"Working with {len(datasets)} datasets.")
for dataset in datasets:
    sly.logger.info(f"Processing dataset {dataset.name}...")
    for group_name, image_infos in image_groups(dataset.id, tag_id=multispectral_tag_meta.sly_id):
        sly.logger.info(
            f"Processing group {group_name} with {len(image_infos)} images."
        )
