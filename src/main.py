import os
from collections import defaultdict
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import supervisely as sly
from dotenv import load_dotenv
from supervisely.project.project_settings import LabelingInterface

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely-dev.env"))
    load_dotenv("local.env")


team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)

channels_input = os.environ.get("modal.state.channelOrder")
channel_order = list(channels_input.split(","))

sly.logger.debug(f"Channel order: {channel_order}")

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


def merge_numpys(numpys: List[np.ndarray]) -> np.ndarray:
    """Merge list of numpy arrays into one numpy array.
    Order of the channels ins defined by the order of the input list.

    :param numpys: List of numpy arrays to merge.
    :type numpys: List[np.ndarray]
    :return: Merged numpy array.
    :rtype: np.ndarray
    """
    return np.stack(numpys, axis=-1)


def image_infos_by_channels(image_infos: List[sly.ImageInfo]) -> List[sly.ImageInfo]:
    channel_image_infos = {}
    for idx, channel_postfix in enumerate(channel_order):
        image_info = get_needed_image(image_infos, channel_postfix)
        if image_info is None:
            raise RuntimeError(f"Image with postfix {channel_postfix} not found.")
        channel_image_infos[idx] = image_info
    return list(channel_image_infos.values())


def get_needed_image(
    image_info: sly.ImageInfo, channel_postfix: str
) -> Optional[sly.ImageInfo]:
    """Get image info with the given postfix from the list of image infos.

    :param image_info: Image info to search for the image with the given postfix.
    :type image_info: sly.ImageInfo
    :param channel_postfix: Postfix of the channel to search for.
    :type channel_postfix: str

    :return: Image info with the given postfix.
    :rtype: Optional[sly.ImageInfo]
    """
    for image_info in image_infos:
        no_ext_name = sly.fs.get_file_name(image_info.name)
        if no_ext_name.endswith(channel_postfix):
            return image_info
    return None


def merge_annotations(anns: List[sly.Annotation]) -> sly.Annotation:
    """Merge list of annotations into one annotation.

    :param anns: List of annotations to merge.
    :type anns: List[sly.Annotation]
    :return: Merged annotation.
    :rtype: sly.Annotation
    """
    merged_ann = anns[0]
    for ann in anns[1:]:
        merged_ann = merged_ann.merge(ann)
    return merged_ann


def get_annotations(
    dataset_id: int, image_infos: List[sly.ImageInfo], project_meta: sly.ProjectMeta
) -> List[sly.Annotation]:
    """Get annotations for the given image info.

    :param image_infos: Image infos to get annotations for.
    :type image_infos: List[sly.ImageInfo]
    :return: List of annotations for the given image info.
    :rtype: List[sly.Annotation]
    """
    image_ids = [image_info.id for image_info in image_infos]
    anns_json = api.annotation.download_json_batch(dataset_id, image_ids)
    return [sly.Annotation.from_json(ann_json, project_meta) for ann_json in anns_json]


if not dataset_id:
    datasets = api.dataset.get_list(project_id)
else:
    datasets = [api.dataset.get_info_by_id(dataset_id)]

new_dataset = api.dataset.get_or_create(project_id, "merged")

sly.logger.info(f"Working with {len(datasets)} datasets.")
for dataset in datasets:
    sly.logger.info(f"Processing dataset {dataset.name}...")
    for group_name, image_infos in image_groups(
        dataset.id, tag_id=multispectral_tag_meta.sly_id
    ):
        sly.logger.info(
            f"Processing group {group_name} with {len(image_infos)} images."
        )

        try:
            channel_image_infos = image_infos_by_channels(image_infos)
            sly.logger.info(f"Found {len(channel_image_infos)} channel images.")
        except Exception as e:
            sly.logger.error(f"Error processing group {group_name}: {e}")
            continue

        ann = merge_annotations(
            get_annotations(dataset.id, channel_image_infos, project_meta)
        )
        sly.logger.info(f"Merged annotations with {len(ann.labels)} labels.")
        image_nps = api.image.download_nps(
            dataset.id, [image_info.id for image_info in channel_image_infos]
        )

        # ! DEBUG! Remove this section, because in test data all images were 3-channeled.
        single_channeled = []
        for old in image_nps:
            print(f"Shape: {old.shape}")
            # ! DEBUG!
            # Take only first channel from the image
            single_channel = old[:, :, 0]
            single_channeled.append(single_channel)
        image_nps = single_channeled
        # ! End of debug
        sly.logger.info(f"Downloaded {len(image_nps)} images.")
        image_np = merge_numpys(image_nps)
        sly.logger.info(f"Merged images with shape {image_np.shape}.")

        image_info = api.image.upload_np(new_dataset.id, f"{group_name}.png", image_np)
        api.annotation.upload_ann(image_info.id, ann)
        sly.logger.info(
            f"Uploaded image {image_info.name} with {len(ann.labels)} labels."
        )
