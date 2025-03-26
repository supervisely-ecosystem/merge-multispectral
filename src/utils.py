from collections import defaultdict
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
import supervisely as sly
import globals as g


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
    image_infos = g.api.image.get_list(dataset_id)
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


def image_infos_by_channels(image_infos: List[sly.ImageInfo], channel_order: List[str]) -> List[sly.ImageInfo]:
    channel_image_infos = {}
    for idx, channel_postfix in enumerate(channel_order):
        image_info = get_needed_image(image_infos, channel_postfix)
        if image_info is None:
            raise RuntimeError(f"Image with postfix {channel_postfix} not found.")
        channel_image_infos[idx] = image_info
    return list(channel_image_infos.values())


def get_needed_image(
    image_infos: List[sly.ImageInfo], channel_postfix: str
) -> Optional[sly.ImageInfo]:
    """Get image info with the given postfix from the list of image infos.

    :param image_infos: List of image infos to search for the image with the given postfix.
    :type image_infos: List[sly.ImageInfo]
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
    
    # Remove duplicate image tags
    unique_img_tags = []
    seen_tag_names = set()
    for tag in merged_ann.img_tags:
        if tag.name not in seen_tag_names:
            seen_tag_names.add(tag.name)
            unique_img_tags.append(tag)
    return merged_ann.clone(img_tags=unique_img_tags)


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
    anns_json = g.api.annotation.download_json_batch(dataset_id, image_ids)
    return [sly.Annotation.from_json(ann_json, project_meta) for ann_json in anns_json] 