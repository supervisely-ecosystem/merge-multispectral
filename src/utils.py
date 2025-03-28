from collections import defaultdict
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

import globals as g
import supervisely as sly


def process_dataset(
    dataset: sly.DatasetInfo, dst_dataset_id: int, project_meta: sly.ProjectMeta
) -> None:
    """Process dataset and upload merged images to destination dataset.
    Only uploads new images that don't exist in destination dataset.

    :param dataset: Source dataset info.
    :type dataset: sly.DatasetInfo
    :param dst_dataset_id: Destination dataset ID.
    :type dst_dataset_id: int
    :param project_meta: Project meta.
    :type project_meta: sly.ProjectMeta
    """
    sly.logger.info(f"Processing dataset {dataset.name}...")
    all_groups = list(image_groups(dataset.id, tag_id=g.multispectral_tag_meta.sly_id))
    groups_count = len(all_groups)
    
    if groups_count == 0:
        sly.logger.info(f"No image groups found in dataset {dataset.name}")
        return

    progress = sly.Progress(
        message=f"Processing {dataset.name}",
        total_cnt=groups_count,
    )

    dst_images = get_dataset_images(dst_dataset_id)
    dst_names = {img_name[:-4] for img_name, _ in dst_images}
    processed_groups = 0
    skipped_groups = 0
    
    def process_batch(batch_groups):
        nonlocal processed_groups, skipped_groups
        res_image_nps, res_names, res_anns = [], [], []
        for group_name, image_infos in batch_groups:
            if group_name in dst_names:
                sly.logger.info(f"Skipping existing image {group_name}.png")
                progress.iter_done_report()
                skipped_groups += 1
                continue

            try:
                channel_image_infos = image_infos_by_channels(image_infos, g.channel_order)
            except Exception as e:
                sly.logger.error(f"Error processing group {group_name}: {e}")
                progress.iter_done_report()
                skipped_groups += 1
                continue

            anns = get_annotations(dataset.id, channel_image_infos, project_meta)
            ann = merge_annotations(anns)

            image_ids = [image_info.id for image_info in channel_image_infos]
            image_nps = g.api.image.download_nps(dataset.id, image_ids)

            try:
                image_np = merge_numpys(image_nps)
            except Exception as e:
                sly.logger.error(f"Error merging images for group {group_name}: {e}")
                progress.iter_done_report()
                skipped_groups += 1
                continue
                
            res_image_nps.append(image_np)
            res_names.append(f"{group_name}.png")
            res_anns.append(ann)
        
        if res_image_nps:
            batch_processed = len(res_image_nps)
            processed_groups += batch_processed
            res_image_infos = g.api.image.upload_nps(dst_dataset_id, res_names, res_image_nps)
            res_image_ids = [image_info.id for image_info in res_image_infos]
            g.api.annotation.upload_anns(res_image_ids, res_anns)
            progress.iters_done_report(batch_processed)
    
    for batch_groups in sly.batched(all_groups):
        process_batch(batch_groups)
    
    total_processed = processed_groups + skipped_groups
    sly.logger.info(f"Processed {total_processed} of {groups_count} image groups in dataset {dataset.name}")
    sly.logger.info(f"  - Uploaded {processed_groups} new merged images")
    sly.logger.info(f"  - Skipped {skipped_groups} existing images")

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
    Order of the channels is defined by the order of the input list.

    :param numpys: List of numpy arrays to merge.
    :type numpys: List[np.ndarray]
    :return: Merged numpy array in RGB format.
    :rtype: np.ndarray
    """
    processed_arrays = []
    for arr in numpys:
        if len(arr.shape) == 3:
            arr = arr[:, :, 0]
        if len(arr.shape) != 2:
            raise ValueError(f"Expected 2D array after processing, got shape {arr.shape}")
        processed_arrays.append(arr)
    result = np.dstack(processed_arrays)
    if len(result.shape) != 3 or result.shape[2] != 3:
        raise RuntimeError(f"Expected merged image shape (h,w,3), got {result.shape}")
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


def image_infos_by_channels(
    image_infos: List[sly.ImageInfo], channel_order: List[str]
) -> List[sly.ImageInfo]:
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


# Upload Utils
def create_datasets_tree(src_ds_tree: dict, dst_project_id: int) -> dict:
    """Create datasets tree in destination project.

    :param src_ds_tree: Source datasets tree.
    :type src_ds_tree: dict
    :param dst_project_id: Destination project ID.
    :type dst_project_id: int
    :return: Mapping of source dataset IDs to destination dataset IDs.
    :rtype: dict
    """
    src_dst_ds_id_map = {}

    def _create_datasets_tree(src_ds_tree, parent_id=None):
        for src_ds, nested_src_ds_tree in src_ds_tree.items():
            # Get full path to dataset
            path = get_dataset_path(src_ds.id, src_ds.project_id)
            dst_ds_existing = g.api.dataset.get_info_by_name(dst_project_id, src_ds.name, parent_id)
            if dst_ds_existing:
                sly.logger.info(f"Using existing dataset: {path}")
                dst_ds = dst_ds_existing
            else:
                sly.logger.info(f"Creating new dataset: {path}")
                dst_ds = g.api.dataset.get_or_create(
                    project_id=dst_project_id,
                    name=src_ds.name,
                    description=src_ds.description,
                    parent_id=parent_id,
                )
            
            src_dst_ds_id_map[src_ds.id] = dst_ds.id
            _create_datasets_tree(nested_src_ds_tree, parent_id=dst_ds.id)

    _create_datasets_tree(src_ds_tree)
    return src_dst_ds_id_map


def create_project_structure(
    workspace_id: int, project_info: sly.ProjectInfo
) -> Tuple[sly.ProjectInfo, dict]:
    """Create project structure and return project info and dataset mapping.

    :param workspace_id: Workspace ID.
    :type workspace_id: int
    :param project_info: Source project info.
    :type project_info: sly.ProjectInfo
    :return: Tuple of (destination project info, dataset ID mapping).
    :rtype: Tuple[sly.ProjectInfo, dict]
    """
    project_name = f"Merged multispectral {project_info.name}"

    if g.dataset_id is None:
        description = f"Merged multispectral images from {project_info.name} (id: {project_info.id})"
    else:
        description = f"Merged multispectral images from {project_info.name} (id: {project_info.id}) and dataset (id: {g.dataset_id})"

    dst_project = g.api.project.create(
        workspace_id=workspace_id,
        name=project_name,
        description=description,
        change_name_if_conflict=True,
    )

    g.api.project.update_meta(dst_project.id, g.project_meta_json)

    if g.dataset_id is None:
        src_ds_tree = g.api.dataset.get_tree(project_info.id)
        src_dst_ds_id_map = create_datasets_tree(src_ds_tree, dst_project.id)
    else:
        src_dataset = g.dataset_infos[0]
        dst_dataset = g.api.dataset.create(
            project_id=dst_project.id,
            name=src_dataset.name,
            change_name_if_conflict=True,
        )
        src_dst_ds_id_map = {src_dataset.id: dst_dataset.id}
    return src_dst_ds_id_map

def get_project_by_name(workspace_id: int, project_name: str) -> Optional[sly.ProjectInfo]:
    """Get project by name in workspace.

    :param workspace_id: Workspace ID.
    :type workspace_id: int
    :param project_name: Project name to search for.
    :type project_name: str
    :return: Project info if found, None otherwise.
    :rtype: Optional[sly.ProjectInfo]
    """
    project = g.api.project.get_info_by_name(workspace_id, project_name)
    return project


def get_dataset_images(dataset_id: int) -> List[Tuple[str, sly.ImageInfo]]:
    """Get all images in dataset with their names.

    :param dataset_id: Dataset ID.
    :type dataset_id: int
    :return: List of tuples (image_name, image_info).
    :rtype: List[Tuple[str, sly.ImageInfo]]
    """
    images = g.api.image.get_list(dataset_id)
    return [(img.name, img) for img in images]


def get_dataset_path(dataset_id: int, project_id: int) -> str:
    """Get full path to dataset in project hierarchy.

    :param dataset_id: Dataset ID.
    :type dataset_id: int
    :param project_id: Project ID.
    :type project_id: int
    :return: Full path to dataset.
    :rtype: str
    """
    path = []
    current_id = dataset_id
    
    while current_id is not None:
        dataset = g.api.dataset.get_info_by_id(current_id)
        path.append(dataset.name)
        current_id = dataset.parent_id
    
    return "/".join(reversed(path))


def get_merged_image_name(group_name: str) -> str:
    """Get merged image name from group name.

    :param group_name: Group name (base name without channel suffixes).
    :type group_name: str
    :return: Merged image name.
    :rtype: str
    """
    return f"{group_name}.png"


def get_channel_image_names(group_name: str, channel_order: List[str]) -> List[str]:
    """Get channel image names from group name and channel order.

    :param group_name: Group name (base name without channel suffixes).
    :type group_name: str
    :param channel_order: List of channel suffixes.
    :type channel_order: List[str]
    :return: List of channel image names.
    :rtype: List[str]
    """
    return [f"{group_name}{suffix}.png" for suffix in channel_order]


def has_new_images(dataset_id: int, dst_dataset_id: int) -> bool:
    """Check if dataset has new images to process.

    :param dataset_id: Source dataset ID.
    :type dataset_id: int
    :param dst_dataset_id: Destination dataset ID.
    :type dst_dataset_id: int
    :return: True if dataset has new images to process, False otherwise.
    :rtype: bool
    """
    src_path = get_dataset_path(dataset_id, g.project_info.id)
    src_images = get_dataset_images(dataset_id)
    dst_images = get_dataset_images(dst_dataset_id)
    src_groups = set()
    for img_name, _ in src_images:
        for suffix in g.channel_order:
            if img_name.endswith(suffix + ".png"):
                base_name = img_name[:-len(suffix + ".png")]
                src_groups.add(base_name)
                break
    
    dst_names = {img_name[:-4] for img_name, _ in dst_images}
    groups_to_upload = src_groups - dst_names
    if not groups_to_upload:
        sly.logger.info(f"No new images to upload in dataset {src_path}")
        return False
        
    sly.logger.info(f"Found {len(groups_to_upload)} new image groups in dataset {src_path}")
    return True


def create_or_sync_project(
    workspace_id: int,
    project_info: sly.ProjectInfo
) -> dict:
    """Create new project or sync with existing one.

    :param workspace_id: Workspace ID.
    :type workspace_id: int
    :param project_info: Source project info.
    :type project_info: sly.ProjectInfo
    :return: Mapping of source dataset IDs to destination dataset IDs.
    :rtype: dict
    """
    project_name = f"Merged multispectral {project_info.name}"
    dst_project = get_project_by_name(workspace_id, project_name)
    
    if dst_project is None:
        sly.logger.info(f"Creating new project: {project_name}")
        src_dst_ds_id_map = create_project_structure(workspace_id, project_info)
        g.datasets_with_new_images = g.dataset_infos.copy()
        return src_dst_ds_id_map
    else:
        sly.logger.info(f"Found existing project: {project_name}")
        if g.dataset_id is None:
            src_ds_tree = g.api.dataset.get_tree(project_info.id)
            src_dst_ds_id_map = create_datasets_tree(src_ds_tree, dst_project.id)
        else:
            src_dataset = g.dataset_infos[0]
            dst_dataset = g.api.dataset.create(
                project_id=dst_project.id,
                name=src_dataset.name,
                change_name_if_conflict=True,
            )
            src_dst_ds_id_map = {src_dataset.id: dst_dataset.id}
        for dataset in g.dataset_infos:
            dst_dataset_id = src_dst_ds_id_map[dataset.id]
            if has_new_images(dataset.id, dst_dataset_id):
                g.datasets_with_new_images.append(dataset)
        
        if not g.datasets_with_new_images:
            sly.logger.info("No new images to process in any dataset")
        else:
            sly.logger.info(f"Found {len(g.datasets_with_new_images)} datasets with new images")
            
        return dst_project, src_dst_ds_id_map

# Test Utils
def is_single_channel(image: np.ndarray) -> bool:
    """Check if image is single channel.

    :param image: Image array to check.
    :type image: np.ndarray
    :return: True if image is single channel, False otherwise.
    :rtype: bool
    """
    return len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)


def divide_into_channels(img: np.ndarray, ensure_rgb_order: bool = True) -> List[np.ndarray]:
    """Divide image into channels.

    :param img: Image array to divide.
    :type img: np.ndarray
    :param ensure_rgb_order: If True, ensures channels are returned in RGB order. Default is True.
    :type ensure_rgb_order: bool
    :return: List of channels in order [R,G,B] if ensure_rgb_order is True, otherwise in order they appear in image.
    :rtype: List[np.ndarray]
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected image with shape (h,w,3), got {img.shape}")
    if ensure_rgb_order:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.dtype == np.uint8 else img
        return [rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]]
    else:
        channels = []
        for i in range(img.shape[2]):
            channels.append(img[:, :, i])
        return channels