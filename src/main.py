import supervisely as sly
import globals as g
from utils import (
    image_groups,
    merge_numpys,
    image_infos_by_channels,
    merge_annotations,
    get_annotations
)

new_dataset = g.api.dataset.create(g.project_id, "merged", change_name_if_conflict=True)
for dataset in g.dataset_infos:
    sly.logger.info(f"Processing dataset {dataset.name}...")
    for group_name, image_infos in image_groups(
        dataset.id, tag_id=g.multispectral_tag_meta.sly_id
    ):
        try:
            channel_image_infos = image_infos_by_channels(image_infos, g.channel_order)
        except Exception as e:
            sly.logger.error(f"Error processing group {group_name}: {e}")
            continue

        anns = get_annotations(dataset.id, channel_image_infos, g.project_meta)
        ann = merge_annotations(anns)

        image_ids = [image_info.id for image_info in channel_image_infos]
        image_nps = g.api.image.download_nps(dataset.id, image_ids)

        try:
            image_np = merge_numpys(image_nps)
        except Exception as e:
            sly.logger.error(f"Error merging images for group {group_name}: {e}")
            continue

        image_info = g.api.image.upload_np(new_dataset.id, f"{group_name}.png", image_np)
        g.api.annotation.upload_ann(image_info.id, ann)
