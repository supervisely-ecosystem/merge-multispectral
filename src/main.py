import globals as g
from utils import process_dataset, create_or_sync_project

dst_project, src_dst_ds_id_map = create_or_sync_project(g.workspace_id, g.project_info)
for dataset in g.datasets_with_new_images:
    dst_dataset_id = src_dst_ds_id_map[dataset.id]
    process_dataset(dataset, dst_dataset_id, g.project_meta)

g.api.task.set_output_project(g.task_id, dst_project.id, dst_project.name)
