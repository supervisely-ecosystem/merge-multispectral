import globals as g
from utils import process_dataset, create_project_structure

src_dst_ds_id_map = create_project_structure(g.workspace_id, g.project_info)
for dataset in g.dataset_infos:
    dst_dataset_id = src_dst_ds_id_map[dataset.id]
    process_dataset(dataset, dst_dataset_id, g.project_meta)
