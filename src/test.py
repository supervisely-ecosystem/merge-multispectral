import supervisely as sly
import globals as g
import numpy as np
import cv2
from utils import (
    is_single_channel,
    divide_into_channels,
)

# Test with image from Supervisely
remote_img = g.api.image.download_np(133863) # 
is_single = is_single_channel(remote_img)
print(f"Is input image single channel: {is_single}")
print(f"Remote image shape: {remote_img.shape}")

# Test Local Image
img_path = "single_c.png"

# CV2 Read
# cv2_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
# is_single_cv2 = is_single_channel(cv2_img)
# print(f"Is OpenCV image single channel: {is_single_cv2}")
# print(f"Original image shape (OpenCV): {cv2_img.shape}")

# Supervisely Read
# local_img = np.array(sly.image.read(img_path))
# is_single = is_single_channel(local_img)
# print(f"Is local image single channel: {is_single}")
# print(f"Local image shape (Supervisely): {local_img.shape}")

if not is_single:
    new_dataset = g.api.dataset.create(g.project_id, "test_channels", change_name_if_conflict=True)
    channels = divide_into_channels(remote_img)
    
    image_infos = g.api.image.upload_multispectral(new_dataset.id, "test.png", channels)
    image_ids = [image_info.id for image_info in image_infos]
    image_nps = g.api.image.download_nps(new_dataset.id, image_ids)
    for image_np in image_nps:
        is_single = is_single_channel(image_np)
        print(f"Is output image single channel: {is_single}")
