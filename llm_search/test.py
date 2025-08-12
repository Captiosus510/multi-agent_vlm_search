import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R

from sam2.build_sam import build_sam2 # type: ignore
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator # type: ignore

import clip # type: ignore
import cv2

class Vector3D:
    def __init__(self, x: float, y: float, z: float, x_axis: float, y_axis: float, z_axis: float, rotation: float):
        self.x = x
        self.y = y
        self.z = z
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.rotation = rotation

    def __repr__(self):
        return f"Vector3D({self.x}, {self.y}, {self.z}, {self.x_axis}, {self.y_axis}, {self.z_axis}, {self.rotation})"
    


np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


image = Image.open('/home/mahdlinux/ros2_ws/src/llm_search/llm_search/birds_eye_segment.png')
image = np.array(image.convert("RGB"))



plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.axis('off')
plt.show()



sam2_checkpoint = "/home/mahdlinux/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)

masks = mask_generator.generate(image)


print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 