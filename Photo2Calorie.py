# coding: utf-8

# In[1]:


# %load mcnn_with_coco.py
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import pycocotools.mask as mask
import numpy

# Root directory of the project
ROOT_DIR = os.path.abspath(".\\")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples\\coco"))  # To find local version
import coco  # 왜 빨간줄 뜨는지 모르겠음. 실제로 import는 되는듯

# Import coco API mask.py
import pycocotools.mask


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
# config.display() 설정 출력

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Load COCO dataset
"""
dataset = coco.CocoDataset()
dataset.load_coco("C:/Program Files/Python36/Lib/site-packages/pycocotools/coco", "train")
dataset.prepare()
"""

# In[30]:
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'coin', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Print class names


# In[265]:


# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
print("파일 이름:",file_names)
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# In[266]:


# Run detection
results = model.detect([image], verbose=1)

# In[267]:


print("-----------------visualization------------------")
# Visualize results
r = results[0]

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                           class_names, r['scores'])
"""
results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
"""

# In[270]:


class Fruit:
    def __init__(self):
        # 바나나,사과,오렌지
        self.name = ['banana', 'apple', 'orange']
        self.class_ids = [47, 48, 50, 55]
        self.mean_mass = [175.18, 244.063, 282.79]
        self.mean_volume = [190, 310, 270]
        # 평균 fruit 크기: 평균 동전 크기 = 1:mean_ratio
        self.mean_ratio = [15.033, 13.9, 13.88]
        #self.mean_ratio = [14.0911, 13.444, 13.88]
        self.num = [0, 0, 0]
        self.mass = [0, 0, 0]
        self.volume = [0, 0, 0]
        self.coin_pixel = 0
        self.kcal = [0, 0, 0]
        self.kcal_per_g = [0.89, 0.52, 0.47]
        self.density = [0.922,0.7873,0.95476]

    def is_fruit(self, class_id):
        if class_id in self.class_ids:
            if class_id == 55:
                class_id = 48
            return "True", self.class_ids.index(class_id)
        return "False", -1

    def calculate_volume2mass2kcal(self, fruit_pixel, fruit_index):
        mass = 0
        volume = 0
        ratio = fruit_pixel / self.coin_pixel
        relative_size = ratio / self.mean_ratio[fruit_index]
        # print("평균 질량에 비해",pow(relative_size,3/2),"배 ")
        volume += self.mean_volume[fruit_index] * pow(relative_size, (3 / 2))
        self.volume[fruit_index] += volume
        mass = volume*self.density[fruit_index]
        self.mass[fruit_index] += mass
        self.kcal[fruit_index] += mass * self.kcal_per_g[fruit_index]
        # print(self.kcal[fruit_index])

    def print_result(self):
        total_calory = 0
        for index in range(len(self.name)):
            if self.num[index] == 0:  # detect 되지 않았으면 pass
                continue
            print("-------------------------------------------------")
            print("과일이름: ", self.name[index])
            print("개수: ", self.num[index])
            print("volume 합: ", self.volume[index])
            print("mass 합: ", self.mass[index])
            print("칼로리 합: ", self.kcal[index])
            total_calory += self.kcal[index]

        print("---------------------------------------------------")
        print("총 칼로리: ", total_calory)


# In[272]:


height = image.shape[0]
width = image.shape[1]

fruit = Fruit()
# 미리 coin mask 크기 구하기
coin_index = numpy.where(r['class_ids'] == 75)
for i in range(height):
    for j in range(width):
        if r['masks'][i][j][coin_index]:
            fruit.coin_pixel += 1

print("전체 그림 크기: ", height * width, "\n")

# class별 mask 크기 구하기
for k in range(len(r['class_ids'])):
    pixel_cnt = 0
    for i in range(height):
        for j in range(width):
            if r['masks'][i][j][k]:  # 해당 클래스에 대한 mask pixel 존재할 경우
                pixel_cnt += 1

    is_fruit, fruit_index = fruit.is_fruit(r['class_ids'][k])
    if is_fruit == "True":
        fruit.num[fruit_index] += 1  # 해당 과일 mask 개수
        fruit.calculate_volume2mass2kcal(pixel_cnt, fruit_index)

    print(class_names[r['class_ids'][k]], "의 mask 크기: ", pixel_cnt)

# 결과 출력
fruit.print_result()

