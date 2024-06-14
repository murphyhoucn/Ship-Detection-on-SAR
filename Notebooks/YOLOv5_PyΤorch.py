#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/JasonManesis/Ship-Detection-on-Remote-Sensing-Synthetic-Aperture-Radar-Data/blob/main/Notebooks/YOLOv5_Py%CE%A4orch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


get_ipython().system('pip install pycocotools')
get_ipython().system('pip install scikit-learn')


# In[3]:


import os
import shutil
import random
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import pandas as pd
import json
import matplotlib.pyplot as plt
import glob


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


# Clone github repo.
get_ipython().system('git clone https://github.com/ultralytics/yolov5.git')

# Change dir to yolov5
get_ipython().run_line_magic('cd', 'yolov5/')

# Install requirements
get_ipython().system('pip install -r requirements.txt')

import torch
from yolov5 import utils
display = utils.notebook_init()  # checks


# In[ ]:


# Run this to fix some issues in Kaggle.
get_ipython().system('add-apt-repository ppa:ubuntu-toolchain-r/test -y')
get_ipython().system('apt-get update')
get_ipython().system('apt-get upgrade libstdc++6 -y')


# In[4]:


# Download pre-trained weights.
get_ipython().system('wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt')


# In[5]:


# Define the dataset.yaml file keys.
import yaml
parsed_yaml_file = dict()
parsed_yaml_file['path'] = ''
parsed_yaml_file['train'] = '/content/drive/MyDrive/Datasets/HRSID_YOLO/train'
parsed_yaml_file['val'] = '/content/drive/MyDrive/Datasets/HRSID_YOLO/val'
parsed_yaml_file['test'] = '/content/drive/MyDrive/Datasets/HRSID_YOLO/test'
parsed_yaml_file['nc'] = 1
parsed_yaml_file['names'] = ['Ship']
parsed_yaml_file


# In[6]:


# Overwrite the original .yaml file.
path_4_saving = '/content/drive/MyDrive/Datasets/HRSID_YOLO/dataset.yaml'
with open(path_4_saving, 'w') as outfile:
    yaml.dump(parsed_yaml_file, outfile, default_flow_style=False)


# In[7]:


# Check that the .yaml file saved correctly.
a_yaml_file = open(path_4_saving)
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
parsed_yaml_file


# # **Train the model with wandb**

# In[ ]:


get_ipython().system('pip install wandb')


# In[ ]:


import wandb
personal_wandb_api_key = ''
wandb.login(key=personal_wandb_api_key)


# In[ ]:


get_ipython().system('python train.py --img 1344 --batch 2 --epochs 20 --data $path_4_saving --weights /content/yolov5/yolov5m.pt')


# # **Analysing results using tensorboard**

# In[ ]:


# Launch after you have started training
# logs save in the folder "runs"
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir runs')


# # **Download weights**

# In[ ]:


# Download model's weights.
get_ipython().system("zip -r /content/exp2.zip './runs/train/exp2'")
from google.colab import files
files.download('/content/exp2.zip')


# # **Evaluation**

# ## YOLOv5m best:

# In[ ]:


get_ipython().system('python val.py --weights /content/drive/MyDrive/Thesis_Jason_Manesis_Scripts/YOLOv5/best.pt --data /content/drive/MyDrive/Datasets/HRSID_YOLO/dataset.yaml --img 1344 --iou 0.5 --task test --save-json')


# # **Inference**

# ## Run inference on images.

# In[9]:


# Define test and weights paths
test_path = '/content/drive/MyDrive/Datasets/HRSID_YOLO/test/images'
weight_path = '/content/drive/MyDrive/Datasets/HRSID_YOLO/last_2.pt'


# In[ ]:


# Inference on all test images.
get_ipython().system('python detect.py --source $test_path --weights $weight_path --img 1344 --save-txt --save-conf --line-thickness 1')


# In[ ]:


# Zip the output results
folder_name = 'inference_on_imgs.zip'
get_ipython().system('zip -r $folder_name /content/yolov5/runs/detect/exp/')

# Download the results
from google.colab import files
files.download(folder_name)


# In[20]:


# Inference on one random test image:
get_ipython().run_line_magic('cd', '/content/yolov5')
f_name = random.choice(os.listdir(test_path))
path_2_image = os.path.join(test_path, f_name)
get_ipython().system('python detect.py --weights $weight_path --img 640 --conf 0.25 --source $path_2_image')


# In[25]:


# Plot the above image with it's predictions:
img = plt.imread(glob.glob(os.path.join('runs/detect/*', f_name))[0])
plt.figure(figsize=(10,10));
plt.imshow(img);


# ## Run inference on video.

# In[ ]:


get_ipython().system('python /content/yolov5/detect.py --source /content/drive/MyDrive/ICEYE_SAR_Video.mp4 --weights /content/drive/MyDrive/Datasets/HRSID_YOLO/last_2.pt --conf 0.5 --imgsz 1300 --line-thickness 1')


# # **Evaluation with COCO API - Οfficial evaluation method**.

# # Correction of the .json file with the ground trouth annotations.
# 
# We must fix the .json file with the ground truth annotations because the generated .json with predictions from YOLOv5 has ID names which -unlike the ID names concerning the ground truth bounding boxes- do not include the file extension (.jpg). As a result the COCO API calculates falsly all the extracted performance metrics. 

# In[26]:


# We load the dataframe which contains all instances.
gtdf = pd.read_pickle('/content/drive/MyDrive/Datasets/HRSID/df_instances.pkl')

# We select from the dataframe which contains all instances only the ones from the test set.
gtdf = gtdf.loc[gtdf['image_for']=='Testing']

# Remove the file extension of all images in the Test set.
gtdf['file_name']=gtdf['file_name'].apply(lambda x: x[:-4])

# Set the above column as ID.
gtdf['image_id'] = gtdf['file_name']

#Show df:
gtdf


# In[27]:


def df_2_COCO_JSON(df, sdir, rotated=False):
    
    '''This function takes a Pandas DataFrame and produces a .json file with 
    annotations in MS COCO format.

    Args:
        df (detectron2.config.config.CfgNode): Input DataFrame.
        sdir (string): Full path for saving the generated .json file.
        rotated (bool): True if bounding boxes are rotated.
    '''

    json_decoded = {}

    # We want the following 6 keys in our .json:
    # 1 | info          | <class 'dict'>   | 6 element/s
    # 2 | licenses      | <class 'list'>   | 1 element/s
    # 3 | images        | <class 'list'>   | (number of images) element/s
    # 4 | type          | <class 'str'>    | 9 element/s
    # 5 | annotations   | <class 'list'>   | (number of instances) element/s
    # 6 | categories    | <class 'list'>   | 1 element/s
    
    # 1 | info   
    json_decoded['info'] = {'contributor': None,
                            'date_created': '2020-02-20 11:26:43.819685',
                            'description': None,
                            'url': None,
                            'version': None,
                            'year': 2020}
    
    # 2 | licenses
    json_decoded['licenses'] = [{'id': 0, 'name': None, 'url': None}]

    # 3 | images
    json_decoded['images'] = [{'date_captured': None,
                            'file_name': df['file_name'].unique()[i],
                            'height': 800,
                            'id': df['image_id'].unique()[i],
                            'license': 0,
                            'url': None,
                            'width': 800} for i in range(len(df['file_name'].unique()))]
    
    # 4 | type 
    json_decoded['type'] = 'instances'
    
    # 5 | annotations
    if rotated:
        json_decoded['annotations'] = [{'area': df['area'].iloc[i],
                                    'bbox': [df['bbox'].iloc[i][0],
                                            df['bbox'].iloc[i][1],
                                            df['bbox'].iloc[i][2],
                                            df['bbox'].iloc[i][3],
                                            df['bbox'].iloc[i][4]],
                                    'category_id': 0,
                                    'id': i,
                                    'image_id': int(df['image_id'].iloc[i]),
                                    'iscrowd': 0} for i in range(len(df))]
    else:
        json_decoded['annotations'] = [{'area': df['area'].iloc[i],
                                    'bbox': [df['bbox'].iloc[i][0],
                                            df['bbox'].iloc[i][1],
                                            df['bbox'].iloc[i][2],
                                            df['bbox'].iloc[i][3]],
                                    'category_id': 0,
                                    'id': i,
                                    'image_id': df['image_id'].iloc[i],
                                    'iscrowd': 0} for i in range(len(df))]

    # 6 | categories                         
    json_decoded['categories'] = [{'id': 0, 'name': 'ship', 'supercategory': None}]
 

    # Save .json file to the target directory.
    json2open = sdir 
    with open(json2open, 'w') as json_file:
        json.dump(json_decoded, json_file)


# In[28]:


df_2_COCO_JSON(gtdf, '/content/drive/MyDrive/Datasets/HRSID_YOLO/annotations/newtrue.json')


# ## Evaluate with COCO API

# In[29]:


annFile = '/content/drive/MyDrive/Datasets/HRSID_YOLO/annotations/newtrue.json'
cocoGt=COCO(annFile)

resFile ='/content/drive/MyDrive/Datasets/HRSID_YOLO/annotations/the_best_predictions.json'
cocoDt=cocoGt.loadRes(resFile)


# In[30]:


# Evaluate with COCO API:
imgIds=sorted(cocoGt.getImgIds())
cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

