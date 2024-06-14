# Ship Detection on Remote Sensing Synthetic Aperture Radar Data.

<div align="justify">
  
The present project was conducted as part of my diploma thesis which focuses on the investigation of methods for the effective detection of ships in synthetic aperture radar satellite imagery utilizing deep learning techniques. These methods use the Faster-RCNN and YOLOv5 network architectures to create three different detectors. More specifically, the first two models created are based on the Faster-RCNN network architecture and utilize a set of normal and rotated bounding boxes for the detection process. The one-stage detection network is based on the architecture of the YOLOv5 model and uses regular bounding boxes to delimit the estimated targets. The produced models are trained and evaluated on the HRSID dataset. The greatest accuracy is found in models that use regular bounding boxes to derive estimates. While, the model with rotated bounding boxes, shows the largest localization errors and is characterized by an increased number of false negative detections.
  
</div align="justify">

## HRSID Properties.
* The High-Resolution SAR Images Dataset contains 116 co-polarized and 20 cross-polarized SAR imageries.
* The original imageries for constructing HRSID are 99 Sentinel-1B imageries, 36 TerraSAR-X and 1 TanDEM-X images.
* The above 136 panoramic SAR imageries cropped to 5604 high-resolution SAR images.
* These 5604 images have dimensions of 800 × 800 pixels, resolution of 96 dpi, and there are in .jpeg format.
* The colour depth of the images is 8 bits (one channel). 
* The extracted 5604 high-resolution SAR images contain 16951 ship instances.
* The spatial resolutions of SAR images are 0.5, 1 and 3 meters per pixel.
* The annotations of each instance are the corresponding bounding box and the ship’s outline. 
* The annotations of each SAR image constitute a .json file in MS COCO dataset format.
* Paper Link: https://ieeexplore.ieee.org/abstract/document/9127939
* Dataset Link: https://github.com/chaozhong2010/HRSID 

## Proposed architectures of Faster-RCNN. 
<div align="justify">

Faster-RCNN is a two stage detection architecture and contains 3 different submodules: a) Backbone Network, b) Region Proposal Network and c) Fast-RCNN. At the proposed model, Feature Pyramid Network with ResNet backbone was used for the creation of **P2-P6** spatial levels. Region Proposal Network receives serially the **P2-P6** feature maps and for every **Pi** level creates a hidden representation, which is shared between the regression and classification layers, and produces two output tensors with predicted objectness logits and anchor deltas for every anchor in the **Pi**. Next, predicted anchor deltas are applied to the corresponding anchors and the above boxes are sorted by the predicted objectness scores at each **Pi** level. Then, after the application of a confidence threshold and the NMS algorithm, RPN retains a subset of the anchor boxes from which **k** ROIs were extracted. Finally, ROI (Box) Head takes the outputs from the FPN and RPN networks, which are the multiscale feature maps and the ROIs respectively, and uses the latter to crop the regions of interest from the feature maps. The cropped regions are then pooled (transformed into the same dimensions) and fed as flattened feature vectors into a pair of fully connected layers that extract the class probabilities and the corresponding coordinates for a predefined number of boxes. 
  
</div align="justify">


![1_unZ995FzCFMCgrQ0l1R5mw](https://user-images.githubusercontent.com/74200033/159125727-d9468867-160a-4f52-8c45-41077360f7d8.png)

*Image source: https://medium.com/@hirotoschwert/digging-into-detectron-2-part-4-3d1436f91266*

## Proposed architecture of YOLOv5.

<div align="justify">
  
YOLOv5 is a one shot detector which contains 2 different networks: a) Feature Extraction Network (Backbone Network) and b) PANet. Backbone network is used for feature extraction and It uses the main modules of **C3** (VGP+FLOPS↓) and **SPPF** (multiscale feature fusion). The PANet network creates a set of feature maps in 3 different spatial scales (**P3-P5**) which have 3 different anchors at every spatial location. The above tensors (**P3-P5**) are then fed into the corresponding layer of the “Head” network and after the application of a confidence threshold and the NMS algorithm the final bounding box predictions (**class_id, x1, y1, x2, y2, confidence_score**) were extracted.
  
</div align="justify">

![YOLOV5](https://user-images.githubusercontent.com/74200033/159246431-be1231fb-af0c-474f-8dde-fc95a1c7b264.png)



## Quantitative Evaluation

<div align="left"> 
  
 ***Mean Average Precision***
  
|          **Metric**        | **Faster - RCNΝ (Normal Bboxes)**     |**Faster - RCNΝ (Rotated Bboxes)** | **YOLOv5** |  **STANet<sup>1</sup>**  |  **DB-YOLO<sup>2</sup>**  |
|:--------------------------:|:-------------------------------------:|:---------------------------------:|:----------:|:------------:|:-------------:|
|AP<sup>0.50:.05:.95</sup> | *68.1*|*42.9*|*71.1*|*69.5*|***72.0***|
|AP<sup>0.50</sup> |*91.4*|*75.3*|***94.2***|*92.4*|***94.4***|
|AP<sup>0.75</sup> |*79.3*|*45.5*|***82.0***|*81.1*|-|
|AP<sup>small</sup> |*69.3*|*41.3*|*62.9*|***70.9***|-|
|AP<sup>medium</sup> |*68.5*|*51.1*|***80.7***|*68.6*|-|
|AP<sup>large</sup> |*44.1*|*20.9*|***55.1***|*37.8*|-|

 ***Mean Average Recall***  
|          **Metric**        | **Faster - RCNΝ (Normal Bboxes)**     |**Faster - RCNΝ (Rotated Bboxes)** | **YOLOv5** |  **STANet<sup>1</sup>**  |  **DB-YOLO<sup>2</sup>**  |
|:--------------------------:|:-------------------------------------:|:---------------------------------:|:----------:|:------------:|:-------------:|
|AR<sup>max=1</sup> |*27.8*|*21.9*|***28.2***|-|-|
|AR<sup>max=10</sup> |*61.6*|*44.9*|***63.5***|-|-|
|AR<sup>max=100</sup> |*74.0*|*48.3*|***75.9***|-|-|
|AR<sup>small</sup> |***73.5***|*46.4*|*69.5*|-|-|
|AR<sup>medium</sup> |*79.1*|*57.9*|***84.5***|-|-|
|AR<sup>large</sup> |*64.3*|*29.7*|***65.1***|-|-|
</div align="left">
  
*<sup>1</sup> SOTA Two Stage Detector (Wang et. al.)* [`See paper`](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9353475 )    
*<sup>2</sup> SOTA One Stage Detector (Zhu et. al.)* [`See paper`](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8662457/ )

## Qualitative Evaluation

<div align="justify">
  
I created a short video from the large **ALOS-2** scene which is provided in the [official repository](https://github.com/chaozhong2010/HRSID) of the HRSID dataset and I run the Faster-RCNN and YOLOv5 models with normal bounding boxes. The rotated bounding boxes are not supported by the Detectron2 framework for video inference so the corresponding Faster-RCNN which utilizes the above bounding box type it is not used.    
  
**Faster RCNN with normal bounding boxes**
  
https://user-images.githubusercontent.com/74200033/159692748-08e85410-c274-4692-a136-d7de7155a141.mp4

**YOLOv5**

https://user-images.githubusercontent.com/74200033/159700295-efb119cb-72c1-4c68-83b9-dbe7632e7558.mp4

  

</div align="justify">

## Requirements

    torch == 1.7.1+cu110                           torchvision==0.8.2+cu110                       pyyaml == 5.1     
    detectron2 == 0.5                              cv2 == 4.1.2                                   wandb == 0.12.11
                  
  

# README of Murphy

## Jupyter Notebook

``` markdown

- Run this command from client:

ssh 8888:localhost:8888 <username>@<private ip>

Then access through http://localhost:8080

- Run this command on server

jupyter notebook --no-browser --port 8888

Remember to add token to the url for the first time, token is printed to stdout when starting the server.
```

Jupyter Notebook 切换虚拟环境

``` markdown
第一步：创建一个新的虚拟环境，这里我电脑已经有了一个装有torch的环境AAA，为了不污染这个环境，我直接复制AAA环境中的包到环境BBB中：conda create -n BBB --clone AAA
第二步：在虚拟环境下创建kernel：conda install -n BBB ipykernel
第三步：激活虚拟环境：source activate BBB
第四步：将该虚拟环境写进notebook的kernel中：python -m ipykernel install --user --name BBB --display-name "python deep_pytorch"
这时你在base环境中输入jupyter notebook打开notebook，点击右上角的”new”，这时“notebook:”列表下便会显示两个kerbel名称了。（这里要注意只能在base环境中打开jupyter notebook，因为只有base环境中装了它，虚拟环境中并没有装，而只是装了它的kernel）
还有一个tip是如果在代码编辑界面想要更改kernel，直接点击菜单栏的“Kernel”，接着点击”change kernel”，选择你想要的kernel即可。
```

``` bash
(d2l-zh) ubuntu@VM-8-15-ubuntu:~/userDoc/limu-d2l$ pip install ipykernel

(d2l-zh) ubuntu@VM-8-15-ubuntu:~/userDoc/limu-d2l$ python -m ipykernel install --user --name d2l-zh --display-name "D2L-ZH"
Installed kernelspec d2l-zh in /home/ubuntu/.local/share/jupyter/kernels/d2l-zh
```

> [动手学深度学习-环境配置|CosmicDusty](https://cosmicdusty.cc/post/AI/D2L/#%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE%E9%97%AE%E9%A2%98)


## Resullt

尝试了一下，没跑起来，原因可能是HRSID数据集配置的不对！