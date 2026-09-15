"""
Label trong dataset hiện tại có dạng: class x_center y_center width height
các giá trị x_center, y_center, width, height đều được chuẩn hóa về khoảng [0, 1] so với kích thước ảnh
Đòi hỏi phải qua một số bước tiền xử lý để có thể đưa vào mô hình YOLOv3
"""

import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    iou_width_height as iou,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True # cho phép load các hình ảnh bị cụt

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir, 
        label_dir,
        anchors, # anchors truyền vào sẽ là một list gồm 3 list con, mỗi list con chứa 3 anchor box
        image_size=416,
        S=[13, 26, 52], # scales
        C=20, # số class  
        transform=None, # transform cho phép thực hiện các phép biến đổi trên hình ảnh
    ):
        self.annotations = pd.read_csv(csv_file) # annotations sẽ chứa 2 cột là tên file ảnh và tên file label
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S # S is an array of scales
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) 
        self.num_anchors = self.anchors.shape[0] # self.num_anchors lúc này đã là 9
        self.num_anchors_per_scale = self.num_anchors // 3 
        self.C = C
        self.ignore_iou_thresh = 0.5 # ý tưởng dùng ignore_iou_thresh giống NMS

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index): # each index is corresponding to one image
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # 1 because label path is in the second column of the csv file
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=' ', ndmin=2), 4, axis=1).tolist() # to roll the row with the format [class, x, y, w, h] to [x, y, w, h, class]
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0]) # 0 because image path is in the first column of the csv file
        image = np.array(Image.open(img_path).convert("RGB")) # convert to RGB to avoid grayscale images

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes) # transform function will be built by albumentations library
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.S] # torch.zeros creates a tensor with [number of anchors, scale, scale, 6(objectness, x, y, w, h, class)], so target is a label tensor of YOLOv3, we have 3 elements in targets
        # targets is a label of an image

        for box in bboxes: # each cell and anchor is responsible for predicting one bounding box, so we need to assign each target bouding box to a cell and anchor of one of 3 scales
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors) # to find which anchor is assigned to the target bounding box
            anchors_indices = iou_anchors.argsort(descending=True, dim=0)  # return a list of indices of anchors
            x, y, width, height, class_label = box # each box can be assigned to 3 cell of 3 scales
            has_anchor = [False] * 3 # to check if the target bounding box has been assigned to an anchor of each scale

            # 2 reason why we need to loop through acnhors_indices:
            #  - We need to ignore some noise bounding box
            #  - There can be 2 object in a cell and they can have the same best anchor. When we loop, the best anchor will be used for the first object, 
            # and the usable other anchor will be used for the second object but it can be pretty much bad for matching
            for anchor_idx in anchors_indices: 
                scale_idx = anchor_idx // self.num_anchors_per_scale # tell us which scale we gonna use
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # tell us which anchor of the scale we gonna use
                S = self.S[scale_idx]

                # the label bounding box is associated with entire image, so we need to convert it to the cell coordinates
                i, j = int(S * y), int(S * x) # i, j are the cell coordinates, that cell will contain the midpoint of object
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] # return the objectness score of the anchor
 
                if not anchor_taken and not has_anchor[scale_idx]: # to make sure that anchor is only one in that scale and the anchor have no object
                    # khi vào đây objectness từ 0 đã thành 1 và mọi đằng sau như bouding box, class cũng được gán
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1 # objectness score
                    has_anchor[scale_idx] = True # mỗi scale chỉ được dùng 1 trong 3 anchor, nên sẽ có 6 anchor ko xét đến
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = (
                        width * S, height * S
                    ) # to convert the width and height of the bounding box to the cell coordinates
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    ) 
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates 
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) 

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh: # bản chất các anchor gây nhiễu vẫn có khả năng đúng rất cao, nếu không ignore nó thì mô hình sẽ coi nó là sai, và tất nhiên điều đó ko đúng, mô hình sẽ coi cái đúng là sai
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # to ignore the anchors that make noise

        return image, tuple(targets) 

            # first of all, we have label of an image that contains [class, x, y, w, h]
            # targets have 3 cells, each cell is a tensor of shape [number of anchors per scale, S, S, 6], where 6 is [objectness, x, y, w, h, class]
            # what we gonna do is to place matching anchor(which we need to find by calculating IoU), cell coordinates into target tensor
            # x, y are used to caculate the cell coordinates
            # w, h are used to caculate the highest iou anchor
            # anchor index is used to find which scale and which anchor on that scale we gonna use
            # the final step is to place scale index, anchor index on scale, cell coordinates, objectness score(1), box coordinates, class label into the target tensor
            # the target tensor could be as follows:


            # [
            # [[0, 0, 0, 0, 0, 0],...],
            # [[0, 0, 0, 0, 0, 0],...
            # [1, 0.5, 0.5, 3, 4, 19],...
            # [0, 0, 0, 0, 0, 0]],
            # [[0, 0, 0, 0, 0, 0],...]]
            # ]

            # ---> pretty much to do for the dataset

            ############################################################################################################################
            # Nôm na quá trình xử lý dữ liệu là từ x, y, w, h ban đầu để tìm ra scale, anchor, tọa độ cell, tọa độ midpoint trong cell #
            # Lấy scale, anchor, tọa độ cell, tọa độ midpoint mới tìm được để tạo ra label mới phù hợp cho yolov3                      #
            ############################################################################################################################

            # Luồng:
            # lặp tất cả các box trong một ảnh
            # xét 1 box:
            # sử dụng 9 anchor, sắp xếp các anchor theo thứ tự có iou cao đến thấp
            # quy định trong mỗi scale chỉ có 1 anchor được xét
            # các cell và anchor đã được gán với 1 box khác rồi sẽ ko đc xét nx, các cell và anchor bị ignore rồi thì cũng không đc xét
            # các cell và anchor được xét sẽ đc gán bbox, class. Còn cell ko xét đến thì bbox, class là 0 hết. Riêng anchor bị ignore cũng có bbox và class là 0 tuy nhiên nó sẽ ko đc dùng để tính no object loss