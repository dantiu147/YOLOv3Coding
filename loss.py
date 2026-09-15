from pandas._libs import properties
import torch
import torch.nn as nn

from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss() # for box prediction
        self.bce = nn.BCELoss() # bce is sigmoid cross entropy loss
        self.entropy = nn.CrossEntropyLoss() # crossentropy is softmax cross entropy loss
        self.sigmoid = nn.Sigmoid()

        # Constants
        self.lambda_class = 1 
        self.lambda_noobj = 10 
        self.lambda_obj = 1
        self.lambda_box = 10
        # all these stuff above is for balancing loss between different types of loss
        # but that guy said there is nothing magical about these stuff

    def forward(self, predictions, target, anchors): # we gonna compute the loss for each of 3 scales
        obj = target[..., 0] == 1 # take out the cell, the anchor that have object # obj mất 1 chiều so với target
        noobj = target[..., 0] == 0 # take out the cell, the anchor that don't have object
        # what we have done is to ignore the cell with the anchor that is noise for used anchors

        # No object loss
        no_object_loss = self.bce(
            torch.sigmoid(predictions[..., 0:1][noobj]), target[..., 0:1][noobj] # it's called slicing and boolean mask indexing and it's new to me # predictions[..., 0:1] phải hơn noobj 1 chiều để boolean mask indexing không bị lỗi 
            # (predictions[..., 0:1][noobj]) will return a 1D tensor of confidence score of cells and anchors
            # (target[..., 0:1][noobj]) will return 1D tensor of zeros, that means there no object in those cells and anchors
        )
        # Object Loss (Tell us which anchor and cell will have an object)
        anchors = anchors.reshape(1,3,1,1,2) # [3,2] -> [1,3,1,1,2] # Anchor is a tensor
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1) # sigmoid(predictions[...,1:3]) is to make sure the coordinate is between 0-1, [B,3,13,13,2]*[1,3,1,1,2](broadcasting)
        # cái thực chất mô hình đoán ra đó là t_w và t_h, còn w thực sự là anchor_w * exp(t_w) và tương tự với h
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach() # compute the iou of the box predictions and the target bounding boxes, only for the cells and anchors that have objects
        object_loss = self.bce(torch.sigmoid(predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj])) 
        # việc tính loss ở đây không chỉ tính giữa việc có obj ở những cell hay anchor hay không
        # việc tính loss ở đây là tính giữa việc đoán có object là bao nhiêu phần trăm và so sánh với hiệu quả của việc đoán bouding box có tốt không
        # tránh trường hợp cell dự đoán là có object, nhưng bouding box thì không trùng với vật thể
        # còn đối với tính loss cho trường hợp no object, khi một cell và anchor gán là không object thì nó cũng không có bbox gì đằng sau cả
        

        # Box Coordinate Loss
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3]) # x, y to be between [0,1]
        target[..., 3:5] = torch.log( # để tính t_w, t_h từ w, h của target
            (1e-16 + target[..., 3:5] / anchors) # w, h của target là giá trị được tính dựa trên scale; target[..., 3:5] / anchors = (exp(t_w), exp(t_h)), do đó (t_w, t_h) = ln(target[..., 3:5] / anchors)
        )
        
        box_loss = self.mse(
            predictions[..., 1:5][obj], target[..., 1:5][obj] # calculating loss of x, y, t_w, t_h (x, y is the midpoint of the object in a cell)
        ) 

        # Class Loss
        class_loss = self.entropy(predictions[..., 5:][obj], target[..., 5][obj].long())
        # the last dimension of the predictions have 25 elements
        # the 1st element, we set it as the objectness
        # the 2nd - 5th elements, we set is as the bbox (x, y, w, h)
        # the 6th - 25th elements, we set is as the class probability 
        # we already set the targets first, and set the predictions accordingly to the targets
        # the different between targets and predictions is in the class probability, the predictions have 20 elements of class probability, while the targets have only 1 element of class probability

        return(
            self.lambda_box * box_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_obj * object_loss
            + self.lambda_class * class_loss
        )

        

    
