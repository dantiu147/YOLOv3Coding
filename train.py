# what is train file gonna do?

import config 
import torch
import torch.optim as optim # optimizer ?

from model import YOLOv3
import torch.nn as nn

from tqdm import tqdm # tell me the progress of the training
from utils import (
    mean_average_precision,
    cells_to_bboxes, # getting output of each cell
    get_evaluation_bboxes, # not really essential to yolo
    save_checkpoint, # we use checkpoint to save the state of each epoch
    load_checkpoint,
    check_class_accuracy,
    get_loaders, # what is loader gonna do? 
    plot_couple_examples
)

from loss import YoloLoss

torch.backends.cudnn.benchmark = True

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors): # train_fn is gonna train 1 epoch
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop): # x is data, y is label, Batch_size in config file
        x = x.to(config.DEVICE) # Những biến được model dùng thì phải có cùng thiết bị với model
        y0, y1, y2 = (
            y[0].to(config.DEVICE), # scale 1
            y[1].to(config.DEVICE), # scale 2
            y[2].to(config.DEVICE)  # scale 3
        )

        with torch.cuda.amp.autocast(): # automatic use float 16 or float 32
            out = model(x)
            loss = ( # compute loss for each scales, because in loss.py we already set loss funtion for each scale
                loss_fn(out[0], y0, scaled_anchors[0]) # use anchor for caculating bbox, more detail in loss.py
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item()) # take the value of a tensor that contain only one element
        optimizer.zero_grad() # reset gradient before next batch
        scaler.scale(loss).backward() # float 16 can have very small gradient that can be considered as 0, so it can lead to underflow, so we need to scale it before caculating the gradient. Same with overflow case. When gradient = 0, it's called vanishing gradient
        scaler.step(optimizer) # update weights after caculating the gradient
        scaler.update() # justify scale factor for next iteration

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss) # to view the loss in the progress bar

def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE) # create model, moving model to GPU
    # if torch.cuda.device_count() > 1: # for using multiple GPUs
    #     print(f"Sử dụng {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET+"/train.csv",
        test_csv_path=config.DATASET+"/test.csv",
    )

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)
    
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)
    ).to(config.DEVICE) # anchor is a tensor

    for epoch in range(config.NUM_EPOCHS):
        # train model
        train_fn(test_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, config.CHECKPOINT_SAVE_PATH)

        if epoch > 0 and epoch % 9 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD) # Check the accuracy of each class
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision( # caculate MAP
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train() 

if __name__ == "__main__":
    main()
