import json
import os
import time

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from tqdm import tqdm


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class ObjectDetection(object):
    def __init__(self, root, Entity, transforms=None):
        self.root = root
        self.transforms = transforms
        self.json = list(sorted(os.listdir(os.path.join(root, "Json"))))
        self.imgs = [os.path.splitext(x)[0] for x in self.json]
        self.scale = 1

        self.Entity_list = Entity

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        json_path = os.path.join(self.root, "Json", self.json[idx])
        img = cv2.imread(img_path)
        img = np.asarray(img)
        with open(json_path, 'r') as jsonfile:
            json_data = json.load(jsonfile)

        boxes = []
        labels = []
        idx_ = 0
        for key in json_data:
            values = json_data[key]
            for i in values:
                i = [round(self.scale * val) for val in i]
                labels.append(self.Entity_list[key] + 1)
                boxes.append([min(i[0], i[2]), min(i[1], i[3]), max(i[0], i[2]), max(i[1], i[3])])
                idx_ += 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.as_tensor([0], dtype=torch.float32)
            boxes = torch.empty((0, 4), dtype=torch.float32)

        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {"boxes": boxes,
                  "labels": labels,
                  "image_id": image_id,
                  "area": area}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        img = cv2.resize(img, (round(img.shape[1] * self.scale), round(img.shape[0] * self.scale)))
        img = np.array(img)
        img = img * (1.0 / 255.0)
        img = torch.tensor(img, dtype=torch.float)
        img = img.permute(2, 0, 1)
        return img, target, idx

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':

    with open("ObjectDetectionConfig.json") as JsonReader:
        ObjectDetectionConfig = json.load(JsonReader)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = len(ObjectDetectionConfig['Entities']) + 1

    if not os.path.exists(ObjectDetectionConfig['SaveDir']):
        os.mkdir(ObjectDetectionConfig['SaveDir'])

    backbone = mobilenet_backbone(ObjectDetectionConfig['ModelID'], True, True, trainable_layers=6)

    anchor_sizes = ((32, 64, 128, 256, 512,),) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=1)
    model = FasterRCNN(backbone, num_classes, rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
                       box_roi_pool=roi_pooler)

    train_dataset = ObjectDetection(ObjectDetectionConfig['DatasetDir'], ObjectDetectionConfig['Entities'])
    indices = torch.randperm(len(train_dataset)).tolist()

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=ObjectDetectionConfig['BatchSize'],
        shuffle=False,
        collate_fn=collate_fn
    )

    loss_hist = Averager()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=3e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(ObjectDetectionConfig['Epochs']):
        loss_hist.reset()
        time.sleep(1)
        for images, targets, image_ids in tqdm(train_data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_hist.send(loss_value)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
            print("Learning rate:" + str(lr_scheduler.get_last_lr()))


        torch.save(model.state_dict(), "Models/model_" + str(epoch) + ".pth")

        print(f"Epoch #{epoch} loss: {loss_hist.value}")
