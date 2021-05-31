import torch
import torchvision
import json
import cv2
import numpy as np
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


class DetectImg:

    def __init__(self, ModelPath):

        with open("ObjectDetectionConfig.json") as JsonReader:
            ObjectDetectionConfig = json.load(JsonReader)

        self.EntityList = ObjectDetectionConfig['Entities']
        self.REntityList = {v: k for k, v in self.EntityList.items()}
        self.numClasses = len(ObjectDetectionConfig['Entities']) + 1
        self.imgSize = (1280, 720)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.scale = 1

        backbone = mobilenet_backbone(ObjectDetectionConfig['ModelID'], True, True, trainable_layers=6)
        anchor_sizes = ((32, 64, 128, 256, 512,),) * 3
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=1)
        self.model = FasterRCNN(backbone, self.numClasses, rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios), box_roi_pool=roi_pooler)
        self.model.load_state_dict(torch.load(ModelPath))
        self.model.eval()
        self.model.to(self.device)

    def Predict(self, Image):

        OutputDict = {key: [] for key in self.EntityList.keys()}

        Image = cv2.resize(Image, self.imgSize)
        ResizedImg = cv2.resize(Image, (round(Image.shape[1] * self.scale), round(Image.shape[0] * self.scale)))
        ResizedImg = ResizedImg * (1. / 255.0)
        ResizedImg = np.expand_dims(ResizedImg, 0)
        ResizedImg = torch.tensor(ResizedImg, dtype=torch.float)
        ResizedImg = ResizedImg.permute(0, 3, 1, 2)
        ResizedImg = ResizedImg.to(device=self.device)
        outputs = self.model(ResizedImg)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        boxes = outputs[0]['boxes'].data
        scores = outputs[0]['scores'].data
        labels = outputs[0]['labels'].data.tolist()

        FinalFilter = []
        size = torchvision.ops.remove_small_boxes(boxes, min_size=60).data.tolist()
        filter = torchvision.ops.nms(boxes, scores, 0.2).data.tolist()
        for i, element in enumerate(boxes):
            if (i in filter) and (i in size):
                FinalFilter.append(i)

        for i, box in enumerate(outputs[0]['boxes'].data.tolist()):
            if i in FinalFilter:
                point1 = (round(box[0] / self.scale), round(box[1] / self.scale))
                point2 = (round(box[2] / self.scale), round(box[3] / self.scale))

                OutputDict[self.REntityList[labels[i] - 1]].append([point1, point2])

        return OutputDict


if __name__ == '__main__':

    Camera = cv2.VideoCapture(0)
    Camera.set(3, 1280)
    Camera.set(4, 720)

    Predictor = DetectImg("Models/model_25.pth")

    while True:
        _, img = Camera.read()

        OutputDict = Predictor.Predict(img)

        for key in OutputDict.keys():
            for box in OutputDict[key]:

                point1 = box[0]
                point2 = box[1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (box[0][0], box[0][1] - 10)
                fontScale = 2
                color = (0, 0, 255)
                thickness = 2
                img = cv2.putText(img, key, org, font,fontScale, color, thickness, cv2.LINE_AA)
                cv2.rectangle(img, point1, point2, (220, 0, 0), 3)

        cv2.imshow("Preview", img)
        cv2.waitKey(1)



