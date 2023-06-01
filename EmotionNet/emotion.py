#!/usr/bin/env python
# coding: utf-8
from PIL import Image
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from models import *


#class_labels = ['happy', 'suprise', 'angry', 'anxious', 'hurt', 'sad', 'neutral']
class_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
class_labels_dict = {'기쁨': 0, '당황': 1, '분노': 2,
                     '불안': 3, '상처': 4, '슬픔': 5, '중립': 6}


face_classifier = cv2.CascadeClassifier('face_classifier.xml')


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def predict(model, image, image_size):
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)

    if np.sum([image]) == 0:
        return

    roi = tt.functional.to_pil_image(image)
    roi = tt.functional.to_grayscale(roi)
    roi = tt.ToTensor()(roi).unsqueeze(0)

    # make a prediction on the ROI
    tensor = model(roi)
    probs = {class_labels[i]: round(prob, 2) for i, prob in enumerate(F.softmax(tensor, dim=1).detach().numpy()[0] * 100)}
    
    pred = torch.max(tensor, dim=1)[1].tolist()
    label = class_labels[pred[0]]

    return label, probs

def predictImage(args):
    image_size = (args.image_size, args.image_size)

    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model_state = torch.load(args.model_path, map_location=torch.device(device))
    model = getModel(args.model, silent=True)
    model.load_state_dict(model_state['model'])

    image = imread(args.img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    predictions = []

    if (args.detect_face == False):
        label, probs = predict(model, image, image_size)
        predictions.append({'label': label, 'probs': probs})
    else:
        faces = face_classifier.detectMultiScale(image, 1.3, 5)
        if len(faces) == 0:
            label, probs = predict(model, image, image_size)
            predictions.append({
                'rect': '({}, {}, {}, {})'.format(0, 0, image.shape[0], image.shape[1]),
                'label': label,
                'probs': probs
            })
        else:
            for (x, y, w, h) in faces:
                roi = image[y:y+h, x:x+w]
                label, probs = predict(model, image, image_size)
                prediction = {
                    'rect': '({}, {}, {}, {})'.format(x, y, w, h),
                    'label': label,
                    'probs': probs
                }
                predictions.append(prediction)

    print(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', action='store',
                        default='samples/happy.jpg', help='path of image to predict')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='set a switch to use GPU')
    parser.add_argument('--detect_face', action='store_true', default=False,
                        help='turn on face detection')
    
    parser.add_argument('--model_path', action='store', default='model.pth',
                        help='path of model')

    parser.add_argument('--model', action='store',
                        default='emotionnet', help='network architecture')
    
    # cnn, resnet, resmotionnet, vgg19, vgg22: 48 | vgg24: 96 | efficientnet: 224, any
    parser.add_argument('--image_size', action='store', type=int,
                        default=48, help='input image size of the network')

    # 3 for efficientnet, 1 for the rest
    parser.add_argument('--image_channel', action='store', type=int,
                        default=1, help='input image layers')
                        
    args = parser.parse_args()

    if not os.path.isfile(args.img):
        print("No image file.")
    else:
        predictImage(args)


if __name__ == "__main__":
    main()
