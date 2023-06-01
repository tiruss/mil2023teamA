#!/usr/bin/env python
# coding: utf-8
from PIL import Image, ImageFont, ImageDraw
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

display_color = (246, 189, 86)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def main(args):

    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
        #print('GPU On')
    else:
        device = 'cpu'
        #print('GPU Off')

    model_state = torch.load(args.model_path, map_location=torch.device(device))
    model = getModel(args.model)
    model.load_state_dict(model_state['model'])

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), display_color, 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48),
                                  interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = tt.functional.to_pil_image(roi_gray)
                roi = tt.functional.to_grayscale(roi)
                roi = tt.ToTensor()(roi).unsqueeze(0)

                # make a prediction on the ROI
                tensor = model(roi)
                probs = torch.exp(tensor).detach().numpy()
                prob = np.max(probs) * 100
                pred = torch.max(tensor, dim=1)[1].tolist()
                label = ('{} ({:.0f}%)'.format(class_labels[pred[0]], prob))

                label_position = (x, y)

                SUPPORT_UTF8 = True
                if SUPPORT_UTF8:
                    font_path = "./fonts/NotoSansKR-Regular.otf"
                    font = ImageFont.truetype(font_path, 32)
                    img_pil = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text(label_position, label, font=font, fill=display_color)
                    frame = np.array(img_pil)
                else:
                    cv2.putText(frame, label, label_position,
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, 'No Face Found', (20, 60),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow('Facial Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', action='store', default='sad.jpg',
                        help='path of image to predict')
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

    parser.add_argument('--gpu', action='store_true', default=False,
                        help='set a switch to use GPU')
    parser.add_argument('--detect_face', action='store_true',
                        default=False, help='turn on face detection')
    args = parser.parse_args()

    main(args)

