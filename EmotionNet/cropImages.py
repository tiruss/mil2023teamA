import argparse
import csv
from pathlib import Path

import os
import logging
import json
import numpy as np
from scipy import spatial
from PIL import Image, ExifTags, ImageOps, ImageFile

# 이미지 파일에 TRUNCATED ERROR가 있는 것들이 있음 -> 제외하여 학습에 사용하지 않는다.
ImageFile.LOAD_TRUNCATED_IMAGES = False

# 프로젝트에서 정해진 감정의 종류
class_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
#class_labels = ['happy', 'suprise', 'angry','anxious', 'hurt', 'sad', 'neutral']
class_labels_dict = {'기쁨': 0, '당황': 1, '분노': 2,
                     '불안': 3, '상처': 4, '슬픔': 5, '중립': 6}
class_labels_dict_index = {0: '기쁨', 1: '당황',
                           2: '분노', 3: '불안', 4: '상처', 5: '슬픔', 6: '중립'}

'''
TSV 파일 인덱스 (1행)
name	url	TaskA_emotion	TaskB_emotion	TaskC_emotion	TaskA_background	TaskB_background	TaskC_background	TaskA_result	TaskB_result	TaskC_result
'''
indexFilename = 0
indexUrlColumn = 1
indexLabelA = 2
indexLabelB = 3
indexLabelC = 4
indexTaskA_result = 8
indexTaskB_result = 9
indexTaskC_result = 10


def log(name, message):
    print(name, message)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)

    file_handler = logging.FileHandler(
        name + '.log', mode='a', encoding='utf-8')

    logger = logging.getLogger(name)
    logger.addHandler(stream_hander)
    logger.addHandler(file_handler)
    logger.error(message)


def crop(inputFilename, outputFilename, box):
    path = Path(outputFilename).parent.absolute()
    if not os.path.exists(path):
        os.makedirs(path)

    image = Image.open(inputFilename)

    '''
    EXIF orientation    
    =======================
    1	Top	Left side
    2*	Top	Right side
    3	Bottom	Right side
    4*	Bottom	Left side
    5*	Left side	Top
    6	Right side	Top
    7*	Right side	Bottom
    8	Left side	Bottom
    '''
    '''
    exif = { ExifTags.TAGS[k]: v for k, v in image._getexif().items() if k in ExifTags.TAGS }
    if ("Orientation" in exif) and (exif["Orientation"] == 3):
        image = image.rotate(180)
    '''
    image = ImageOps.exif_transpose(image)

    cropped = image.crop(box)
    cropped.save(outputFilename)


def getEmotionClassByString(string):
    for emotion in class_labels:
        if emotion in string:
            return emotion
    return "UNKNOWN"


def getMaxVoteAndCount(nparray):
    (unique, counts) = np.unique(nparray, return_counts=True)
    vote = unique[np.argmax(counts)]
    return vote, np.max(counts)


def getEmotionClassByLabel(row):
    labels = np.array([row[indexLabelA], row[indexLabelB], row[indexLabelC]])

    vote, count = getMaxVoteAndCount(labels)
    # 세 명의 의견이 다 다른 상황
    if count == 1:
        return "UNKNOWN"

        # 사진 원작자의 의도를 추가 고려
        filename = row[indexFilename]
        labels = np.append(labels, filename)
        vote, count = getMaxVoteAndCount(labels)

        if count == 1:
            return "UNKNOWN"

    return getEmotionClassByString(vote)


def dequote(s):
    """
    If a string has single or double quotes around it, remove them.
    Make sure the pair of quotes match.
    If a matching pair of quotes is not found, return the string unchanged.
    """
    if (s[0] == s[-1]) and s.startswith(("'", '"')):
        return s[1:-1]
    return s


def jsonParse(s):
    jsonResult = s.replace('""', '"')
    jsonResult = dequote(jsonResult)
    return json.loads(jsonResult)


def readImageAndCrop(delimiter, csvInputFilename, originalFolder, outputFolder):

    with open(csvInputFilename, 'r', encoding='utf-8') as csvFile:
        reader = csv.reader(csvFile, delimiter=delimiter, quotechar="'")

        for i, row in enumerate(reader):

            if (i == 0) and ("name" in row):
                # 첫번째 줄은 컬럼명 명시
                # 컬럼명이 없는 경우에도 기본값으로 동작
                indexFilenameColumn = row.index("name")
                indexTaskA_result = row.index("TaskA_result")
                indexTaskB_result = row.index("TaskB_result")
                indexTaskC_result = row.index("TaskC_result")
                continue

            # 원본이 저장되어 있는 파일명
            filename = row[indexFilenameColumn]
            emotionClass = getEmotionClassByString(filename)
            imagePath = os.path.join(originalFolder, emotionClass, filename)

            # crop 후 저장할 파일명
            labeledClass = getEmotionClassByLabel(row)
            savePath = os.path.join(outputFolder, labeledClass, filename)

            # 얼굴이 들어 있는 box 영역에 대해 3명의 의견 정리
            taskResults = []
            taskResults.append(jsonParse(row[indexTaskA_result]))
            taskResults.append(jsonParse(row[indexTaskB_result]))
            taskResults.append(jsonParse(row[indexTaskC_result]))
            boxPoints = np.zeros((len(taskResults), 4), dtype=float)

            try:
                for indexResult, taskResult in enumerate(taskResults):
                    # print(taskResult["boxes"])

                    # 파일 양식에 있는 오류 처리: "boxes" 항이 없이 잘못 기록된 데이터가 존재함
                    if isinstance(taskResult, list):
                        taskResult = {'boxes': taskResult}

                    if "boxes" not in taskResult:
                        raise Exception(str(i) + ' no "boxes" property')

                    for box in taskResult["boxes"]:
                        #print([box["minX"], box["minY"], box["maxX"], box["maxY"]])
                        boxPoints[indexResult] = [box["minX"],
                                                  box["minY"], box["maxX"], box["maxY"]]
            except Exception as e:
                log("parse.error", str(i) + "\t" + imagePath)

                # 오류를 기록 후 다음 파일로 진행한다
                continue

            avgBoxPoints = np.zeros(4)

            PICKY_SELECTION = True
            if PICKY_SELECTION:
                # 그 중, 가장 의견이 다른 1명의 의견을 배제하고, 나머지 2명의 의견을 선택하여 평균
                dists = spatial.distance.pdist(boxPoints)
                minIndex = np.argmin(dists)
                if minIndex == 0:
                    avgBoxPoints = np.average(boxPoints[0:2], axis=0)
                elif minIndex == 1:
                    avgBoxPoints = np.average(boxPoints[0:3:2], axis=0)
                else:
                    avgBoxPoints = np.average(boxPoints[1:3], axis=0)
                # print(avgBoxPoints)
            else:
                # 세 명의 평균 -> 한 명 정도 오차가 있음
                avgBoxPoints = np.average(boxPoints, axis=0)

            print(str(i), "cropping...\n\t", imagePath, "\n\t->", savePath)

            try:
                crop(imagePath, savePath, avgBoxPoints)
            except Exception as e:
                log("crop.error", str(i) + "\t" + imagePath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        #"-i", "--input", help=".csv file input", default="./metadata/emotions_39631.tsv"
        # "-i", "--input", help=".csv file input", default="./metadata/emotions_253286.tsv"
        #"-i", "--input", help=".csv file input", default="./metadata/emotions_250000.tsv"
        "-i", "--input", help=".csv file input", default="./metadata/emotions.tsv"
    )
    parser.add_argument(
        #"-f", "--folder", help="original image folder path", default="original_39631"
        # "-f", "--folder", help="original image folder path", default="original_253286"
        #"-f", "--folder", help="original image folder path", default="original_250000"
        "-f", "--folder", help="original image folder path", default="original"
    )
    parser.add_argument(
        #"-o", "--output", help="output folder path for data download", default="crop_39631"
        # "-o", "--output", help="output folder path for data download", default="crop_253286"
        #"-o", "--output", help="output folder path for data download", default="crop_250000"
        "-o", "--output", help="output folder path for data download", default="crop"
    )
    args = parser.parse_args()

    readImageAndCrop("\t", args.input, args.folder, args.output)
