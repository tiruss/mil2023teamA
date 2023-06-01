import argparse
import csv
from pathlib import Path

import os
import requests
import logging


# 프로젝트에서 정해진 감정의 종류 
class_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
#class_labels = ['happy', 'suprise', 'angry','anxious', 'hurt', 'sad', 'neutral']
class_labels_dict = {'기쁨': 0, '당황': 1, '분노': 2, '불안': 3, '상처': 4, '슬픔': 5, '중립': 6}


'''
TSV 파일 인덱스 (1행)
name	url	TaskA_emotion	TaskB_emotion	TaskC_emotion	TaskA_background	TaskB_background	TaskC_background	TaskA_result	TaskB_result	TaskC_result
'''
indexUrlColumn = 1            
indexLabelA = 2
indexLabelB = 3
indexLabelC = 4


def log(name, message):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)

    file_handler = logging.FileHandler(name + '.log', encoding='utf-8')
    
    logger = logging.getLogger(name)
    logger.addHandler(stream_hander)
    logger.addHandler(file_handler)
    logger.error(message)


def download(url, filename): 
    path = Path(filename).parent.absolute()
    if not os.path.exists(path):
        os.makedirs(path)

    with open(filename, "wb") as file:   # open in binary mode
        response = requests.get(url)      # get request
        file.write(response.content)      # write to file


def getEmotionClassByFilename(filename):
    for emotion in class_labels:
        if (emotion in filename):
            return emotion

    return "UNKNOWN"

def readCSVandDownload(delimiter, csvInputFilename, outputFolder): 
 
    with open(csvInputFilename, 'r', encoding='utf-8') as csvFile:
        reader = csv.reader(csvFile, delimiter=delimiter, quotechar="'")

        for i, row in enumerate(reader):    
            if (i == 0) and ("url" in row):                
                # 첫번째 줄은 컬럼명 명시
                # 컬럼명이 없는 경우에도 기본값으로 동작
                indexUrlColumn = row.index("url")
                continue

            url = row[indexUrlColumn] 
            filename = url.split('/')[-1]
            emotionClass = getEmotionClassByFilename(filename)
            savePath = os.path.join(outputFolder, emotionClass, filename)

            print(str(i), "\tdownloading...\n\t", url, "\n\t->", savePath)
                
            try:
                download(url, savePath)
            except Exception as e:
                log("download.error", str(i) + "\t" + url) 
                print("download.error", str(i) + "\t" + url)
       

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        #"-i", "--input", help=".csv file input", default="./metadata/emotions_39631.tsv" 
        #"-i", "--input", help=".csv file input", default="./metadata/emotions_253286.tsv"
        #"-i", "--input", help=".csv file input", default="./metadata/emotions_250000.tsv"
        "-i", "--input", help=".csv file input", default="./metadata/emotions.tsv"
    ) 
    parser.add_argument(
        #"-o", "--output", help="output folder path for data download", default="original_250000"
        "-o", "--output", help="output folder path for data download", default="original"
    ) 
    args = parser.parse_args()

    readCSVandDownload("\t", args.input, args.output)
