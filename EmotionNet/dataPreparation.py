
import numpy as np
import os
import shutil
from glob import glob
from tqdm import tqdm
import argparse

# 프로젝트에서 정해진 감정의 종류
class_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
#class_labels = ['happy', 'suprise', 'angry','anxious', 'hurt', 'sad', 'neutral']
class_labels_dict = {'기쁨': 0, '당황': 1, '분노': 2,
                     '불안': 3, '상처': 4, '슬픔': 5, '중립': 6}


def main(args):
    croppedPath = args.input
    dataPath = args.output

    # UNKNOWN 폴더를 제외하지 않으려면
    #files = glob(os.path.join(croppedPath, '*/*.jpg'))
    #files.extend(glob(os.path.join(croppedPath, '*/*.jpeg')))

    # UNKNOWN 폴더는 학습에서 제외
    files = []
    for folder in class_labels:
        path = os.path.join(croppedPath, folder)
        files = files + glob(os.path.join(path, '*.jpg')) + \
            glob(os.path.join(path, '*.jpeg'))

    print(f'Total no of images {len(files)}')
    numOfImages = len(files)

    # 데이터 집합을 만드는 데 사용할 셔플 색인
    shuffle = np.random.permutation(numOfImages)

    # 레이블명으로 디렉토리 생성
    for t in ['train', 'test', 'val']:
        for folder in class_labels:
            targetPath = os.path.join(dataPath, t, folder)
            if not os.path.exists(targetPath):
                os.makedirs(targetPath)

    numOfTenPercent = int(numOfImages * 0.1)
    part1 = numOfTenPercent
    part2 = 2 * numOfTenPercent

    # test 폴더에 이미지 10% 복사
    for i in tqdm(shuffle[:part1], desc="test"):
        splits = files[i].split(os.sep)
        shutil.copy(files[i], os.path.join(
            dataPath, 'test', splits[-2], splits[-1]))

    # val 폴더에 이미지 10% 복사
    for i in tqdm(shuffle[part1:part2], desc="validation"):
        splits = files[i].split(os.sep)
        shutil.copy(files[i], os.path.join(
            dataPath, 'val', splits[-2], splits[-1]))

    # 나머지는 train
    for i in tqdm(shuffle[part2:], desc="train"):
        splits = files[i].split(os.sep)
        shutil.copy(files[i], os.path.join(
            dataPath, 'train', splits[-2], splits[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 원본 크롭 이미지가 저장되어 있는 폴더
    parser.add_argument(
        #"-i", "--input", help="image folder path", default="crop_503286"
        "-i", "--input", help="image folder path", default="crop"
    )
    # 데이터 집합이 저장될 폴더
    parser.add_argument(
        #"-o", "--output", help="output folder path", default="data_503286"
        "-o", "--output", help="output folder path", default="data"
    )
    args = parser.parse_args()

    main(args)
