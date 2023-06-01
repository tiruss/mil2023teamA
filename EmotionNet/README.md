# Facial Emotion Recognition

얼굴 표정 인식을 위한 데이터 구축 및 인식 모델 학습

## 데이터셋 이미지 정보

- 얼굴 표정 데이터는 크게 "연기자" 데이터와 "일반인" 데이터로 구성되어 있으며 그 비율은 약 5:5로 되어 있음
- 얼굴 표정은 ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립’] 7개로 분류됨
- 메타데이터로는 3명의 작업자가 판단한 얼굴 표정, 이미지에서 얼굴이 위치한 영역, 배경 정보가 포함되어 있음

주어진 TSV(tab-separated variables)는 다음과 같은 컬럼들로 구성되어 있다.

```
name
url	TaskA_emotion
TaskB_emotion
TaskC_emotion
TaskA_background
TaskB_background
TaskC_background
TaskA_result
TaskB_result
TaskC_result
```

## 개발 및 테스트 환경

- Ubuntu 18.04
- python 3.7
- pytorch 1.7.1
- PIL 8.1.0 \*(8버전 이하 시 이미지 crop시 오류 발생)
- Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz
- NVIDIA Quadro RTX 8000 48GB

## 의존 프로그램 설치

```python
pip install -r reqiurements.txt
```

## 데이터 다운로드

```python
python downloadImage.py
```

- TSV 파일에 있는 URL에 접근하여 파일을 다운로드 받는다.
- /original 폴더 아래에 [기쁨, 당황, 분노, ...] 하위 폴더를 만들어 파일을 저장한다.
- 다운로드시 발생하는 오류는 download.error.log 파일에 기록된다.
- 일시적인 네트워크 오류인 경우, 재시도하면 다운로드 받을 수 있다.
- 오류 파일을 재시도 할 경우, 오류 로그 파일을 입력으로 넣어서 오류 파일만 다시 받을 수 있다.

  ```sh
  python downloadImage.py --input download.error.log
  ```

## 얼굴 영역 자르기

```python
python cropImages.py
```

- 얼굴 표정 인식은 얼굴 영역만 잘린 이미지를 기준으로 수행한다.
- TSV 파일에 있는 영역을 기준으로 /original 폴더 아래에 있는 데이터에서 얼굴 영역만 자른다.
- 얼굴 영역만 잘린 이미지는 /crop 폴더 아래에 [기쁨, 당황, 분노, ...] 하위 폴더를 만들어 파일을 저장한다.
- TSV 파일 양식 오류는 parse.error.log 파일에 기록된다.
- 이미지 파일에 있는 오류는 crop.error.log 파일에 기록된다.
- 얼굴 표정을 분류한 3명의 판단이 모두 다른 경우, /crop/UNKNOWN 폴더에 별도로 분류하며, 이는 학습 데이터에 포함하지 않는다.
- 전체 503221장의 이미지 파일 중, UNKNOWN으로 분류되는 이미지는 73264장이다.

## 학습을 위한 데이터셋 준비

```python
python dataPreparation.py
```

- 얼굴 영역 이미지들을 train, validation, test 세 개의 세트로 나누어 저장한다.
- /data 폴더 아래에 [train, validation, test] 하위 폴더를 만들어 파일을 저장한다.
- 분할하는 비율은 train 80% (343967), validation 10% (42995), test 10% (42995) 씩으로 정해져 있으며, 변경할 수 있다.
-

## 데이터 학습

```python
python train.py [args]
```

- 학습에 사용하는 각종 파라미터를 정할 수 있다.
- 입력 파라미터(기본값)는 다음과 같이 설정되어 있다. (다른 스크립트의 파라미터도 이와 유사함)

  - --data_path (data): 학습에 사용할 데이터가 담긴 root folder
  - --model_path (model.pt): 학습한 결과를 저장할 모델의 파일명
  - --model (emotionnet): 학습에 사용할 네트워크 아키텍쳐로 cnn, resnet, vgg19, efficientnet 등이 구현되어 있음
  - --optimizer (adadelta): 학습에 사용할 optimizer로 adadelta, adam, sgd을 사용할 수 있음
  - --image_size (48): 이미지 리사이즈 크기
  - --image_channel (1): 이미지 채널 크기 
  - --gpu (True): 시스템에 GPU가 지원되는 경우, GPU를 사용하는지 여부
  - --cuda_idx (0): 여러 개의 GPU를 사용할 경우, GPU 선택
  - --epochs (50): 학습을 진행할 epoch 설정
  - --batch_size (128): 배치의 크기 설정
  - --lr (0.1): 학습률 설정

## 학습 결과

```python
python test.py [args]
```

- /data/test 데이터를 바탕으로 모델의 정확도를 평가할 수 있다.
- 입력 파라미터(기본값)는 다음과 같이 설정되어 있다.

  - --model_path (model.pt): 학습한 결과를 저장할 모델의 파일명
  - --data_path (data): test set이 위치한 폴더
  - --batch_size (128): 배치의 크기 설정
  - --image_size (48): 이미지 리사이즈 크기
  - --image_channel (1): 이미지 채널 크기 


### Accuracy (%)

| Network                       | Validation (%) |  Test (%)  | Model File Size | 
| ----------------------------- | ---------- | ------ | --------------- |
| CNN (EmotionNet)              |   81.130   | 80.858 | 55M             |
| ResEmotionNet                 |   81.014   | 79.879 | 23M             |
| ResNet18                      |   82.351   | 80.941 | 128M            |
| VGG19                         |   82.407   | 80.195 | 230M            |
| VGG22 (wide)                  |   82.303   | 79.181 | 286M            | 
| VGG24 (deep-wide)             |   83.272   | 81.393 | 610M            |
| EffectiveNet-b4               |   83.328   | 82.112 | 223M            |
| EffectiveNet-b5 (img-small)   |   81.872   | 81.851 | 350M            | 
| EffectiveNet-b5 (img-large)   |   83.356   | 83.028 | 350M            | 


### 표정별 인식 결과

| Truth\Prediction  |0|1|2|3|4|5|6|
|-------------------|-|-|-|-|-|-|-|
|0|7304|50|32|18|5|37|58|
|1|39|6202|126|316|22|66|178|
|2|26|181|5025|307|64|200|129|
|3|37|484|317|2110|170|590|309|
|4|8|158|141|275|528|931|193|
|5|24|61|132|257|389|6577|172|
|6|54|178|77|196|119|171|7952|

- EffectiveNet-b5 (img-large), 343967 이미지, 50 epochs 학습, Test set 42995장 결과

## 학습한 얼굴 표정 인식 모델 사용 예시 및 결과

```sh
$ python emotion.py --img happy.jpg
[{'label': '기쁨', 'probs': [0.0, 0.0, 1.16, 0.09, 0.27, 98.48, 0.0]}]
```

- 결과 양식은 JSON string이며, emotion에 얼굴 표정 감정에 대한 prediction 결과를 담고 있다.
- 얼굴 표정 인식 모델은 얼굴 이미지를 대상으로 수행되며, 입력 이미지는 얼굴 영역을 자른 이미지를 입력한다.
- 얼굴 영역을 찾는 face detaction은 본 프로젝트의 범위는 아니지만, 해당 기능을 옵션을 통해 추가 수행할 수 있으며, Haar Cascade Face Classifier를 사용할 수 있도록 제공한다.

  - Haar Cacade Face Classifier

    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html


- 입력 파라미터(기본값)는 다음과 같이 설정되어 있다.

  - --img (기쁨.jpg): 입력 이미지 파일명
  - --model_path (model.pt): 얼굴 표정 인식 모델의 파일명
  - --gpu (True): 시스템에 GPU가 지원되는 경우, GPU를 사용하는지 여부
  - --detect_face (False): 입력 이미지가 얼굴 표정만 담긴 이미지가 아닐 경우, face detection을 진행

    - 옵션이 켜지면 출력에 검출된 얼굴 영역 정보가 다음과 같이 함께 기록된다.

    ```sh
    $ python emotion.py --img sample.jpg --detect_face
    [{'rect': '(886, 292, 1324, 1324)', 'label': '기쁨', 'probs': [0.0, 0.0, 1.16, 0.09, 0.27, 98.48, 0.0]}]
    ```

    - rect (x, y, w, h) 는 얼굴 영역에 대한 prediction 결과를 담고 있다.
    - 이미지에서 얼굴 영역을 찾을 수 없는 경우, 입력 이미지 전체를 얼굴 이미지로 가정하고 얼굴 표정 인식을 진행한다.
    - 이미지에서 복수의 얼굴을 찾은 경우, 이에 대한 얼굴 표정 인식 결과를 list로 출력한다.

## 실시간 비디오 표정 인식

```sh
$ python video.py
```

* 영상을 입력 받을 수 있는 카메라가 연결되어 있을 경우, 카메라로 입력되는 영상에 대해 얼굴 표정 인식을 수행할 수 있다.

