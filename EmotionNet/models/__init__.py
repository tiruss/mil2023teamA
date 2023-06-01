from .emotionnet import *
from .resnet import *
from .faceemotioncnn import *
from .convnet import *
from .cnn import *
from .vgg import *
from .resemotion import *
from .efficientnet import EfficientNet, VALID_MODELS
from .utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)

def getModel(modelName, silent=False):    
    modelName = modelName.lower()
    if modelName == "vgg19":
        if not silent: print("Model - VGG19")
        return VGG("VGG19")
    elif modelName == "vgg22":
        if not silent: print("Model - VGG22")
        return VGG("VGG22")
    elif modelName == "vgg24":
        if not silent: print("Model - VGG24")
        return VGG("VGG24")
    elif modelName == "resnet18":
        if not silent: print("Model - ResNet18")
        return ResNet18()
    elif modelName == "emotionnet":
        if not silent: print("Model - EmotionNet")
        return EmotionNet()
    elif modelName == "resemotionnet":
        if not silent: print("Model - ResEmotionNet")
        return ResEmotionNet()
    elif modelName == "efficientnet-b4":
        if not silent: print("Model - EfficientNet-b4")
        return EfficientNet.from_name('efficientnet-b4')
    elif modelName == "efficientnet-b5":
        if not silent: print("Model - EfficientNet-b5")
        return EfficientNet.from_name('efficientnet-b5')

    if not silent: 
        print("Invalid model input:", modelName)
        print("Use instead: CNN")
        
    return CNN()