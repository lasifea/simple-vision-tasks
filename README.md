简体中文 | [English](README_EN.md)

# simple-vision-tasks

some useful and interesting vision tasks of Python👀

一些使用Python开发的实用且有趣的视觉任务👀

## 简介
该项目致力于将一些实用且有趣的视觉任务以更简单、高效、易于部署和维护的形式展现，推理框架目前均采用**onnxruntime**，尽可能达到Python可优化的极限，模型大多来自著名的项目，代码中会指明出处。功能与原项目相比，模型推理步骤大幅度重构，依赖包更少且执行速度更快🚀。

如果有帮助到您的话，不妨点击右上角的小星星⭐吧，您的支持就是作者分享的最大动力。

## 功能
- **人脸检测** （来自 [**insightface**](https://github.com/deepinsight/insightface)）
- **人脸关键点检测** （同上）
- **性别年龄预测** （同上）
- **人脸比对** （同上）
- **文字识别** （来自 [**PaddleOCR**](https://github.com/PaddlePaddle/PaddleOCR)）
- **coming soon...** （如图像分类、目标检测）

## 近期更新
**`2024-08-18`** 修改项目结构，增加所有视觉任务的基类；增加性别年龄预测功能，代码位于utils/genderage_predictor.py，使用方式与其它功能均保持一致；修改README文件。

---

## 下载安装

推荐使用[**Python>=3.8**](https://www.python.org/)，补丁版本越高越好
```bash
pip install -r requirements.txt
```

## 在Python中使用

功能代码位于**utils**目录下，创建类示例时可指定**线程数**和是否使用**显卡**推理，使用方式如下。
```python
from utils.face_detector import FaceDetector

detector = FaceDetector('models/det_500m.onnx', thread_num=2, use_gpu=False)
results = detector('images/1.jpg')
```
![](images/output.jpg)

通过将简单的功能组合在一起，可实现一些有趣的功能🚗，以下可用于人脸比对，建议相似度阈值设定为**0.45**。
```python
import cv2
from utils.face_detector import FaceDetector
from utils.face_recognizer import FaceRecognizer, one2one, one2many

img = cv2.imread('images/3.jpg')
img2 = cv2.imread('images/4.jpg')
detector = FaceDetector('models/det_500m.onnx')
recognizer = FaceRecognizer('models/w600k_mbf.onnx')

results = detector.forward(img)
results2 = detector(img2)
known_embedding = recognizer.forward(img, results[0]['points'])
unknown_embedding = recognizer(img2, results2[0]['points'])
similarity = one2one(known_embedding, unknown_embedding)

print('人脸相似度：', similarity)
```

以下参考**PaddleOCR**实现的文字识别功能，模型使用的是PP-OCRv4，通常效果比PP-OCRv3更好，但耗时增加约10%，想了解更多，请[**点击这里**](https://github.com/PaddlePaddle/PaddleOCR)。
```python
from utils.paddleocr import OCRProcessor

ocr = OCRProcessor('models/ch_ppocrv4_det.onnx',
                   'models/ch_ppocrv4_rec.onnx',
                   'models/rec_word_dict.txt',
                   cls_model_path='models/ch_ppocrv2_cls.onnx',
                   thread_num=3)

results = ocr('images/11.jpg')
print(results)
```
![](images/output2.jpg)

## 速度测试

- 测试使用的Python版本为3.11，测试平台为Windows11，cuda版本为11.8，i5 5200U的环境为Python3.10 Ubuntu22.04。
- 测试为完整功能的速度（包括预处理，模型推理，后处理），增加线程数或更换显卡能提升模型推理速度，但不能改变预处理及后处理的用时。
- 文字识别速度视图片中的文本数量，因此以下仅测试单次使用的耗时，文字检测使用的图片为images/11.jpg，方向分类及文字识别使用的图片为images/12.jpg。

| 模型/速度（ms）                               | i5 13400<br>（2 threads） | i5 13400<br>（3 threads） | i5 5200U<br>（2 threads） | GTX 2060s 8G |
|-----------------------------------------|-------------------------|-------------------------|-------------------------|--------------|
| **det_500m** <br>(FaceDetector)         | 15                      | 12.5                    | 41                      | 7            |
| **2d106det** <br>(Face2DLandmarker)     | 3.7                     | 3.4                     | 7.1                     | /            |
| **genderage** <br>(GenderAgePredictor)  | 1.3                     | 1.3                     | 3                       | /            |
| **w600k_mbf** <br>(FaceRecognizer)      | 6                       | 4.7                     | 15                      | /            |
| **ch_ppocrv4_det** <br>(TextDetector)   | 122                     | 101                     | 486                     | 23.5         |
| **ch_ppocrv4_cls** <br>(TextClassifier) | 4.5                     | 3.7                     | 12.5                    | 2.2          |
| **ch_ppocrv4_rec** <br>(TextRecognizer) | 52                      | 38                      | 305                     | 9.1          |

---

## 作者有话说

- 这是本人的第一次代码分享，希望大家能友善交流，指出代码中还可以优化的地方，相互学习促进😉，如果效果好，下次将增加人像修复及换脸的有趣功能。
- 笔者并不擅长深度学习领域，在深度学习模型部署中踩坑无数，最终总结出一些模型部署心得。
- 模型部署最重要的是实现思路，清楚如何做图像预处理和后处理后，可根据自己需求更换不同的推理框架或不同编程语言实现。
- 项目持续更新中。。。。。。

## 常见问题

- **Q**：如何安装Python？<br>**A**：😑请自行搜索查找或参考Python官方教程安装，也许你不该来到此项目。
- **Q**：为什么不支持使用TensorRT推理？<br>**A**：🤔我测试发现在onnxruntime中使用TensorRT相比cuda速度提升不大，但显存占用显著提高，因此不使用TensorRT，也可能是我使用方式不对。
- **Q**：我不想使用opencv模块，能不能使用pillow代替？<br>**A**：不好意思，作者已经习惯使用opencv了，并且部分功能使用pillow模块较难实现，之后我在简单的图像处理中尽量使用pillow替代。
- **Q**：我使用部分功能与原项目功能对比，发现速度略慢且结果有些偏差。<br>**A**：造成这个情况的原因在于，我在图像预处理步骤中调整尺寸方式使用的是INTER_CUBIC或INTER_AREA，而不是默认的INTER_LINEAR，我认为这样做通常会有更准确的结果。如果你不喜欢这个步骤，可以将预处理中调整尺寸地方做修改，速度大约能提升2-3ms，图片尺寸与模型输入尺寸的差异越大速度提升越明显。
- **Q**：如何使用显卡推理？<br>**A**：目前代码仅支持cuda推理，需要使用到Nvidia显卡，请确保电脑有至少一张Nvidia显卡，并安装显卡驱动、cuda及cudnn，再下载onnxruntime-gpu模块，创建类实例时，将use_gpu参数更改为True，如FaceDetector('models/det_500m.onnx', use_gpu=True)。
