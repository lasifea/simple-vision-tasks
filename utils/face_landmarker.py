# 本代码由lasifea编写，参考文档地址：https://github.com/deepinsight/insightface，模型来自insightface
# 适用于insightface开源人脸关键点识别的onnx模型做推理
# 首次完成时间于2024-06-11，最后修改时间于2024-06-17

from onnxruntime import InferenceSession, SessionOptions
import cv2
import numpy as np
from math import ceil
from typing import Union, Any, Tuple
from .helpers import read_image


class Face2DLandmarker:
    def __init__(self,
                 model_path: str,
                 thread_num: int = 2,
                 use_gpu: bool = False):
        """
        人脸关键点检测器
        :param model_path: onnx模型文件路径
        :param thread_num: 线程数量，默认2个线程
        :param use_gpu: 是否使用显卡推理，目前仅支持cuda
        """
        if use_gpu:
            import os
            os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
            providers = ('CUDAExecutionProvider', 'CPUExecutionProvider')
        else:
            providers = ('CPUExecutionProvider',)

        options = SessionOptions()
        options.log_severity_level = 3
        if thread_num > 0:
            options.intra_op_num_threads = thread_num

        self.model = InferenceSession(model_path, providers=providers, sess_options=options)
        self.input_name = self.model.get_inputs()[0].name

        self._box_side = 192
        self._scale = 1.5
        self._input_mean = 0
        self._input_std = 1

        self._author = 'lasifea'
        self._email = 'lasifea@163.com'

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _preprocess(self,
                    img_obj: Union[str, bytes, np.ndarray],
                    face_box: Any) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        图像预处理，将人脸区域等比例缩放，再从原图中截取
        :param img_obj: 图片对象
        :param face_box: 人像区域，xyxy格式
        :return: 返回三个参数，第一个参数为处理后的图片，第二个参数为脸部图片的缩放倍数，第三个参数为脸部图片在原图中的偏移坐标
        """
        img = read_image(img_obj)
        h, w = img.shape[:2]
        face_w, face_h = face_box[2] - face_box[0], face_box[3] - face_box[1]
        max_side = self._scale * max(face_w, face_h)

        offset_w = (max_side - face_w) / 2
        offset_h = (max_side - face_h) / 2
        x, y, x2, y2 = face_box[0] - offset_w, face_box[1] - offset_h, face_box[2] + offset_w, face_box[3] + offset_h

        out_x, out_y, out_x2, out_y2 = 0, 0, 0, 0
        if x < 0:
            out_x = -x
            x = 0
        if x2 > w:
            out_x2 = x2 - w
        if y < 0:
            out_y = -y
            y = 0
        if y2 > h:
            out_y2 = y2 - h

        face_img = img[round(y):round(y2), round(x):round(x2)]
        h, w = face_img.shape[:2]

        mask = np.zeros((ceil(max_side), ceil(max_side), 3), dtype=np.uint8)
        start_x = int((out_x + out_x2) / 2)
        start_y = int((out_y + out_y2) / 2)
        mask[start_y:start_y + h, start_x:start_x + w] = face_img
        offset_point = (x - start_x, y - start_y)

        scale = self._box_side / mask.shape[0]
        img2 = cv2.resize(mask, (self._box_side, self._box_side), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        # img2 = cv2.resize(img, (self._box_side, self._box_side))

        return img2, scale, offset_point

    @staticmethod
    def _postprocess(output_tensor: np.ndarray, scale: float, offset_point: Tuple[float, float]) -> np.ndarray:
        """
        后处理，计算人脸关键点位置
        :param output_tensor: 模型推理结果
        :param scale: 原图的缩放比例
        :param offset_point: 脸部图片在原图中的偏移坐标
        :return: 返回106个关键点在原图中的位置
        """
        points = output_tensor.reshape((-1, 2))
        points += 1
        points *= 96
        points /= scale
        points += offset_point
        return points

    def forward(self, img_obj: Union[str, bytes, np.ndarray], face_box: Any) -> np.ndarray:
        """
        输入图像得到人脸关键点位置
        :param img_obj: 图片对象
        :param face_box: 人像区域，xyxy格式
        :return: 返回106个关键点在原图中的位置
        """
        input_img, scale, offset_point = self._preprocess(img_obj, face_box)
        input_tensor = input_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)[np.newaxis, ...]

        output_tensor = self.model.run(None, {self.input_name: input_tensor})[0][0]
        return self._postprocess(output_tensor, scale, offset_point)
