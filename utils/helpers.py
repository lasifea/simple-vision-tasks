import cv2
import numpy as np
from onnxruntime import InferenceSession, SessionOptions
from typing import Union


def read_image(img_obj: Union[str, bytes, np.ndarray]) -> np.ndarray:
    """
    读取图像
    :param img_obj: 图片对象
    :return: ndarray格式的图片对象
    """
    if isinstance(img_obj, str):
        img = cv2.imdecode(np.fromfile(img_obj, dtype=np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(img_obj, bytes):
        img = cv2.imdecode(np.frombuffer(img_obj, dtype=np.uint8), cv2.IMREAD_COLOR)
    elif isinstance(img_obj, np.ndarray):
        img = img_obj.copy()
    else:
        raise TypeError('Image object must be one of (str, bytes, np.ndarray)')

    return img


class BaseVisionTask:
    def __init__(self, model_path: str, thread_num: int = 2, use_gpu: bool = False):
        """
        视觉任务基类
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

        self._author = 'lasifea'
        self._email = 'lasifea@163.com'

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        ...
