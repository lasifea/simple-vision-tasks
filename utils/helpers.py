import cv2
import numpy as np
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
