import os
import shutil
from pathlib import Path

import yaml
import numpy as np
import cv2
import argparse
import torch


# 色盘，可根据类别添加新颜色，注意第一个别动，这是字体的颜色也就是白色，往后面推
class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

def read_classes_from_yaml(file_path):
    try:
        # 打开并读取 YAML 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            # 提取类别信息
            if 'names' in data:
                return data['names']
            else:
                print("YAML 文件中未找到 'names' 字段。")
                return []
    except FileNotFoundError:
        print(f"未找到文件: {file_path}")
        return []
    except yaml.YAMLError as e:
        print(f"解析 YAML 文件时出错: {e}")
        return []


# 坐标转换
def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def box_label(img, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False):
    """Add one xyxy box to image with label."""
    if isinstance(box, torch.Tensor):
        box = box.tolist()
    
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            0.5,
            txt_color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return img


def draw_gt_boxes(imgPth, labelPth, output_folder, cls):
    """
    绘制 ground truth 目标框
    """
    # 验证图像和标签文件是否存在
    if not os.path.exists(imgPth):
        print("图像文件不存在！")
        return
    if not os.path.exists(labelPth):
        print("标签文件不存在！")
        return
    
    # 读取图像
    img = cv2.imread(imgPth)
    imgH, imgW = img.shape[:2]
    
    # 读取标签
    with open(labelPth, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # 读取类别
    classes = read_classes_from_yaml(cls)
    
    # 绘制目标框
    for line in lines:
        label, x, y, w, h = line.split()
        x, y, w, h = float(x), float(y), float(w), float(h)
        boxes = xywh2xyxy(np.array([x, y, w, h]))
        
        # 反归一化
        boxes = boxes * [imgW, imgH, imgW, imgH]
        boxes = boxes.astype(np.int32)
        
        # 绘制矩形框
        color = colors(int(label))
        box_label(img, boxes, classes[int(label)], color, rotated=False)
    
    # 保存图像
    cv2.imwrite(output_folder + '/' + imgPth.split('\\')[-1], img)
    


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type = str, default = 'H:\\Dataset\\NEU-DET\\yolo_dataset\\val\\images\\inclusion_93.jpg', help = '数据目录')
    parser.add_argument('--label', type = str, default = 'H:\\Dataset\\NEU-DET\\yolo_dataset\\val\\labels\\inclusion_93.txt', help = 'labels目录')
    parser.add_argument('--cls', type = str, default = 'ultralytics\\cfg\\datasets\\NEU-DET.yaml', help = '类别')
    parser.add_argument('--save', type = str, default = 'H:\\ultralytics\\save', help = '保存目录')
    args = parser.parse_args()


    # 绘制GT box
    draw_gt_boxes(args.img, args.label, args.save, args.cls)

