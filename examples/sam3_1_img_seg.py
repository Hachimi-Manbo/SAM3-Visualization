import torch
import numpy as np
import cv2
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 1. 加载模型和处理器
model = build_sam3_image_model(checkpoint_path="../models/sam3/sam3.pt")
processor = Sam3Processor(model)

# 2. 加载图像并初始化推理状态
image = Image.open("../resources/img.jpg")
inference_state = processor.set_image(image)

# 3. 设置文本提示并推理
output = processor.set_text_prompt(state=inference_state, prompt="person")

# 4. 提取结果：掩码、边界框（xyxy格式）、置信度分数
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# 5. 打印结果形状和内容
print("Masks shape:", masks.shape)
print("Boxes:", boxes)
print("Scores:", scores)

# 6. 可视化
masks = masks.cpu().numpy()
boxes = boxes.cpu().numpy()
scores = scores.cpu().numpy()

filename = "1_img_seg.png"
image_cv = cv2.cvtColor(cv2.imread("../resources/img.jpg"), cv2.COLOR_BGR2RGB)
for i in range(masks.shape[0]):
    if scores[i] < 0.5:
        continue
    mask = masks[i]
    box = boxes[i].astype(int)
    color = (0, 255, 0)
    image_cv = cv2.rectangle(image_cv, (box[0], box[1]), (box[2], box[3]), color, 2)
    colored_mask = np.zeros_like(image_cv)
    colored_mask[:, :, 1] = (mask * 255).astype(np.uint8)
    image_cv = cv2.addWeighted(image_cv, 1.0, colored_mask, 0.5, 0)

cv2.imwrite(filename, cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR))