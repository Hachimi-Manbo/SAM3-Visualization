import torch
import numpy as np
import cv2
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model_path = "../models/sam3/sam3.pt"
image_path = "../resources/img.jpg"
text = "building,person"  # 文本提示
file_name = "1_img_seg.png"

# 1. 加载模型和处理器
model = build_sam3_image_model(checkpoint_path=model_path)
processor = Sam3Processor(model)
conf_threshold = 0.1

# 2. 加载图像并初始化推理状态
image = Image.open(image_path)
inference_state = processor.set_image(image)

# 3. 设置文本提示并推理
processor.set_confidence_threshold(conf_threshold)
output = processor.set_text_prompt(state=inference_state, prompt=text)

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

image_cv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
for i in range(masks.shape[0]):
    if scores[i] < conf_threshold:
        continue
    mask = masks[i]
    box = boxes[i].astype(int)
    color = (0, 255, 0)
    image_cv = cv2.rectangle(image_cv, (box[0], box[1]), (box[2], box[3]), color, 2)
    colored_mask = np.zeros_like(image_cv)
    colored_mask[:, :, 1] = (mask * 255).astype(np.uint8)
    image_cv = cv2.addWeighted(image_cv, 1.0, colored_mask, 0.5, 0)

    if mask.ndim == 3:
        mask = mask[0]
cv2.imwrite(file_name, cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR))