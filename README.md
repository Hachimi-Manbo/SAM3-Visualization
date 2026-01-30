# SAM3-Visualization

基于 SAM3 官方 Demo 的可视化与推理示例仓库，支持图像与视频分割结果的可视化输出（仅掩码叠加，不绘制 BBox）。

## 目录结构

- examples/：示例脚本
- models/：模型权重与配置（本仓库已放置）
- resources/：输入视频/图片资源（请自行放置）

## 环境准备

参见sam3官方仓库。

## 模型与资源

- 模型权重位于：models/sam3/sam3.pt
- 资源文件放在：resources/

## 快速开始

### 1) 图像分割

运行 [examples/sam3_img_seg.py](examples/sam3_img_seg.py)：

```bash
python examples/sam3_img_seg.py
```

### 2) 视频分割（单提示词）

运行 [examples/sam3_video_seg.py](examples/sam3_video_seg.py)：

```bash
python examples/sam3_video_seg.py
```

### 3) 视频分割（多提示词顺序叠加）

运行 [examples/sam3_textlist_video_seg.py](examples/sam3_textlist_video_seg.py)：

```bash
python examples/sam3_textlist_video_seg.py
```

该脚本会对 `text_list` 中的提示词逐个推理，并将掩码结果叠加在同一输出视频上，以降低显存峰值占用。

## 输入与输出说明

- 输入视频：resources/ 目录下的 mp4 文件或视频帧目录
- 输出视频：默认保存为 output_video.mp4（脚本内可修改）

## 常见问题

**Q：提示词过多导致显存溢出怎么办？**

A：请使用多提示词顺序推理脚本（text list 逐个推理），每个提示词使用独立会话，推理完成后关闭会话并叠加掩码结果。

## 备注

本仓库依赖上层 SAM3 代码与模型接口，若需调整模型加载与推理逻辑，请参考主仓库中的相关实现。
