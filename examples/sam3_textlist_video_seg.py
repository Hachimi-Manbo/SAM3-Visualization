import cv2
import numpy as np
from sam3.model_builder import build_sam3_video_predictor

model_path = "../models/sam3/sam3.pt"
video_path = "../resources/ros_output_compress.mp4"  # MP4文件或JPEG帧文件夹
text_list = ["pavement", "wall", "vehicle", "person", "plant"]  # 文本提示列表
conf_threshold = 0.5

# 1. 初始化视频预测器
video_predictor = build_sam3_video_predictor(
    checkpoint_path=model_path,
)

# 2. 读取视频帧并初始化可视化结果
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

results_frames = [f.copy() for f in frames]

def _color_from_id(obj_id: int):
    # 分配颜色
    r = (37 * obj_id + 23) % 255
    g = (17 * obj_id + 91) % 255
    b = (29 * obj_id + 47) % 255
    return int(b), int(g), int(r)  # BGR for OpenCV

# 3. 逐个提示词推理并叠加可视化结果（仅mask）
#    不需要每个提示词都重新初始化预测器，只需为每个提示词开启/关闭会话即可。
for i in range(len(text_list)):
    text = text_list[i]
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]

    video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=text,
        )
    )

    outputs_by_frame = {}
    for item in video_predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction="forward",
            start_frame_index=0,
            max_frame_num_to_track=None,
        )
    ):
        frame_index = item["frame_index"]
        outputs_by_frame[frame_index] = item["outputs"]

    for idx, results in enumerate(results_frames):
        if idx not in outputs_by_frame:
            continue
        data = outputs_by_frame[idx]
        masks = data["out_binary_masks"]
        probs = data["out_probs"]
        obj_ids = data["out_obj_ids"]

        if masks is None or len(masks) == 0:
            continue

        h, w = results.shape[:2]
        for i in range(masks.shape[0]):
            score = float(probs[i]) if probs is not None else 1.0
            print(f"Text: {text}, Frame: {idx}, Obj: {i}, Score: {score}")
            if score < conf_threshold:
                continue
            obj_id = int(obj_ids[i]) if obj_ids is not None else i
            color = _color_from_id(obj_id)
            mask = masks[i]
            if mask is None:
                continue
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            overlay = np.zeros_like(results, dtype=np.uint8)
            overlay[mask] = color
            results_frames[idx] = cv2.addWeighted(results_frames[idx], 1.0, overlay, 0.5, 0)

    # 释放本提示词对应的会话，避免显存堆积
    video_predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

# 4. 写入视频
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (results_frames[0].shape[1], results_frames[0].shape[0]))
for frame in results_frames:
    out.write(frame)
out.release()