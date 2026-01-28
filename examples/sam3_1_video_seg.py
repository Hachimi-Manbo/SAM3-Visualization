from sam3.model_builder import build_sam3_video_predictor
import cv2
import numpy as np

# 1. 初始化视频预测器
video_predictor = build_sam3_video_predictor(checkpoint_path="../models/sam3/sam3.pt")
video_path = "../resources/1.mp4"  # MP4文件或JPEG帧文件夹
text = "vehicle"  # 文本提示

# 2. 启动会话
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]

# 3. 添加文本提示（指定帧索引）
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,  # 任意帧索引
        text=text, # 文本提示
    )
)

# 4. 将提示传播到整段视频，逐帧获取输出
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

# for i, item in outputs_by_frame.items():
#     print(f"Frame {i} outputs: {item}")
#     print(outputs_by_frame[i])

# 5. 写入视频
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

def _color_from_id(obj_id: int):
    # 分配颜色
    r = (37 * obj_id + 23) % 255
    g = (17 * obj_id + 91) % 255
    b = (29 * obj_id + 47) % 255
    return int(b), int(g), int(r)  # BGR for OpenCV

for idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    results = frame.copy()
    if idx in outputs_by_frame:
        data = outputs_by_frame[idx]
        masks = data["out_binary_masks"]
        probs = data["out_probs"]
        bboxes = data["out_boxes_xywh"]
        obj_ids = data["out_obj_ids"]

        if masks is not None and len(masks) > 0:
            num_objs = masks.shape[0]
            h, w = results.shape[:2]

            for i in range(num_objs):
                score = float(probs[i]) if probs is not None else 1.0
                if score < 0.5:
                    continue

                obj_id = int(obj_ids[i]) if obj_ids is not None else i
                color = _color_from_id(obj_id)

                # 归一化的bbox
                if bboxes is not None and len(bboxes) > i:
                    x, y, bw, bh = bboxes[i]
                    x1 = int(x * w)
                    y1 = int(y * h)
                    x2 = int((x + bw) * w)
                    y2 = int((y + bh) * h)
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(0, min(w - 1, x2))
                    y2 = max(0, min(h - 1, y2))
                    cv2.rectangle(results, (x1, y1), (x2, y2), color, 2)
                    label = f"id:{obj_id} {score:.2f}"
                    cv2.putText(
                        results,
                        label,
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

                # mask 覆盖
                mask = masks[i]
                if mask is not None:
                    if mask.shape[0] != h or mask.shape[1] != w:
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    overlay = np.zeros_like(results, dtype=np.uint8)
                    overlay[mask] = color
                    results = cv2.addWeighted(results, 1.0, overlay, 0.5, 0)

    out.write(results)


cap.release()
out.release()