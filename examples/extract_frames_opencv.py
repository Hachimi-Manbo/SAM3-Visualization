import os
import cv2

def extract_frames(video_path: str, output_dir: str, max_digits: int = 6) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    idx = 1
    max_index = 10 ** max_digits - 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx > max_index:
            raise RuntimeError(f"帧数超过 {max_digits} 位上限: {max_index}")

        filename = f"{idx:0{max_digits}d}.png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, frame)
        idx += 1

    cap.release()


if __name__ == "__main__":
    video_path = "/home/wsl2/sam3/test58/SAM3-Visualization/resources/ros_output.mp4"
    output_dir = "/home/wsl2/sam3/test58/SAM3-Visualization/resources/ros_output_frames"
    extract_frames(video_path, output_dir)
