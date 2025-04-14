# video_stretch.py
import cv2
import numpy as np

def read_video_frames(video_path):
    """读取视频所有帧，返回帧列表"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def write_video_frames(frames, output_path, fps=30):
    """将帧列表写入视频文件"""
    if len(frames) == 0:
        return
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def stretch_video_frames(frames, target_length):
    """
    将帧列表通过线性插值拉伸（或缩短）到 target_length 帧。
    采用相邻帧的加权混合方式生成中间帧。
    """
    original_length = len(frames)
    if original_length == 0:
        return []
    if original_length == target_length:
        return frames
    new_frames = []
    for i in range(target_length):
        # 计算在原始帧序列中的对应位置
        pos = i * (original_length - 1) / (target_length - 1)
        low = int(np.floor(pos))
        high = int(np.ceil(pos))
        if low == high:
            new_frames.append(frames[low])
        else:
            weight = pos - low
            # 将两帧转换为 float32 进行插值，再转换回 uint8
            frame_low = frames[low].astype(np.float32)
            frame_high = frames[high].astype(np.float32)
            frame_interp = cv2.addWeighted(frame_low, 1 - weight, frame_high, weight, 0)
            new_frames.append(frame_interp.astype(np.uint8))
    return new_frames

def stretch_videos_to_same_length(video_path1, video_path2, output_path1, output_path2, fps=30):
    """
    将两个视频拉伸或压缩到相同的帧数：
      1. 分别读取两个视频的所有帧；
      2. 取两者中较多的帧数作为目标帧数；
      3. 分别对两个视频进行帧插值，生成相同长度的新视频；
      4. 将新视频保存到 output_path1 和 output_path2。
    返回目标帧数。
    """
    frames1 = read_video_frames(video_path1)
    frames2 = read_video_frames(video_path2)
    target_length = max(len(frames1), len(frames2))
    stretched1 = stretch_video_frames(frames1, target_length)
    stretched2 = stretch_video_frames(frames2, target_length)
    write_video_frames(stretched1, output_path1, fps)
    write_video_frames(stretched2, output_path2, fps)
    return target_length
