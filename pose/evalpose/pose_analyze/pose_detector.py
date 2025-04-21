# pose_detector.py
import cv2
import mediapipe as mp
import math
import numpy as np

class PoseDetector:
    def __init__(self, mode=False, smooth=True, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.results = None

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img, self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        self.lm_list = []
        if self.results.pose_landmarks:
            for idx, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w = img.shape[:2]
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([idx, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lm_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        if len(self.lm_list) < max(p1, p2, p3) + 1:
            return 0

        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        x3, y3 = self.lm_list[p3][1:]

        vector1 = (x1 - x2, y1 - y2)
        vector2 = (x3 - x2, y3 - y2)

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        angle = math.degrees(math.atan2(cross_product, dot_product))
        angle = abs(angle)  # 取绝对值表示0-180度范围

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            for x, y in [(x1, y1), (x2, y2), (x3, y3)]:
                cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x, y), 15, (0, 0, 255), 2)
            cv2.putText(img, f"{int(angle)}°", (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return angle


class VideoAnalyzer(PoseDetector):
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        # Use provided config or import default if none provided
        if config is None:
            from .config import Config
            self.key_angles = Config.KEY_ANGLES
            self.normalization_joints = Config.NORMALIZATION_JOINTS
        else:
            self.key_angles = config.KEY_ANGLES
            self.normalization_joints = getattr(config, 'NORMALIZATION_JOINTS', [11, 12, 23])
        
        # Store the config object for other methods to access
        self.config = config

    def process_video(self, video_path, skip_frames=2):
        """
        提取视频中的姿势特征序列
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{video_path}")
        sequence = []
        frame_count = 0

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            # 如果需要跳帧，可取消注释下面的代码
            # if frame_count % (skip_frames+1) != 0:
            #     frame_count += 1
            #     continue

            self.find_pose(img)
            lm_list = self.find_position(img, draw=False)
            angles = self._get_frame_angles(img)

            frame_data = {
                'landmarks': lm_list,
                'angles': angles,
                'norm_landmarks': self._normalize_landmarks(lm_list)
            }
            sequence.append(frame_data)
            frame_count += 1

        cap.release()
        return sequence

    def _normalize_landmarks(self, lm_list):
        """
        使用新的归一化方法，参考 Compare_pose.py 中的 l2_normalize 算法：
        1. 根据所有关键点计算包围盒（box = [min_x, min_y, max_x, max_y]）
        2. 计算 temp_x = (max_x - min_x)/2, temp_y = (max_y - min_y)/2
        3. 若 temp_x <= temp_y，则 sub_x = min_x - (temp_y - temp_x), sub_y = min_y；
           否则，sub_x = min_x, sub_y = min_y - (temp_x - temp_y)
        4. 对每个关键点，计算偏移坐标 (x - sub_x, y - sub_y)，并将所有偏移坐标组成一维向量，
           计算该向量的 L2 范数 norm_val，再除以 norm_val得到归一化后的坐标。
        """
        if len(lm_list) == 0:
            return []
        
        # 对于并非所有关键点存在的情况，返回空列表
        if not all(k in [lm[0] for lm in lm_list] for k in self.normalization_joints):
            return []

        # 计算包围盒
        xs = [lm[1] for lm in lm_list]
        ys = [lm[2] for lm in lm_list]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        # 计算半宽和半高
        temp_x = (max_x - min_x) / 2
        temp_y = (max_y - min_y) / 2

        # 按照 Compare_pose.py 的处理（这里 box[0]=min_x, box[1]=min_y, box[2]=max_x, box[3]=max_y，且 min_x<=max_x, min_y<=max_y）
        if temp_x <= temp_y:
            sub_x = min_x - (temp_y - temp_x)
            sub_y = min_y
        else:
            sub_x = min_x
            sub_y = min_y - (temp_x - temp_y)

        # 对每个关键点，计算相对于 sub_x, sub_y 的偏移
        normalized_coords = []
        for lm in lm_list:
            norm_x = lm[1] - sub_x
            norm_y = lm[2] - sub_y
            normalized_coords.append([norm_x, norm_y])
        # 将所有坐标平铺成一维向量，计算 L2 范数
        flat_coords = [coord for pair in normalized_coords for coord in pair]
        norm_val = np.linalg.norm(flat_coords)
        if norm_val == 0:
            return normalized_coords
        # 对每个关键点进行归一化
        normalized = [[x / norm_val, y / norm_val] for (x, y) in normalized_coords]
        return normalized

    def _get_frame_angles(self, img):
        return {k: self.find_angle(img, *v, draw=False) for k, v in self.key_angles.items()}
    
    # Adding the draw_bone function from visualization
    def draw_bone(self, img, landmarks, connections, color=(0, 255, 0)):
        """
        绘制骨架连接
        :param img: 图像
        :param landmarks: 关节点坐标列表
        :param connections: 连接关节点的索引对列表
        :param color: 绘制颜色
        """
        for connection in connections:
            x1, y1 = landmarks[connection[0]]
            x2, y2 = landmarks[connection[1]]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    # Adding the function to process overlap video
    def _process_overlap_video(self, std_video, pat_video, dtw_result, output_video_path, video_path_pat, save_lowest_scores=True,config=None):
        """
        生成带有患者骨架和标准骨架的视频（标准骨架通过平移与患者肩关节对齐）。
        
        :param std_video: 标准视频的骨架数据
        :param pat_video: 患者视频的骨架数据
        :param dtw_result: DTW对齐结果
        :param output_video_path: 输出视频路径
        :param video_path_pat: 患者视频路径
        :param save_lowest_scores: 是否保存得分最低的帧
        """
        try:
            from .evaluation import detect_action_stages, select_lowest_score_frames
            import os
            
            # 自动识别动作阶段
            stages = detect_action_stages(pat_video)
            
            # 获取每个阶段的最低评分帧
            if save_lowest_scores:
                lowest_score_frames = select_lowest_score_frames(dtw_result, stages)
            
            # 打开患者视频文件
            cap_pat = cv2.VideoCapture(video_path_pat)
            
            if not cap_pat.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path_pat}")
                    
            # 视频编写器设置
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = cap_pat.get(cv2.CAP_PROP_FPS)
            width = int(cap_pat.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_pat.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # 构建患者帧索引到标准帧索引的映射
            pat_to_std = {}
            for std_idx, pat_idx in dtw_result['alignment_path']:
                if pat_idx not in pat_to_std:
                    pat_to_std[pat_idx] = std_idx
            
            frame_idx = 0
            idx = 1
            
            while cap_pat.isOpened():
                success_pat, img_pat = cap_pat.read()
                
                if not success_pat:
                    break
                
                # 获取当前帧的患者骨架数据
                pat_frame = pat_video[frame_idx]
                
                # 尝试获取对应的标准帧索引
                if frame_idx in pat_to_std:
                    std_frame_idx = pat_to_std[frame_idx]
                else:
                    available_indices = sorted(pat_to_std.keys())
                    candidates = [idx for idx in available_indices if idx <= frame_idx]
                    if candidates:
                        std_frame_idx = pat_to_std[candidates[-1]]
                    elif available_indices:
                        std_frame_idx = pat_to_std[available_indices[0]]
                    else:
                        std_frame_idx = 0  # 默认取第0帧

                # 检查索引是否有效
                if std_frame_idx >= len(std_video):
                    # 避免索引越界
                    std_frame_idx = len(std_video) - 1
                    
                std_frame = std_video[std_frame_idx]
                
                # 获取患者骨架坐标
                pat_landmarks = [(lm[1], lm[2]) for lm in pat_frame['landmarks']]
                
                # 只绘制与角度相关的骨架
                for angle_name, joints in self.key_angles.items():
                    # 获取每个夹角的三个关节
                    p1, p2, p3 = joints
                    # 绘制骨架连接（线段连接）
                    self.draw_bone(img_pat, pat_landmarks, [(p1, p2), (p2, p3)], (0, 0, 255))  # 红色线条
                
                # 获取标准骨架坐标并进行平移
                pat_shoulder_left = np.array(pat_frame['landmarks'][11][1:])
                pat_shoulder_right = np.array(pat_frame['landmarks'][12][1:])
                shoulder_width_pat = np.linalg.norm(pat_shoulder_left - pat_shoulder_right)
                
                std_landmarks = [(lm[1], lm[2]) for lm in std_frame['landmarks']]
                
                # 获取患者左肩和标准左肩坐标
                std_shoulder_left = np.array(std_frame['landmarks'][11][1:])
                
                # 计算平移量（使标准骨架的左肩与患者的左肩对齐）
                translation_vector = pat_shoulder_left - std_shoulder_left
                
                # 平移标准骨架
                std_landmarks_translated = []
                for x, y in std_landmarks:
                    x_translated = x + translation_vector[0]
                    y_translated = y + translation_vector[1]
                    std_landmarks_translated.append((x_translated, y_translated))
                
                # 绘制平移后的标准骨架
                for angle_name, joints in self.key_angles.items():
                    # 获取每个夹角的三个关节
                    p1, p2, p3 = joints
                    # 绘制标准骨架的线条（绿色）
                    self.draw_bone(img_pat, std_landmarks_translated, [(p1, p2), (p2, p3)], color=(0, 255, 0))  # 绿色线条
                
                # 保存得分最低的帧
                if save_lowest_scores:
                    if frame_idx in [frame[0] for frame in lowest_score_frames]:
                        img_name = os.path.join(os.path.dirname(output_video_path), f'patient_frame_{idx}.jpg')
                        idx += 1
                        cv2.imwrite(img_name, img_pat)
                        print(f"保存最低得分的患者帧：{img_name}")
                
                # 写入输出视频
                out.write(img_pat)
                
                frame_idx += 1
            
            cap_pat.release()
            out.release()
        except Exception as e:
            print(f"Error in _process_overlap_video: {str(e)}")
            raise

    # Adding enhanced version of generate_video_with_selected_frames compatible with dynamic config
    def generate_video_with_selected_frames(self, std_video, pat_video, dtw_result, output_video_path, video_path_pat, stages, config=None, save_lowest_scores=True):
        """
        生成包含骨架对比的输出视频，同时保存最低得分帧的图像
        :param std_video: 标准视频的帧数据序列
        :param pat_video: 患者视频的帧数据序列
        :param dtw_result: DTW 对齐结果
        :param output_video_path: 输出视频保存路径
        :param video_path_pat: 患者视频文件路径
        :param stages: 动作阶段列表
        :param config: 配置对象，包含KEY_ANGLES等参数
        :param save_lowest_scores: 是否保存最低得分帧
        """
        from .evaluation import select_lowest_score_frames
        import os
        
        # Use provided config or instance config
        if config is None:
            config = self.config
        
        key_angles = config.KEY_ANGLES if config else self.key_angles
        
        lowest_score_frames = select_lowest_score_frames(dtw_result, stages)

        cap_pat = cv2.VideoCapture(video_path_pat)
        if not cap_pat.isOpened():
            raise ValueError("无法打开视频文件")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap_pat.get(cv2.CAP_PROP_FPS)
        width = int(cap_pat.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_pat.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # 构建患者帧索引到标准帧索引的映射
        pat_to_std = {}
        for std_idx, pat_idx in dtw_result['alignment_path']:
            if pat_idx not in pat_to_std:
                pat_to_std[pat_idx] = std_idx

        frame_idx = 0
        idx = 1
        
        while cap_pat.isOpened():
            success_pat, img_pat = cap_pat.read()
            if not success_pat:
                break

            # 尝试获取对应的标准帧索引
            if frame_idx < len(pat_video):
                pat_frame = pat_video[frame_idx]
            else:
                break  # 如果超出患者视频帧数，则退出循环
                
            # 尝试获取对应的标准帧索引
            if frame_idx in pat_to_std:
                std_frame_idx = pat_to_std[frame_idx]
            else:
                available_indices = sorted(pat_to_std.keys())
                candidates = [idx for idx in available_indices if idx <= frame_idx]
                if candidates:
                    std_frame_idx = pat_to_std[candidates[-1]]
                elif available_indices:
                    std_frame_idx = pat_to_std[available_indices[0]]
                else:
                    std_frame_idx = 0  # 默认取第0帧

            if std_frame_idx >= len(std_video):
                std_frame_idx = len(std_video) - 1  # 避免越界
            std_frame = std_video[std_frame_idx]

            pat_landmarks = [(lm[1], lm[2]) for lm in pat_frame['landmarks']]

            # 绘制患者骨架（红色）
            for angle_name, joints in key_angles.items():
                p1, p2, p3 = joints
                self.draw_bone(img_pat, pat_landmarks, [(p1, p2), (p2, p3)], color=(0, 0, 255))

            pat_shoulder_left = np.array(pat_frame['landmarks'][11][1:])
            std_landmarks = [(lm[1], lm[2]) for lm in std_frame['landmarks']]

            # 将标准骨架平移对齐到患者的左肩位置
            std_shoulder_left = np.array(std_frame['landmarks'][11][1:])
            translation_vector = pat_shoulder_left - std_shoulder_left
            std_landmarks_translated = []
            for x, y in std_landmarks:
                x_translated = x + translation_vector[0]
                y_translated = y + translation_vector[1]
                std_landmarks_translated.append((x_translated, y_translated))

            # 绘制标准骨架（绿色）
            for angle_name, joints in key_angles.items():
                p1, p2, p3 = joints
                self.draw_bone(img_pat, std_landmarks_translated, [(p1, p2), (p2, p3)], color=(0, 255, 0))

            # 保存最低得分帧
            if save_lowest_scores and frame_idx in [frame[0] for frame in lowest_score_frames]:
                img_name = os.path.join(os.path.dirname(output_video_path), f'patient_frame_{idx}.jpg')
                idx += 1
                cv2.imwrite(img_name, img_pat)
                print(f"保存最低得分的患者帧：{img_name}")

            out.write(img_pat)
            frame_idx += 1

        cap_pat.release()
        out.release()
