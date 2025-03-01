import cv2
import mediapipe as mp
import math
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class Config:
    KEY_ANGLES = {
        'left_elbow': [11,13,15],
        'right_elbow': [12,14,16],
        'body_angle': [11,23,25],
        'hip_alignment': [11,23,27]
    }
    NORMALIZATION_JOINTS = [11,12,23]

class PoseDetector:
    def __init__(self, mode=False, smooth=True, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detection_con = detection_con
        self.track_con = track_con
        self.lm_list = []
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )

    def find_pose(self, img, draw=True):
        # 给视频帧标注关键点
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img, self.results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img):
        self.lm_list = []
        self.results = self.pose.process(img)
        if self.results.pose_landmarks:
            for idx, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w = img.shape[:2]
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([idx, cx, cy])
                # if draw:
                #     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lm_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        if len(self.lm_list) < max(p1, p2, p3) + 1:
            return 0

        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        x3, y3 = self.lm_list[p3][1:]

        vector1 = (x1 - x2, y1 - y2)
        vector2 = (x3 - x2, y3 - y2)

        # 使用atan2计算带方向的角度
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        angle = math.degrees(math.atan2(cross_product, dot_product))
        angle = abs(angle)  # 取绝对值表示0-180度范围

        if draw:
            # 绘制连线
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            # 绘制关键点
            for x, y in [(x1, y1), (x2, y2), (x3, y3)]:
                cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x, y), 15, (0, 0, 255), 2)
            # 显示角度
            cv2.putText(img, f"{int(angle)}°", (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return angle
    
class VideoAnalyzer(PoseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_angles = Config.KEY_ANGLES

    def process_video(self, video_path, skip_frames=2):
        """提取视频中的姿势特征序列"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{video_path}")
            
        sequence = []  # 用于存储所有帧的数据
        frame_count = 0

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break
            
            frame_count += 1
                
            lm_list = self.find_position(img)
            angles = self._get_frame_angles(img)
            
            frame_data = {
                'landmarks': lm_list,
                'angles': angles,
                'norm_landmarks': self._normalize_landmarks(lm_list)
            }
            sequence.append(frame_data)
            
        cap.release()
        return sequence  # 返回完整的序列数据

    def _normalize_landmarks(self, lm_list):
        """以肩宽为归一化基准"""

        if len(lm_list) < 25 or not all(k in [lm[0] for lm in lm_list] for k in Config.NORMALIZATION_JOINTS):
            return []
            
        # 使用双肩宽度作为基准
        shoulder_left = lm_list[11][1:]
        shoulder_right = lm_list[12][1:]
        shoulder_width = math.dist(shoulder_left, shoulder_right)
        
        # 以左肩为原点
        origin = np.array(shoulder_left)
        
        normalized = []
        for lm in lm_list:
            # 平移并缩放
            x = (lm[1] - origin[0]) / shoulder_width
            y = (lm[2] - origin[1]) / shoulder_width
            normalized.append([x, y])
            
        return normalized
    
    def _get_frame_angles(self, img):
        return {k: self.find_angle(img, *v, draw=False) for k, v in self.key_angles.items()}

    def generate_report(self, result):
        """生成可视化对比报告"""
        # 只在需要时才导入 matplotlib
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # 角度对比
        plt.subplot(2, 1, 1)
        for joint in self.std_seq[0]['angles']:
            std_angles = [f['angles'][joint] for f in self.std_seq]
            pat_angles = [f['angles'].get(joint, 0) for f in self.pat_seq]
            plt.plot(std_angles, label=f'Std {joint}')
            plt.plot(pat_angles, linestyle='--', label=f'Pat {joint}')
        plt.title('Joint Angle Comparison')
        plt.legend()
        
        # DTW路径可视化
        plt.subplot(2, 1, 2)
        std_idx = [i for i, j in result['alignment_path']]
        pat_idx = [j for i, j in result['alignment_path']]
        plt.plot(pat_idx, std_idx, 'r-', alpha=0.3)
        plt.xlabel('Patient Frame Index')
        plt.ylabel('Standard Frame Index')
        plt.title('DTW Alignment Path')
        
        plt.tight_layout()
        plt.savefig('comparison_report.jpg')
        plt.close()
        
        return 'comparison_report.jpg'
    
    def _compare_frames_with_multiple_matches(self, path):
        """计算患者视频帧与多个标准视频帧的对比评分，并进行百分制标准化"""
        frame_scores = []
        
        # 创建一个字典，存储每个患者帧对应的标准帧
        patient_to_standard = {}
        for i, j in path:
            if j not in patient_to_standard:
                patient_to_standard[j] = []
            patient_to_standard[j].append(i)  # 将标准帧索引添加到患者帧索引下
        
        # 遍历每个患者帧，计算其与对应多个标准帧的评分均值
        for patient_frame_idx, standard_frame_idxs in patient_to_standard.items():
            patient_frame = self.pat_seq[patient_frame_idx]
            patient_features = np.concatenate([np.array(patient_frame['norm_landmarks']).flatten(),
                                            np.array(list(patient_frame['angles'].values()))])
            
            scores = []
            for standard_frame_idx in standard_frame_idxs:
                standard_frame = self.std_seq[standard_frame_idx]
                standard_features = np.concatenate([np.array(standard_frame['norm_landmarks']).flatten(),
                                                    np.array(list(standard_frame['angles'].values()))])
                
                # 计算患者帧与标准帧之间的特征差异（欧几里得距离）
                score = np.linalg.norm(patient_features - standard_features)
                scores.append(score)
            
            # 计算患者帧与多个标准帧的评分均值
            avg_score = np.mean(scores)
            frame_scores.append((patient_frame_idx, avg_score))
        
        # 获取最大值和最小值
        all_scores = [score for _, score in frame_scores]
        min_score = min(all_scores)
        max_score = max(all_scores)
        
        # 标准化
        normalized_scores = []
        for patient_frame_idx, score in frame_scores:
            normalized_score = 100 * (max_score - score) / (max_score - min_score) if max_score != min_score else 100
            normalized_scores.append((patient_frame_idx, normalized_score))
        
        return normalized_scores


# def main():
#     analyzer = VideoAnalyzer()
    
#     # 处理标准俯卧撑视频
#     std_video = analyzer.process_video("PoseVideos\standard.mp4")
    
#     # 处理患者视频demo2.mp4
#     pat_video = analyzer.process_video("PoseVideos\demo2.mp4")
    
#     comparator = ActionComparator(std_video, pat_video)
#     result = comparator.compare_sequences()
#     print(result['dtw_distance'])
#     print(f"动作相似度评分: {250 / (result['dtw_distance']+125):.2%}")
    
#     # 生成报告
#     report_img = comparator.generate_report(result)
    
#     # 俯卧撑改进建议
#     pushup_advice = {
#         'left_elbow': "肘关节活动范围不足，建议加大下沉深度（标准角度：90°±10°）",
#         'right_elbow': "左右肘不对称，注意保持双臂对称运动",
#         'body_angle': "躯干稳定性不足，保持身体成直线（标准角度：170°-180°）",
#         'hip_alignment': "臀部下沉不足/过度抬起，保持髋部与肩部平行"
#     }
    
#     print("\n俯卧撑动作改进建议：")
#     for joint, advice in pushup_advice.items():
#         try:
#             std_avg = np.mean([f['angles'][joint] for f in std_video if f['angles'][joint] > 0])
#             pat_avg = np.mean([f['angles'].get(joint,0) for f in pat_video])
#             print(f"- {joint}: {advice}\n 标准角度：{std_avg:.1f}° vs 患者角度：{pat_avg:.1f}°")
#         except KeyError:
#             continue

class ActionComparator:
    def __init__(self, std_sequence, pat_sequence):
        self.std_seq = std_sequence
        self.pat_seq = pat_sequence
        self.distance_matrix = None
        self.scaler = StandardScaler()

    def compare_sequences(self):
        """返回对齐后的相似度分析"""
        # 特征向量构建
        std_features = self._extract_features(self.std_seq)
        pat_features = self._extract_features(self.pat_seq)
        
        # 标准化特征
        std_features = self.scaler.fit_transform(std_features)
        pat_features = self.scaler.transform(pat_features)

        # FastDTW对齐
        distance, path = fastdtw(std_features, pat_features, dist=euclidean)
        # print(len(path))
        # 对齐路径分析
        aligned_std = [std_features[i] for i, j in path]
        aligned_pat = [pat_features[j] for i, j in path]

        # print(len(self.std_seq))
        # print(len(self.pat_seq))
        # 计算标准化相似度
        max_length = max(len(self.std_seq), len(self.pat_seq))
        normalized_distance = distance / max_length
        similarity = 1 / (1 + normalized_distance)
        
        # 计算患者视频帧与多个标准视频帧的对比评分
        frame_scores = self._compare_frames_with_multiple_matches(path)
        # print(len(frame_scores))
        print(frame_scores)
        return {
            'dtw_distance': distance,
            'alignment_path': path,
            'aligned_std': aligned_std,
            'aligned_pat': aligned_pat,
            'similarity_score': similarity,
            'frame_scores': frame_scores
        }
    
    def _extract_features(self, sequence):
        """构建时空特征向量：[norm_x, norm_y, angle1, angle2,...]"""
        features = []
        for frame in sequence:
            # 关节坐标（使用归一化后的数据）
            coord_feat = np.array(frame['norm_landmarks']).flatten()
            
            # 角度特征
            angle_feat = np.array(list(frame['angles'].values()))
            
            features.append(np.concatenate([coord_feat, angle_feat]))
            
        return np.array(features)
    
    def generate_report(self, result):
        """生成可视化对比报告"""
        plt.figure(figsize=(15, 10))
        
        # 角度对比
        plt.subplot(2, 1, 1)
        for joint in self.std_seq[0]['angles']:
            std_angles = [f['angles'][joint] for f in self.std_seq]
            pat_angles = [f['angles'].get(joint, 0) for f in self.pat_seq]
            plt.plot(std_angles, label=f'Std {joint}')
            plt.plot(pat_angles, linestyle='--', label=f'Pat {joint}')
        plt.title('Joint Angle Comparison')
        plt.legend()
        
        # DTW路径可视化
        plt.subplot(2, 1, 2)
        std_idx = [i for i, j in result['alignment_path']]
        pat_idx = [j for i, j in result['alignment_path']]
        plt.plot(pat_idx, std_idx, 'r-', alpha=0.3)
        plt.xlabel('Patient Frame Index')
        plt.ylabel('Standard Frame Index')
        plt.title('DTW Alignment Path')
        
        plt.tight_layout()
        plt.savefig('comparison_report.jpg')
        plt.close()
        
        return 'comparison_report.jpg'
    
    def _compare_frames_with_multiple_matches(self, path):
        """计算患者视频帧与多个标准视频帧的对比评分，并进行百分制标准化"""
        frame_scores = []
        
        # 创建一个字典，存储每个患者帧对应的标准帧
        patient_to_standard = {}
        for i, j in path:
            if j not in patient_to_standard:
                patient_to_standard[j] = []
            patient_to_standard[j].append(i)  # 将标准帧索引添加到患者帧索引下
        
        # 遍历每个患者帧，计算其与对应多个标准帧的评分均值
        for patient_frame_idx, standard_frame_idxs in patient_to_standard.items():
            patient_frame = self.pat_seq[patient_frame_idx]
            patient_features = np.concatenate([np.array(patient_frame['norm_landmarks']).flatten(),
                                            np.array(list(patient_frame['angles'].values()))])
            
            scores = []
            for standard_frame_idx in standard_frame_idxs:
                standard_frame = self.std_seq[standard_frame_idx]
                standard_features = np.concatenate([np.array(standard_frame['norm_landmarks']).flatten(),
                                                    np.array(list(standard_frame['angles'].values()))])
                
                # 计算患者帧与标准帧之间的特征差异（欧几里得距离）
                score = np.linalg.norm(patient_features - standard_features)
                scores.append(score)
            
            # 计算患者帧与多个标准帧的评分均值
            avg_score = np.mean(scores)
            frame_scores.append((patient_frame_idx, avg_score))
        
        # 获取最大值和最小值
        all_scores = [score for _, score in frame_scores]
        min_score = min(all_scores)
        max_score = max(all_scores)
        
        # 标准化
        normalized_scores = []
        for patient_frame_idx, score in frame_scores:
            normalized_score = 100 * (max_score - score) / (max_score - min_score) if max_score != min_score else 100
            normalized_scores.append((patient_frame_idx, normalized_score))
        
        return normalized_scores
