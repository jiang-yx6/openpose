# action_comparator.py
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class ActionComparator:
    def __init__(self, std_sequence, pat_sequence):
        self.std_seq = std_sequence
        self.pat_seq = pat_sequence
        self.distance_matrix = None
        self.scaler = StandardScaler()

    def compare_sequences(self):
        """
        使用整体特征向量的 DTW 对齐计算，返回对齐后的相似度分析结果，
        包括 DTW 距离、对齐路径、相似度评分以及每帧得分（使用新的帧匹配算法）。
        """
        std_features = self._extract_features(self.std_seq)
        pat_features = self._extract_features(self.pat_seq)

        std_features = self.scaler.fit_transform(std_features)
        pat_features = self.scaler.transform(pat_features)

        distance, path = fastdtw(std_features, pat_features, dist=euclidean)
        aligned_std = [std_features[i] for i, j in path]
        aligned_pat = [pat_features[j] for i, j in path]

        max_length = max(len(self.std_seq), len(self.pat_seq))
        normalized_distance = distance / max_length
        similarity = 1 / (1 + normalized_distance)
        frame_scores = self._compare_frames_with_multiple_matches(path)
        print("帧对齐得分：", frame_scores)
        return {
            'dtw_distance': distance,
            'alignment_path': path,
            'aligned_std': aligned_std,
            'aligned_pat': aligned_pat,
            'similarity_score': similarity,
            'frame_scores': frame_scores
        }

    def _extract_features(self, sequence):
        """
        构建时空特征向量：[归一化坐标, 角度值]
        """
        features = []
        for frame in sequence:
            coord_feat = np.array(frame['norm_landmarks']).flatten()
            angle_feat = np.array(list(frame['angles'].values()))
            features.append(np.concatenate([coord_feat, angle_feat]))
        return np.array(features)

    def _compare_frames_with_multiple_matches(self, path):
        """
        计算患者视频帧与多个标准视频帧的对比评分，并进行百分制归一化。
        新算法：
          - 对每个患者帧，与对应的多个标准视频帧匹配时，
            分别计算所有关节归一化坐标的欧氏距离均值和角度差的平均值（角度差除以 180 缩放到 [0,1]），
            然后将两者相加作为匹配得分。
          - 对所有匹配得分进行 min-max 归一化，将得分转换到 0～100 分（得分越高表示相似性越好）。
        注意：fastdtw 返回 (i, j) 中，i 属于标准序列，j 属于患者序列，
              因此构造字典时以患者帧索引 j 作为键。
        """
        frame_scores = []
        patient_to_standard = {}
        for i, j in path:
            if j not in patient_to_standard:
                patient_to_standard[j] = []
            patient_to_standard[j].append(i)

        for patient_frame_idx, standard_frame_idxs in patient_to_standard.items():
            patient_frame = self.pat_seq[patient_frame_idx]
            scores = []
            for standard_frame_idx in standard_frame_idxs:
                standard_frame = self.std_seq[standard_frame_idx]
                # 计算关节坐标差异（欧氏距离均值）
                coord_diffs = []
                num_joints = len(patient_frame['norm_landmarks'])
                for k in range(num_joints):
                    p_coord = np.array(patient_frame['norm_landmarks'][k])
                    s_coord = np.array(standard_frame['norm_landmarks'][k])
                    diff = np.linalg.norm(p_coord - s_coord)
                    coord_diffs.append(diff)
                avg_coord_diff = np.mean(coord_diffs)

                # 计算角度差异（绝对值平均，缩放到 [0,1]）
                p_angles = np.array(list(patient_frame['angles'].values()))
                s_angles = np.array(list(standard_frame['angles'].values()))
                if p_angles.size > 0:
                    avg_angle_diff = np.mean(np.abs(p_angles - s_angles))
                else:
                    avg_angle_diff = 0
                scaled_angle_diff = avg_angle_diff / 180.0

                total_diff = avg_coord_diff + scaled_angle_diff
                scores.append(total_diff)
            avg_score = np.mean(scores)
            frame_scores.append((patient_frame_idx, avg_score))

        all_scores = [score for _, score in frame_scores]
        min_score = min(all_scores)
        max_score = max(all_scores)
        normalized_scores = []
        for patient_frame_idx, score in frame_scores:
            normalized_score = 100 * (max_score - score) / (max_score - min_score) if max_score != min_score else 100
            normalized_scores.append((patient_frame_idx, normalized_score))

        return normalized_scores

    def generate_report(self, result):
        """
        生成对比报告图：
         - 第一子图绘制标准视频与患者视频中各关键点角度的变化曲线；
         - 第二子图绘制 DTW 对齐路径。
        将图保存为 comparison_report.jpg 并返回该文件名。
        """
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        for joint in self.std_seq[0]['angles']:
            std_angles = [f['angles'][joint] for f in self.std_seq]
            pat_angles = [f['angles'].get(joint, 0) for f in self.pat_seq]
            plt.plot(std_angles, label=f'Std {joint}')
            plt.plot(pat_angles, linestyle='--', label=f'Pat {joint}')
        plt.title('Joint Angle Comparison')
        plt.legend()

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

    def compare_overall_video(self):
        """
        新增对视频全程的整体评分方法。
        计算标准视频和患者视频所有帧特征向量的平均值，
        然后分别采用余弦相似度、L1 和 L2 差异计算相似度，
        转换为 0～100 分后，返回各项得分及一个综合得分（取平均）。
        """
        std_features = self._extract_features(self.std_seq)
        pat_features = self._extract_features(self.pat_seq)
        avg_std = np.mean(std_features, axis=0)
        avg_pat = np.mean(pat_features, axis=0)
        diff = avg_std - avg_pat
        # L1
        l1_diff = np.sum(np.abs(diff))
        # L2
        l2_diff = np.linalg.norm(diff)
        feature_length = len(avg_std)
        normalized_l1 = l1_diff / feature_length
        normalized_l2 = l2_diff / feature_length
        l1_score = 100 * (1 / (1 + normalized_l1))
        l2_score = 100 * (1 / (1 + normalized_l2))
        # 余弦相似度
        cosine_sim = np.dot(avg_std, avg_pat) / (np.linalg.norm(avg_std) * np.linalg.norm(avg_pat))
        cosine_score = cosine_sim * 100
        combined = (cosine_score + l1_score + l2_score) / 3
        return {
            "cosine_score": cosine_score,
            "l1_score": l1_score,
            "l2_score": l2_score,
            "combined_overall": combined
        }



    def compute_alignment_angle_score(self, alignment_path, k=10):
        """
        利用 DTW 对齐路径计算线性拟合得到的直线角度评分：
          1. 对齐路径中每个元素为 (标准帧索引, 患者帧索引)。
          2. 将患者帧索引作为 x 轴，标准帧索引作为 y 轴，进行线性回归，得到拟合直线斜率；
          3. 转换成角度 theta = arctan(slope)，理想情况下应为 45°；
          4. 评分公式：score = 100 - |theta - 45|；
        在计算前先调用 trim_alignment 剔除开头和结尾卡顿部分。
        返回一个字典，包含 slope、theta 和 angle_score。
        """
        import numpy as np
        trimmed_path = self.trim_alignment(alignment_path)
        points = np.array([[j, i] for i, j in trimmed_path])
        if points.shape[0] < 2:
            return {"slope": 1, "theta": 45, "angle_score": 100}
        # 线性回归拟合：y = a*x + b，其中 x 为患者帧索引，y 为标准帧索引
        slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
        theta = np.degrees(np.arctan(slope))
        score = 100 - abs(theta - 45)
        score = max(score, 0)
        return {"slope": slope, "theta": theta, "angle_score": score}


    def dtw_time_variance_score(self, alignment_path):
        """
        利用 DTW 对齐路径计算每一帧的时间差（患者帧索引 - 标准帧索引）的方差，
        然后用公式 score = 100 / (1 + variance) 得到时间一致性得分。
        在计算前先用 trim_alignment 去除开头和结尾连续重复的数据（卡顿）。
        返回一个字典，包含原始方差（time_variance）、得分（time_variance_score）以及时间差列表（differences）。
        """
        trimmed_path = self.trim_alignment(alignment_path)
        differences = [j - i for i, j in trimmed_path]
        if len(differences) == 0:
            return {"time_variance": 0, "time_variance_score": 100, "differences": []}
        variance = np.var(differences)
        score = 100 / (1 + variance)
        return {"time_variance": variance, "time_variance_score": score, "differences": differences}
    def compute_alignment_ratio(self, alignment_path):
        """
        根据 DTW 对齐路径 (alignment_path) 计算患者帧与标准帧之间的时间差差值序列，
        首先利用 trim_alignment 去除开头和结尾卡顿部分（例如连续重复的患者帧），
        然后计算在修剪后的差值序列中，相邻差值相等的比例，
        返回值在 0～1 之间（比例越高表示中间区域动作时间匹配越好）。
        """
        trimmed = self.trim_alignment(alignment_path)
        differences = [j - i for i, j in trimmed]
        if len(differences) <= 1:
            return 1.0  # 只有一帧或没有数据时，认为对齐100%
        aligned_count = sum(1 for k in range(1, len(differences)) if differences[k] == differences[k-1])
        ratio = aligned_count / (len(differences) - 1)
        return ratio

    def trim_alignment(self,alignment_path):
        """修剪 fastdtw 生成的 alignment_path，移除开头和结尾重复段，并打印患者帧索引范围。"""
        if not alignment_path:
            return []  # 空路径直接返回

        # 1. 修剪开头冗余匹配段
        prefix_end = 0
        if len(alignment_path) > 1:
            first_std, first_pat = alignment_path[0]
            # 检查第二个点判断是标准帧卡顿还是患者帧卡顿
            if alignment_path[1][0] == first_std:
                # 标准帧索引重复，标准帧卡顿
                while prefix_end < len(alignment_path) and alignment_path[prefix_end][0] == first_std:
                    prefix_end += 1
            elif alignment_path[1][1] == first_pat:
                # 患者帧索引重复，患者帧卡顿
                while prefix_end < len(alignment_path) and alignment_path[prefix_end][1] == first_pat:
                    prefix_end += 1
        trimmed_path = alignment_path[prefix_end:]  # 去除开头冗余段

        # 2. 修剪结尾冗余匹配段
        if not trimmed_path:
            # 开头移除后无剩余对齐点
            print("保留患者帧索引范围：<空>")  # 或提示路径为空
            return []
        suffix_start = len(trimmed_path) - 1
        if len(trimmed_path) > 1:
            last_std, last_pat = trimmed_path[-1]
            prev_std, prev_pat = trimmed_path[-2]
            if prev_std == last_std:
                # 结尾标准帧卡顿
                while suffix_start >= 0 and trimmed_path[suffix_start][0] == last_std:
                    suffix_start -= 1
            elif prev_pat == last_pat:
                # 结尾患者帧卡顿
                while suffix_start >= 0 and trimmed_path[suffix_start][1] == last_pat:
                    suffix_start -= 1
        trimmed_path = trimmed_path[:suffix_start + 1]  # 去除结尾冗余段

        # 3. 输出调试信息：保留的第一个和最后一个患者帧索引
        if trimmed_path:
            first_pat_idx = trimmed_path[0][1]
            last_pat_idx = trimmed_path[-1][1]
            print(f"保留患者帧索引范围：{first_pat_idx} 至 {last_pat_idx}")
        else:
            print("保留患者帧索引范围：<空>")
        # 4. 返回修剪后的路径
        return trimmed_path

    def segment_indices(self, indices):
        """
        辅助函数：将一个升序的帧索引列表合并为连续区间，返回区间列表，每个区间为 (start, end)。
        """
        if not indices:
            return []
        segments = []
        start = indices[0]
        prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                segments.append((start, prev))
                start = idx
                prev = idx
        segments.append((start, prev))
        return segments

    def remove_tail_segments(self, segments, gap_thresh):
        """
        从尾部开始，检查连续尾部区间：
        如果倒数第二段与最后一段之间的间隔 <= gap_thresh，则将这些连续尾部区间全部剔除。
        返回剔除后的区间列表。
        """
        if len(segments) < 2:
            return segments
        # 从尾部开始，累计连续尾部区间的索引（例如：最后一段一定要剔除，如果前一段和当前段间隔满足条件，也一起剔除）
        remove_indices = set()
        remove_indices.add(len(segments) - 1)
        for i in range(len(segments) - 2, -1, -1):
            # 比较当前区间的结束与下一区间的开始
            gap = segments[i + 1][0] - segments[i][1]
            if gap <= gap_thresh:
                remove_indices.add(i)
            else:
                break
        # 返回未被剔除的区间（保持原顺序）
        new_segments = [seg for idx, seg in enumerate(segments) if idx not in remove_indices]
        return new_segments

    def analyze_speed_variation(self, alignment_path, tail_gap_thresh=1):
        """
        分析 DTW 对齐路径（alignment_path），判断患者动作速度问题，并排除开头和结尾卡顿部分。
        思路：
          1. 根据对齐路径统计患者帧出现次数和标准帧映射，得到初步的慢动作（slow_segments）和快动作（fast_segments）区间。
          2. 定义全局患者帧边界（min_patient 与 max_patient），剔除包含边界的区段。
          3. 对剩余的尾部连续区间（按患者帧索引排序）进行处理：如果倒数连续两段的间隔 <= tail_gap_thresh，则将尾部所有这类区间一并剔除。
        返回格式为字典：
            {"slow_segments": [(start, end), ...],
             "fast_segments": [(start, end), ...]}
        """
        if not alignment_path:
            return {"slow_segments": [], "fast_segments": []}

        # 统计患者帧出现次数
        patient_count = {}
        for std_idx, pat_idx in alignment_path:
            patient_count[pat_idx] = patient_count.get(pat_idx, 0) + 1

        # 统计标准帧映射到的患者帧
        standard_map = {}
        for std_idx, pat_idx in alignment_path:
            if std_idx not in standard_map:
                standard_map[std_idx] = []
            standard_map[std_idx].append(pat_idx)

        # 全局患者帧边界
        all_patients = [pat_idx for _, pat_idx in alignment_path]
        min_patient = min(all_patients)
        max_patient = max(all_patients)

        # 得到慢动作初步区间：患者帧出现次数 > 1
        slow_frames = sorted([pat for pat, count in patient_count.items() if count > 1])
        slow_segments = self.segment_indices(slow_frames)

        # 得到快动作初步区间：标准帧对应多个患者帧
        fast_set = set()
        for std_idx, pat_list in standard_map.items():
            if len(pat_list) > 1:
                fast_set.update(pat_list)
        fast_frames = sorted(list(fast_set))
        fast_segments = self.segment_indices(fast_frames)

        # 剔除包含全局边界的区段
        slow_segments = [seg for seg in slow_segments if seg[0] > min_patient and seg[1] < max_patient]
        fast_segments = [seg for seg in fast_segments if seg[0] > min_patient and seg[1] < max_patient]

        # 对尾部连续区段应用 remove_tail_segments
        slow_segments = self.remove_tail_segments(slow_segments, tail_gap_thresh)
        fast_segments = self.remove_tail_segments(fast_segments, tail_gap_thresh)

        return {"slow_segments": slow_segments, "fast_segments": fast_segments}
