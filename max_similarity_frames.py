"""
此脚本用于从帧间相似度矩阵中选择k帧，使得它们之间的两两相似度之和最小（即选择最不相似的帧）。
可以处理的规模：n在100左右，k在5~10之间。
还可以使用CLIP模型直接计算视频帧之间的相似度矩阵。
"""

import matplotlib
# 设置matplotlib使用不需要中文字体的后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 修复中文字体问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

import numpy as np
import time
import os
import torch
import torchvision.transforms as T
from PIL import Image
from typing import List, Optional, Tuple, Union
import argparse
import math
import tqdm

# 导入用于视频处理和CLIP模型的库
try:
    import decord
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("警告：某些依赖库未安装，如需计算CLIP相似度，请安装：pip install decord transformers torch torchvision tqdm")


def get_video_frames(
    video_path: str,
    width: int,
    height: int,
    skip_frames_start: int,
    skip_frames_end: int,
    max_num_frames: int,
    frame_sample_step: Optional[int],
) -> torch.FloatTensor:
    """
    从视频中提取帧
    
    Args:
        video_path: 视频文件路径
        width: 调整大小后的帧宽度
        height: 调整大小后的帧高度
        skip_frames_start: 从开始跳过的帧数
        skip_frames_end: 从结尾跳过的帧数
        max_num_frames: 最大采样帧数
        frame_sample_step: 采样步长
        
    Returns:
        视频帧张量 [F, H, W, C]
    """
    with decord.bridge.use_torch():
        video_reader = decord.VideoReader(uri=video_path, width=width, height=height)
        video_num_frames = len(video_reader)
        start_frame = min(skip_frames_start, video_num_frames)
        end_frame = max(0, video_num_frames - skip_frames_end)

        if end_frame <= start_frame:
            indices = [start_frame]
        elif end_frame - start_frame <= max_num_frames:
            indices = list(range(start_frame, end_frame))
        else:
            step = frame_sample_step or (end_frame - start_frame) // max_num_frames
            indices = list(range(start_frame, end_frame, step))

        frames = video_reader.get_batch(indices=indices)
        frames = frames[:max_num_frames].float()  # 确保不超过限制

        # 正规化帧
        transform = T.Lambda(lambda x: x / 255.0)
        frames = torch.stack(tuple(map(transform, frames)), dim=0)

        return frames  # [F, H, W, C]


def compute_clip_similarity(frames: torch.Tensor, device: str) -> torch.Tensor:
    """
    计算视频帧之间的CLIP相似度矩阵
    
    Args:
        frames: 视频帧 [F, H, W, C]
        device: 计算设备 ('cuda' 或 'cpu')
        
    Returns:
        相似度矩阵 [F, F]
    """
    # 加载CLIP模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    num_frames = frames.shape[0]
    similarity_matrix = torch.zeros((num_frames, num_frames), device=device)
    
    # 将帧转换为PIL图像列表
    pil_frames = []
    for i in range(num_frames):
        # 转换为PIL图像
        pil_image = Image.fromarray((frames[i].cpu().numpy() * 255).astype(np.uint8))
        pil_frames.append(pil_image)
    
    # 处理所有图像以获取特征
    inputs = processor(images=pil_frames, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # 归一化特征
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    
    # 计算相似度矩阵
    similarity_matrix = torch.mm(image_features, image_features.t())
    
    return similarity_matrix


def visualize_similarity_matrix(
    similarity_matrix: torch.Tensor,
    output_path: str,
    prefix: str = "similarity"
):
    """
    可视化相似度矩阵并保存，包括原始矩阵和最大最小值归一化后的矩阵
    
    Args:
        similarity_matrix: 相似度矩阵 [F, F]
        output_path: 输出目录
        prefix: 输出文件前缀
    """
    os.makedirs(output_path, exist_ok=True)
    
    # 转换为numpy数组
    if isinstance(similarity_matrix, torch.Tensor):
        sim_matrix_np = similarity_matrix.cpu().numpy()
    else:
        sim_matrix_np = similarity_matrix
    
    # 创建原始相似度矩阵图像
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix_np, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Similarity')
    plt.title('Frame Similarity Matrix (Original)')
    plt.xlabel('Frame Index')
    plt.ylabel('Frame Index')
    
    # 保存原始图像
    output_file = os.path.join(output_path, f"{prefix}_matrix_original.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"原始相似度矩阵已保存至: {output_file}")
    
    # 创建最大最小值归一化后的相似度矩阵
    # 获取非对角线元素（对角线元素都是1，不参与归一化）
    mask = ~np.eye(sim_matrix_np.shape[0], dtype=bool)
    non_diagonal = sim_matrix_np[mask]
    
    # 计算最大最小值
    min_val = np.min(non_diagonal)
    max_val = np.max(non_diagonal)
    
    # 创建归一化后的矩阵副本
    normalized_matrix = sim_matrix_np.copy()
    
    # 对非对角线元素进行归一化
    if max_val > min_val:  # 避免除以零
        normalized_matrix[mask] = (normalized_matrix[mask] - min_val) / (max_val - min_val)
    
    # 保持对角线元素为1
    np.fill_diagonal(normalized_matrix, 1.0)
    
    # 创建归一化后的相似度矩阵图像
    plt.figure(figsize=(10, 8))
    plt.imshow(normalized_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Normalized Similarity')
    plt.title('Frame Similarity Matrix (Min-Max Normalized)')
    plt.xlabel('Frame Index')
    plt.ylabel('Frame Index')
    
    # 保存归一化图像
    output_file = os.path.join(output_path, f"{prefix}_matrix_normalized.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"归一化相似度矩阵已保存至: {output_file}")
    
    # 打印归一化信息
    print(f"相似度统计: 最小值={min_val:.4f}, 最大值={max_val:.4f}, 归一化范围=[0,1]")
    
    # 保存数值数据
    np_output_file = os.path.join(output_path, f"{prefix}_matrix.npy")
    np.save(np_output_file, sim_matrix_np)
    print(f"相似度矩阵数据已保存至: {np_output_file}")
    
    # 保存归一化后的数值数据
    np_norm_output_file = os.path.join(output_path, f"{prefix}_matrix_normalized.npy")
    np.save(np_norm_output_file, normalized_matrix)
    print(f"归一化相似度矩阵数据已保存至: {np_norm_output_file}")
    
    return normalized_matrix  # 返回归一化后的矩阵，以便后续处理


def visualize_multiple_matrices(
    matrices: List[np.ndarray],
    titles: List[str],
    output_path: str,
    filename: str = "matrix_comparison.png"
):
    """
    并排可视化多个相似度矩阵
    
    Args:
        matrices: 相似度矩阵列表
        titles: 每个矩阵的标题
        output_path: 输出目录
        filename: 输出文件名
    """
    os.makedirs(output_path, exist_ok=True)
    
    n_matrices = len(matrices)
    if n_matrices == 0:
        return
    
    # 计算子图布局
    n_cols = min(3, n_matrices)  # 最多3列
    n_rows = (n_matrices + n_cols - 1) // n_cols
    
    # 创建图像
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # 确保axes是数组，即使只有一个子图
    axes = np.atleast_1d(axes)
    
    # 展平axes数组，方便索引
    axes_flat = axes.flatten() if axes.ndim > 1 else axes
    
    # 绘制每个矩阵
    for i in range(n_matrices):
        ax = axes_flat[i]
        im = ax.imshow(matrices[i], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(titles[i])
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Frame Index')
        fig.colorbar(im, ax=ax, label='Similarity')
    
    # 隐藏多余的子图
    for i in range(n_matrices, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    output_file = os.path.join(output_path, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matrix comparison visualization saved to: {output_file}")


def adjust_similarity_with_distance(
    similarity_matrix: np.ndarray, 
    distance_weight: float = 0.5,
    max_distance_influence: float = 0.5
) -> np.ndarray:
    """
    根据帧间距离调整相似度矩阵
    
    Args:
        similarity_matrix: 原始相似度矩阵
        distance_weight: 距离权重因子，0表示不考虑距离，1表示距离影响最大
        max_distance_influence: 最大距离影响因子，控制距离对相似度的最大影响程度
        
    Returns:
        调整后的相似度矩阵
    """
    n = similarity_matrix.shape[0]
    adjusted_matrix = similarity_matrix.copy()
    
    # 创建距离矩阵：帧i和帧j之间的归一化距离
    distance_matrix = np.zeros((n, n))
    max_distance = n - 1  # 最大可能距离
    
    for i in range(n):
        for j in range(n):
            # 计算归一化距离 (0到1之间)
            distance_matrix[i, j] = abs(i - j) / max_distance
    
    # 将距离因素整合到相似度矩阵中
    # 距离越大，对相似度的惩罚越大
    for i in range(n):
        for j in range(n):
            if i != j:  # 不调整对角线元素
                # 距离惩罚因子：距离越大，惩罚越大
                distance_penalty = distance_matrix[i, j] 
                
                # 根据权重调整相似度
                # 当distance_weight=0时，保持原始相似度
                # 当distance_weight=1时，最大程度考虑距离因素
                adjusted_matrix[i, j] = (1 - distance_weight) * similarity_matrix[i, j] + distance_weight * (similarity_matrix[i, j] * (1 - distance_penalty))
    
    return adjusted_matrix, distance_matrix


def select_min_similarity_frames_backtracking(similarity_matrix: np.ndarray, k: int, time_limit: int = 30) -> List[int]:
    """
    使用回溯搜索算法从相似度矩阵中选择k帧，使得这k帧之间的两两相似度之和最小（即选择最不相似的帧）。
    增加了时间限制和更强的剪枝策略。
    
    Args:
        similarity_matrix: n×n的相似度矩阵，表示帧之间的相似度
        k: 需要选择的帧数
        time_limit: 算法运行的最大时间（秒）
        
    Returns:
        长度为k的列表，包含选中的帧索引
    """
    n = similarity_matrix.shape[0]
    if k >= n:
        return list(range(n))
    
    # 使用贪心算法获取初始解
    greedy_solution = select_min_similarity_frames_greedy(similarity_matrix, k)
    greedy_similarity = calculate_similarity_sum(similarity_matrix, greedy_solution)
    
    # 存储最佳结果
    best_frames = greedy_solution.copy()
    min_similarity = greedy_similarity
    
    # 记录开始时间
    start_time = time.time()
    
    # 预计算每个节点与其他节点的平均相似度，用于选择优先搜索的节点顺序
    avg_similarities = []
    for i in range(n):
        avg_sim = np.sum(similarity_matrix[i]) / (n - 1)  # 不包括自己
        avg_similarities.append((i, avg_sim))
    
    # 按平均相似度升序排序，优先考虑相似度低的节点
    search_order = [idx for idx, _ in sorted(avg_similarities, key=lambda x: x[1])]
    
    # 回溯函数
    def backtrack(selected, remaining, current_similarity):
        nonlocal best_frames, min_similarity
        
        # 检查时间限制
        if time.time() - start_time > time_limit:
            print(f"回溯搜索达到时间限制 {time_limit} 秒，返回当前最佳结果")
            return True  # 表示达到时间限制
        
        # 达到要求的帧数
        if len(selected) == k:
            if current_similarity < min_similarity:
                min_similarity = current_similarity
                best_frames = selected.copy()
            return False
        
        # 剪枝：如果剩余节点不足以满足要求
        if len(selected) + len(remaining) < k:
            return False
        
        # 剪枝：即使选择相似度最小的组合，也无法超过当前最优解
        remaining_k = k - len(selected)
        if remaining_k > 0:
            # 简化的下界估计
            if current_similarity >= min_similarity:
                return False
        
        # 遍历剩余节点
        for i in range(len(remaining)):
            node = remaining[i]
            
            # 计算添加此节点后的相似度增加量
            new_similarity = current_similarity
            for idx in selected:
                new_similarity += similarity_matrix[node, idx]
            
            # 如果当前相似度已经超过最小值，剪枝
            if new_similarity >= min_similarity:
                continue
            
            # 选择当前节点
            selected.append(node)
            new_remaining = remaining[:i] + remaining[i+1:]
            
            # 递归
            time_limit_reached = backtrack(selected, new_remaining, new_similarity)
            if time_limit_reached:
                return True
            
            # 回溯
            selected.pop()
        
        return False
    
    # 开始回溯搜索
    initial_remaining = search_order.copy()
    time_limit_reached = backtrack([], initial_remaining, 0)
    
    # 如果没有完成搜索，输出一条消息
    if time_limit_reached:
        print(f"警告：回溯搜索未完成，返回的可能不是最优解")
    
    return best_frames


def select_min_similarity_frames_greedy(similarity_matrix: np.ndarray, k: int) -> List[int]:
    """
    使用贪心算法从相似度矩阵中选择k帧，使得这k帧之间的两两相似度之和最小（即选择最不相似的帧）。
    
    Args:
        similarity_matrix: n×n的相似度矩阵，表示帧之间的相似度
        k: 需要选择的帧数
        
    Returns:
        长度为k的列表，包含选中的帧索引
    """
    n = similarity_matrix.shape[0]
    if k >= n:
        return list(range(n))
    
    # 初始化选择第一个帧
    selected_indices = [0]
    
    # 逐步添加能够最小化总相似度的帧
    while len(selected_indices) < k:
        min_gain = float('inf')
        next_idx = -1
        
        for i in range(n):
            if i in selected_indices:
                continue
            
            # 计算添加这一帧能带来的相似度增益
            gain = sum(similarity_matrix[i, j] for j in selected_indices)
            
            if gain < min_gain:
                min_gain = gain
                next_idx = i
        
        selected_indices.append(next_idx)
    
    return selected_indices


def calculate_similarity_sum(similarity_matrix: np.ndarray, indices: List[int]) -> float:
    """
    计算选中帧之间的相似度总和
    
    Args:
        similarity_matrix: 相似度矩阵
        indices: 选中的帧索引
        
    Returns:
        相似度总和
    """
    total = 0
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            total += similarity_matrix[indices[i], indices[j]]
    return total


def save_selected_frames(frames, selected_indices, output_path, prefix="selected", cols_per_row=5):
    """
    保存选中的帧，每行显示cols_per_row张图片
    
    Args:
        frames: 视频帧
        selected_indices: 选中的帧索引
        output_path: 输出路径
        prefix: 文件前缀
        cols_per_row: 每行显示的图片数量
    """
    os.makedirs(output_path, exist_ok=True)
    
    # 保存每一帧
    for i, idx in enumerate(selected_indices):
        frame_image = Image.fromarray((frames[idx].cpu().numpy() * 255).astype(np.uint8))
        frame_path = os.path.join(output_path, f"{prefix}_frame_{idx}.png")
        frame_image.save(frame_path)
    
    # 创建一个包含所有选中帧的图像，每行cols_per_row张
    num_frames = len(selected_indices)
    rows = math.ceil(num_frames / cols_per_row)
    
    plt.figure(figsize=(15, 3 * rows))
    for i, idx in enumerate(selected_indices):
        row = i // cols_per_row
        col = i % cols_per_row
        plt_idx = row * cols_per_row + col + 1
        plt.subplot(rows, cols_per_row, plt_idx)
        plt.imshow(frames[idx].cpu().numpy())
        plt.title(f"Frame {idx}")  # 使用英文替代中文
        plt.axis('off')
    plt.tight_layout()
    all_frames_path = os.path.join(output_path, f"{prefix}_all_frames.png")
    plt.savefig(all_frames_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All selected frames saved to: {all_frames_path}")  # 使用英文输出


def select_min_similarity_frames_dp_greedy(similarity_matrix: np.ndarray, k: int) -> List[int]:
    """
    使用混合动态规划和贪心的方法选择k帧，尝试找到最不相似的帧组合。
    此方法比纯回溯快得多，但可能不会找到全局最优解。
    
    Args:
        similarity_matrix: n×n的相似度矩阵
        k: 需要选择的帧数
        
    Returns:
        长度为k的列表，包含选中的帧索引
    """
    n = similarity_matrix.shape[0]
    if k >= n:
        return list(range(n))
    
    # 计算每帧与其他所有帧的平均相似度
    avg_similarities = np.zeros(n)
    for i in range(n):
        # 排除自身（对角线元素）
        non_diagonal = np.concatenate([similarity_matrix[i, :i], similarity_matrix[i, i+1:]])
        avg_similarities[i] = np.mean(non_diagonal)
    
    # 按平均相似度升序排序（优先考虑相似度低的帧）
    candidates = np.argsort(avg_similarities)[:min(3*k, n)]  # 选取前3k个候选帧
    
    # 从候选帧中贪心地选择k帧
    selected = []
    
    # 先选择平均相似度最低的帧
    selected.append(candidates[0])
    candidates = np.delete(candidates, 0)
    
    # 贪心选择剩余的k-1帧
    while len(selected) < k and len(candidates) > 0:
        min_sim = float('inf')
        best_idx = -1
        best_candidate_idx = -1
        
        for i, candidate in enumerate(candidates):
            # 计算该候选帧与已选帧的总相似度
            sim_sum = sum(similarity_matrix[candidate, idx] for idx in selected)
            
            if sim_sum < min_sim:
                min_sim = sim_sum
                best_idx = candidate
                best_candidate_idx = i
        
        if best_idx != -1:
            selected.append(best_idx)
            candidates = np.delete(candidates, best_candidate_idx)
        else:
            break
    
    # 如果还未选够k帧，从剩余帧中随机选择
    if len(selected) < k:
        remaining = [i for i in range(n) if i not in selected]
        selected.extend(remaining[:k-len(selected)])
    
    return selected


def select_min_similarity_frames_dp(similarity_matrix: np.ndarray, k: int) -> List[int]:
    """
    使用动态规划方法选择k帧，使得它们之间的相似度之和最小。
    适用于较小规模的问题（n≤30, k≤10）。
    
    Args:
        similarity_matrix: n×n的相似度矩阵
        k: 需要选择的帧数
        
    Returns:
        长度为k的列表，包含选中的帧索引
    """
    n = similarity_matrix.shape[0]
    if k >= n:
        return list(range(n))
    
    # 如果问题规模过大，回退到贪心算法
    if n > 30:
        print(f"警告: 帧数 n={n} 过大，动态规划可能内存溢出。正在回退到贪心算法...")
        return select_min_similarity_frames_greedy(similarity_matrix, k)
    
    # 创建状态数组：dp[mask][last]表示选择了mask中的帧且最后一个帧是last时的最小相似度
    # mask是一个二进制掩码，表示哪些帧被选择
    dp = {}
    parent = {}  # 用于回溯路径
    
    # 初始化：选择1个帧的情况
    for i in range(n):
        mask = 1 << i  # 二进制掩码，只有第i位为1
        dp[(mask, i)] = 0  # 只选一个帧时，相似度和为0
    
    # 动态规划填表
    # 枚举所有可能的掩码（即所有可能的帧选择组合）
    for count in range(2, k+1):  # 从选择2个帧开始
        for mask in range(1, 1 << n):
            # 如果mask中1的个数不等于count，跳过
            if bin(mask).count('1') != count:
                continue
            
            # 枚举最后选择的帧
            for last in range(n):
                if not (mask & (1 << last)):  # 如果last不在mask中，跳过
                    continue
                
                # 计算去掉last后的掩码
                prev_mask = mask & ~(1 << last)
                
                # 如果只有一个帧了，处理边界情况
                if prev_mask == 0:
                    continue
                
                # 枚举倒数第二个选择的帧
                min_sim = float('inf')
                best_prev = -1
                
                for prev in range(n):
                    if not (prev_mask & (1 << prev)):  # 如果prev不在prev_mask中，跳过
                        continue
                    
                    # 计算相似度：之前的最小相似度 + last与prev之间的相似度
                    current_sim = dp.get((prev_mask, prev), float('inf')) + similarity_matrix[last, prev]
                    
                    if current_sim < min_sim:
                        min_sim = current_sim
                        best_prev = prev
                
                if best_prev != -1:
                    dp[(mask, last)] = min_sim
                    parent[(mask, last)] = best_prev
    
    # 找到最优解
    final_mask = (1 << k) - 1  # 选择了k个帧的掩码
    if n == k:
        final_mask = (1 << n) - 1  # 如果k=n，选择所有帧
    
    min_sim = float('inf')
    last_node = -1
    
    # 枚举所有可能的最后一个帧
    for last in range(n):
        for mask in range(1, 1 << n):
            if bin(mask).count('1') == k and (mask & (1 << last)):
                if (mask, last) in dp and dp[(mask, last)] < min_sim:
                    min_sim = dp[(mask, last)]
                    last_node = last
                    final_mask = mask
    
    # 回溯构建路径
    selected = []
    curr_mask = final_mask
    curr_node = last_node
    
    while len(selected) < k and curr_node != -1:
        selected.append(curr_node)
        
        # 更新掩码和节点
        next_mask = curr_mask & ~(1 << curr_node)
        if next_mask == 0 or (next_mask, curr_node) not in parent:
            break
            
        curr_node = parent[(curr_mask, curr_node)]
        curr_mask = next_mask
    
    # 如果没有找到完整的路径，回退到贪心算法
    if len(selected) < k:
        print("警告: 动态规划未能找到完整路径，回退到贪心算法...")
        return select_min_similarity_frames_greedy(similarity_matrix, k)
    
    return selected


def select_min_similarity_frames_dp_optimized(similarity_matrix: np.ndarray, k: int) -> List[int]:
    """
    使用优化的动态规划方法选择k帧，使得它们之间的相似度之和最小。
    此方法使用迭代的方式构建解，可处理中等规模问题(n≤100, k≤15)。
    
    Args:
        similarity_matrix: n×n的相似度矩阵
        k: 需要选择的帧数
        
    Returns:
        长度为k的列表，包含选中的帧索引
    """
    n = similarity_matrix.shape[0]
    if k >= n:
        return list(range(n))
    
    # 如果规模太大，使用贪心算法
    if k > 15:
        print(f"警告: 问题规模过大 (n={n}, k={k})，回退到贪心算法...")
        return select_min_similarity_frames_greedy(similarity_matrix, k)
    
    # 计算每个帧的平均相似度，用于预筛选
    avg_similarities = np.zeros(n)
    for i in range(n):
        # 排除自身（对角线元素）
        non_diagonal = np.concatenate([similarity_matrix[i, :i], similarity_matrix[i, i+1:]])
        avg_similarities[i] = np.mean(non_diagonal)
    
    # 选择平均相似度最低的候选帧（减少状态空间）
    if n > 100:
        candidate_indices = np.argsort(avg_similarities)[:min(100, n)]
        # 创建候选帧的相似度子矩阵
        sub_matrix = np.zeros((len(candidate_indices), len(candidate_indices)))
        for i, idx1 in enumerate(candidate_indices):
            for j, idx2 in enumerate(candidate_indices):
                sub_matrix[i, j] = similarity_matrix[idx1, idx2]
        
        # 使用子矩阵计算
        similarity_matrix = sub_matrix
        n = len(candidate_indices)
        
        # 创建索引映射
        index_map = {i: candidate_indices[i] for i in range(n)}
    else:
        index_map = {i: i for i in range(n)}
    
    # 初始化：保存所有大小为i的子集的最优解
    # dp[i][subset] = (min_similarity, selected_frames)
    dp = [{} for _ in range(k+1)]
    
    # 单个帧的情况
    for i in range(n):
        dp[1][frozenset([i])] = (0, [i])
    
    # 动态规划填表
    for size in tqdm.tqdm(range(2, k+1), desc="Dynamic Programming"):
        # 从上一级的最优解中扩展
        for prev_subset, (prev_sim, prev_frames) in dp[size-1].items():
            # 尝试添加一个新的帧
            for new_frame in range(n):
                if new_frame in prev_subset:
                    continue
                
                # 计算添加这个帧后的相似度增量
                sim_increment = sum(similarity_matrix[new_frame, f] for f in prev_frames)
                new_sim = prev_sim + sim_increment
                
                # 创建新的子集
                new_subset = frozenset(list(prev_subset) + [new_frame])
                
                # 如果这个新子集还没有记录，或者新的相似度更小，则更新
                if new_subset not in dp[size] or new_sim < dp[size][new_subset][0]:
                    dp[size][new_subset] = (new_sim, prev_frames + [new_frame])
    
    # 找到大小为k的子集中相似度最小的
    min_sim = float('inf')
    best_frames = []
    
    for subset, (sim, frames) in dp[k].items():
        if sim < min_sim:
            min_sim = sim
            best_frames = frames
    
    # 将内部索引映射回原始索引
    result = [index_map[idx] for idx in best_frames]
    return result


def select_min_similarity_frames_uniform(similarity_matrix: np.ndarray, k: int) -> List[int]:
    """
    最朴素的均匀采样算法，不考虑相似度，只选择均匀分布的帧。
    
    Args:
        similarity_matrix: n×n的相似度矩阵
        k: 需要选择的帧数
        
    Returns:
        长度为k的列表，包含选中的帧索引
    """
    n = similarity_matrix.shape[0]
    if k >= n:
        return list(range(n))
    
    # 计算步长，使得选择的帧均匀分布
    step = n / k
    
    # 选择帧
    selected = []
    for i in range(k):
        # 对于每个位置，选择最接近的整数帧索引
        idx = min(n - 1, int(i * step))
        selected.append(idx)
    
    return selected


def select_min_similarity_frames_random(similarity_matrix: np.ndarray, k: int, seed: int = 42) -> List[int]:
    """
    随机选择k帧，用作基线对比。
    
    Args:
        similarity_matrix: n×n的相似度矩阵
        k: 需要选择的帧数
        seed: 随机种子，确保结果可重现
        
    Returns:
        长度为k的列表，包含选中的帧索引
    """
    np.random.seed(seed)
    n = similarity_matrix.shape[0]
    if k >= n:
        return list(range(n))
    
    # 随机选择k个不重复的帧索引
    selected = np.random.choice(n, size=k, replace=False)
    return selected.tolist()


def select_min_similarity_frames_first_last(similarity_matrix: np.ndarray, k: int) -> List[int]:
    """
    选择第一帧、最后一帧，以及它们之间均匀分布的帧。
    这是视频关键帧提取中的一种常见朴素方法。
    
    Args:
        similarity_matrix: n×n的相似度矩阵
        k: 需要选择的帧数
        
    Returns:
        长度为k的列表，包含选中的帧索引
    """
    n = similarity_matrix.shape[0]
    if k >= n:
        return list(range(n))
    
    if k == 1:
        return [0]  # 只选第一帧
    
    selected = [0, n-1]  # 第一帧和最后一帧
    
    # 如果k>2，在中间均匀选择剩余的帧
    if k > 2:
        step = (n - 1) / (k - 1)
        for i in range(1, k-1):
            idx = min(n - 1, int(i * step))
            selected.append(idx)
    
    return sorted(selected)


def select_min_similarity_frames_brute_force(similarity_matrix: np.ndarray, k: int, max_combinations: int = 10000000) -> List[int]:
    """
    使用暴力穷举算法选择k帧，保证找到全局最优解。
    会尝试所有可能的组合，因此复杂度为O(n choose k)，仅适用于较小规模的问题。
    
    Args:
        similarity_matrix: n×n的相似度矩阵
        k: 需要选择的帧数
        max_combinations: 最大组合数限制，超过此限制会返回警告
        
    Returns:
        长度为k的列表，包含选中的帧索引
    """
    import itertools
    
    n = similarity_matrix.shape[0]
    if k >= n:
        return list(range(n))
    
    # 计算组合数
    from math import comb
    num_combinations = comb(n, k)
    
    if num_combinations > max_combinations:
        print(f"警告: 暴力算法需要评估{num_combinations}种组合，超过限制{max_combinations}")
        print(f"改用贪心算法...")
        return select_min_similarity_frames_greedy(similarity_matrix, k)
    
    print(f"暴力算法正在评估全部{num_combinations}种组合...")
    
    # 计算所有可能的k帧组合的相似度总和
    min_similarity = float('inf')
    best_combination = None
    
    # 使用tqdm显示进度
    import tqdm
    for combination in tqdm.tqdm(itertools.combinations(range(n), k), total=num_combinations, desc="Brute Force"):
        combination = list(combination)
        similarity_sum = calculate_similarity_sum(similarity_matrix, combination)
        
        if similarity_sum < min_similarity:
            min_similarity = similarity_sum
            best_combination = combination
    
    print(f"暴力算法完成，找到最优解: 相似度总和 = {min_similarity:.4f}")
    return best_combination


def visualize_specific_frames(frames, indices, output_path, title="Specific Frames", cols_per_row=5):
    """
    可视化指定索引的帧
    
    Args:
        frames: 视频帧
        indices: 要可视化的帧索引列表
        output_path: 输出路径
        title: 图像标题
        cols_per_row: 每行显示的图片数量
    """
    os.makedirs(output_path, exist_ok=True)
    
    num_frames = len(indices)
    rows = math.ceil(num_frames / cols_per_row)
    
    plt.figure(figsize=(15, 3 * rows))
    for i, idx in enumerate(indices):
        row = i // cols_per_row
        col = i % cols_per_row
        plt_idx = row * cols_per_row + col + 1
        plt.subplot(rows, cols_per_row, plt_idx)
        plt.imshow(frames[idx].cpu().numpy())
        plt.title(f"Frame {idx}")
        plt.axis('off')
    plt.tight_layout()
    
    # 保存图像
    output_file = os.path.join(output_path, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"指定帧可视化已保存至: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="从视频帧中选择k帧，使相似度总和最小（即选择最不相似的帧）")
    
    # 基本参数
    parser.add_argument("--k", type=int, default=5, help="要选择的帧数")
    parser.add_argument("--output_path", type=str, default="output", help="输出结果的路径")
    parser.add_argument("--method", type=str, default="greedy", 
                        choices=["backtracking", "greedy", "dp_greedy", "dp", "dp_optimized", 
                                "uniform", "random", "first_last", "brute_force"], 
                        help="使用的算法：backtracking（回溯）、greedy（贪心）、dp_greedy（贪心启发）、"
                             "dp（动态规划）、dp_optimized（优化的动态规划）、uniform（均匀采样）、"
                             "random（随机选择）、first_last（首尾帧）、brute_force（暴力穷举）")
    parser.add_argument("--time_limit", type=int, default=30, 
                        help="回溯算法的时间限制（秒），超过此时间将返回当前最佳结果")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="随机算法的种子")
    parser.add_argument("--max_combinations", type=int, default=10000000,
                        help="暴力算法的最大组合数限制")
    
    # 距离相关参数
    parser.add_argument("--distance_weight", type=float, default=0.5, 
                        help="帧间距离的权重因子（0-1之间），0表示不考虑距离，1表示距离影响最大")
    parser.add_argument("--max_distance_influence", type=float, default=0.5, 
                        help="最大距离影响因子（0-1之间），控制距离对相似度的最大影响程度")
    
    # 视频处理参数
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_path", type=str, help="视频文件路径，用于直接从视频计算相似度")
    group.add_argument("--similarity_matrix", type=str, help="预先计算的相似度矩阵文件路径（.npy格式）")
    
    parser.add_argument("--width", type=int, default=720, help="调整大小后的视频帧宽度")
    parser.add_argument("--height", type=int, default=480, help="调整大小后的视频帧高度")
    parser.add_argument("--skip_frames_start", type=int, default=0, help="从开始跳过的帧数")
    parser.add_argument("--skip_frames_end", type=int, default=0, help="从结尾跳过的帧数")
    parser.add_argument("--frame_sample_step", type=int, default=None, help="采样帧的时间步长")
    parser.add_argument("--max_num_frames", type=int, default=101, help="最大采样帧数")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="计算设备")
    parser.add_argument("--cols_per_row", type=int, default=5, help="在结果图片中每行显示的帧数量")
    
    # 比较算法参数
    parser.add_argument("--compare_methods", action="store_true",
                        help="比较多种算法并输出结果")
    
    # 可视化特定帧的参数
    parser.add_argument("--visualize_frames", action="store_true",
                        help="可视化特定帧而不是运行选择算法")
    parser.add_argument("--frame_indices", type=str, default="0,10,20,30,40,50,60,70,80,90",
                        help="要可视化的帧索引，以逗号分隔")
    
    args = parser.parse_args()
    
    # 检查输出路径
    os.makedirs(args.output_path, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    # 获取相似度矩阵
    if args.video_path:
        print(f"正在读取视频: {args.video_path}")
        # 提取视频帧
        frames = get_video_frames(
            video_path=args.video_path,
            width=args.width,
            height=args.height,
            skip_frames_start=args.skip_frames_start,
            skip_frames_end=args.skip_frames_end,
            max_num_frames=args.max_num_frames,
            frame_sample_step=args.frame_sample_step,
        )
        
        print(f"视频帧数: {frames.shape[0]}")
        
        # 如果只需要可视化特定帧
        if args.visualize_frames:
            frame_indices = [int(idx.strip()) for idx in args.frame_indices.split(",")]
            print(f"正在可视化指定的帧: {frame_indices}")
            visualize_specific_frames(frames, frame_indices, args.output_path, 
                                    title="Specified Frames", cols_per_row=args.cols_per_row)
            return
        
        # 计算CLIP相似度
        print("正在计算CLIP相似度...")
        similarity_matrix = compute_clip_similarity(frames, device)
        
        # 将相似度矩阵转换为numpy数组用于后续处理
        similarity_matrix_np = similarity_matrix.cpu().numpy()
    else:
        # 从文件加载相似度矩阵
        print(f"从文件加载相似度矩阵: {args.similarity_matrix}")
        similarity_matrix_np = np.load(args.similarity_matrix)
        frames = None  # 没有帧数据
        
        # 如果只需要可视化特定帧但没有帧数据
        if args.visualize_frames:
            print("错误: 要可视化帧，必须提供视频路径而不是相似度矩阵")
            return
    
    print(f"相似度矩阵大小: {similarity_matrix_np.shape}")
    
    # 保存原始矩阵，但不显示
    original_matrix = similarity_matrix_np.copy()
    
    # 计算最大最小值归一化矩阵
    normalized_matrix = visualize_similarity_matrix(similarity_matrix_np, args.output_path, prefix="normalized")
    
    # 准备可视化矩阵列表，按照要求的顺序
    matrices_to_compare = []
    titles = []
    
    # 添加归一化矩阵
    matrices_to_compare.append(normalized_matrix)
    titles.append("Min-Max Normalized Matrix")
    
    # 如果指定了距离权重，则调整相似度矩阵
    if args.distance_weight > 0:
        print(f"应用帧间距离调整，距离权重: {args.distance_weight}, 最大影响: {args.max_distance_influence}")
        adjusted_matrix, distance_matrix = adjust_similarity_with_distance(
            similarity_matrix_np, 
            distance_weight=args.distance_weight,
            max_distance_influence=args.max_distance_influence
        )
        
        # 添加距离矩阵
        matrices_to_compare.append(distance_matrix)
        titles.append("Distance Matrix")
        
        # 添加距离调整后的矩阵
        adjusted_normalized_matrix = visualize_similarity_matrix(adjusted_matrix, args.output_path, prefix="distance_adjusted")
        matrices_to_compare.append(adjusted_matrix)
        titles.append("Distance-Adjusted Matrix")
        
        # 创建综合可视化
        visualize_multiple_matrices(
            matrices_to_compare,
            titles,
            args.output_path,
            "matrix_comparison.png"
        )
        
        # 使用调整后的矩阵
        similarity_matrix_np = adjusted_matrix
    
    # 如果需要比较多种算法
    if args.compare_methods:
        methods = {
            "backtracking": lambda: select_min_similarity_frames_backtracking(similarity_matrix_np, args.k, args.time_limit),
            "greedy": lambda: select_min_similarity_frames_greedy(similarity_matrix_np, args.k),
            "dp_greedy": lambda: select_min_similarity_frames_dp_greedy(similarity_matrix_np, args.k),
            "dp_optimized": lambda: select_min_similarity_frames_dp_optimized(similarity_matrix_np, args.k),
            "uniform": lambda: select_min_similarity_frames_uniform(similarity_matrix_np, args.k),
            "random": lambda: select_min_similarity_frames_random(similarity_matrix_np, args.k, args.random_seed),
            "first_last": lambda: select_min_similarity_frames_first_last(similarity_matrix_np, args.k),
            "brute_force": lambda: select_min_similarity_frames_brute_force(similarity_matrix_np, args.k, args.max_combinations)
        }
        
        results = {}
        times = {}
        
        for method_name, method_func in methods.items():
            print(f"\n正在使用 {method_name} 算法选择{args.k}帧...")
            start_time = time.time()
            selected = method_func()
            elapsed = time.time() - start_time
            similarity_sum = calculate_similarity_sum(similarity_matrix_np, selected)
            
            # 计算帧间距离
            sorted_indices = sorted(selected)
            distances = [sorted_indices[i+1] - sorted_indices[i] for i in range(len(sorted_indices)-1)]
            avg_distance = sum(distances) / len(distances) if distances else 0
            
            results[method_name] = {
                "selected": sorted(selected),
                "similarity_sum": similarity_sum,
                "avg_distance": avg_distance,
                "time": elapsed
            }
            
            print(f"  选中的帧索引: {sorted(selected)}")
            print(f"  相似度总和: {similarity_sum:.4f}")
            print(f"  平均帧间距离: {avg_distance:.2f}")
            print(f"  执行时间: {elapsed:.2f}秒")
        
        # 保存比较结果
        with open(os.path.join(args.output_path, "method_comparison.txt"), "w") as f:
            f.write(f"帧数: {similarity_matrix_np.shape[0]}, 选择帧数: {args.k}\n\n")
            
            # 按相似度总和排序
            sorted_methods = sorted(results.items(), key=lambda x: x[1]["similarity_sum"])
            
            f.write("按相似度总和排序（从小到大）:\n")
            for i, (method_name, result) in enumerate(sorted_methods):
                f.write(f"{i+1}. {method_name}: 相似度总和={result['similarity_sum']:.4f}, "
                        f"平均距离={result['avg_distance']:.2f}, 时间={result['time']:.2f}秒\n")
                f.write(f"   选中帧: {result['selected']}\n\n")
        
        print(f"\n各算法比较结果已保存至: {os.path.join(args.output_path, 'method_comparison.txt')}")
        
        # 使用指定的方法继续处理
        if args.method in methods:
            selected_frames = methods[args.method]()
        else:
            print(f"未知方法: {args.method}，使用贪心算法")
            selected_frames = methods["greedy"]()
    else:
        # 记录开始时间
        start_time = time.time()
        
        # 根据选择的方法执行算法
        if args.method == "backtracking":
            print(f"使用回溯算法选择{args.k}帧，目标是找出最不相似的帧，时间限制：{args.time_limit}秒...")
            selected_frames = select_min_similarity_frames_backtracking(similarity_matrix_np, args.k, args.time_limit)
        elif args.method == "dp":
            print(f"使用动态规划算法选择{args.k}帧，目标是找出最不相似的帧...")
            selected_frames = select_min_similarity_frames_dp(similarity_matrix_np, args.k)
        elif args.method == "dp_optimized":
            print(f"使用优化的动态规划算法选择{args.k}帧，目标是找出最不相似的帧...")
            selected_frames = select_min_similarity_frames_dp_optimized(similarity_matrix_np, args.k)
        elif args.method == "dp_greedy":
            print(f"使用贪心启发的算法选择{args.k}帧，目标是找出最不相似的帧...")
            selected_frames = select_min_similarity_frames_dp_greedy(similarity_matrix_np, args.k)
        elif args.method == "uniform":
            print(f"使用均匀采样算法选择{args.k}帧...")
            selected_frames = select_min_similarity_frames_uniform(similarity_matrix_np, args.k)
        elif args.method == "random":
            print(f"使用随机选择算法选择{args.k}帧...")
            selected_frames = select_min_similarity_frames_random(similarity_matrix_np, args.k, args.random_seed)
        elif args.method == "first_last":
            print(f"使用首尾帧算法选择{args.k}帧...")
            selected_frames = select_min_similarity_frames_first_last(similarity_matrix_np, args.k)
        elif args.method == "brute_force":
            print(f"使用暴力穷举算法选择{args.k}帧...")
            selected_frames = select_min_similarity_frames_brute_force(similarity_matrix_np, args.k, args.max_combinations)
        else:
            print(f"使用贪心算法选择{args.k}帧，目标是找出最不相似的帧...")
            selected_frames = select_min_similarity_frames_greedy(similarity_matrix_np, args.k)
        
        # 计算执行时间
        elapsed_time = time.time() - start_time
        
        # 计算所选帧之间的相似度总和
        similarity_sum = calculate_similarity_sum(similarity_matrix_np, selected_frames)
        
        print(f"算法执行时间: {elapsed_time:.2f}秒")
        print(f"选中的帧索引: {sorted(selected_frames)}")
        print(f"选中帧之间的相似度总和: {similarity_sum:.4f}")
    
    # 可视化选中的帧在不同矩阵中的位置
    if len(matrices_to_compare) > 0:
        # 创建选中帧的掩码矩阵
        selected_mask = np.zeros_like(original_matrix)
        for i in selected_frames:
            for j in selected_frames:
                if i != j:
                    selected_mask[i, j] = 1
        
        # 可视化选中帧的位置
        mask_matrices = [selected_mask]
        mask_titles = ["Selected Frame Pairs"]
        
        visualize_multiple_matrices(
            mask_matrices,
            mask_titles,
            args.output_path,
            "selected_frames_mask.png"
        )
    
    # 计算所选帧的平均帧间距离
    sorted_indices = sorted(selected_frames)
    distances = [sorted_indices[i+1] - sorted_indices[i] for i in range(len(sorted_indices)-1)]
    avg_distance = sum(distances) / len(distances) if distances else 0
    min_distance = min(distances) if distances else 0
    max_distance = max(distances) if distances else 0
    
    print(f"选中帧的帧间距离统计:")
    print(f"  平均距离: {avg_distance:.2f}")
    print(f"  最小距离: {min_distance}")
    print(f"  最大距离: {max_distance}")
    
    # 将结果写入文件
    with open(os.path.join(args.output_path, "selected_frames.txt"), "w") as f:
        for idx in sorted(selected_frames):
            f.write(f"{idx}\n")
    
    # 如果有帧数据，保存选中的帧
    if frames is not None:
        save_selected_frames(frames, selected_frames, args.output_path, 
                            prefix="dissimilar", cols_per_row=args.cols_per_row)


if __name__ == "__main__":
    main() 