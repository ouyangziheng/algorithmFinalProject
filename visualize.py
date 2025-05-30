"""
这个脚本用于计算视频帧之间的CLIP相似度，并可视化相似度矩阵。

使用方法:
    python clip_similarity.py --video_path /path/to/video.mp4 --output_path /path/to/output
    
    python /mnt/weka/hw_workspace/ziheng_workspace/papers/CogVideo/inference/clip_similarity.py --video_path /mnt/weka/hw_workspace/ziheng_workspace/papers/CogVideo/output_1746972884.2423837.mp4 --output_path /mnt/weka/hw_workspace/ziheng_workspace/papers/CogVideo/output/visualization
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
from typing import List, Optional, Tuple, Union

import decord
from transformers import CLIPProcessor, CLIPModel

def get_args():
    parser = argparse.ArgumentParser(description="计算视频帧之间的CLIP相似度")
    
    parser.add_argument(
        "--video_path", type=str, required=True, help="视频文件路径"
    )
    parser.add_argument(
        "--output_path", type=str, default="output", help="输出结果的路径"
    )
    parser.add_argument(
        "--skip_frames_start", type=int, default=0, help="从开始跳过的帧数"
    )
    parser.add_argument(
        "--skip_frames_end", type=int, default=0, help="从结尾跳过的帧数"
    )
    parser.add_argument(
        "--frame_sample_step", type=int, default=None, help="采样帧的时间步长"
    )
    parser.add_argument(
        "--max_num_frames", type=int, default=81, help="最大采样帧数"
    )
    parser.add_argument(
        "--width", type=int, default=720, help="调整大小后的视频帧宽度"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="调整大小后的视频帧高度"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="计算设备"
    )
    parser.add_argument(
        "--k", type=int, default=10, help="选择的帧数量"
    )
    parser.add_argument(
        "--selection_mode", type=str, default="dissimilar", choices=["similar", "dissimilar"], 
        help="选择模式: 'similar'选择最相似的帧, 'dissimilar'选择最不相似的帧"
    )
    
    return parser.parse_args()

def get_video_frames(
    video_path: str,
    width: int,
    height: int,
    skip_frames_start: int,
    skip_frames_end: int,
    max_num_frames: int,
    frame_sample_step: Optional[int],
) -> torch.FloatTensor:
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
    video_name: str
):
    """
    可视化相似度矩阵并保存
    
    Args:
        similarity_matrix: 相似度矩阵 [F, F]
        output_path: 输出目录
        video_name: 视频名称
    """
    os.makedirs(output_path, exist_ok=True)
    
    # 转换为numpy数组
    sim_matrix_np = similarity_matrix.cpu().numpy()
    
    # 创建图像
    plt.figure(figsize=(10, 8))
    
    # 绘制热图
    plt.imshow(sim_matrix_np, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='CLIP Similarity')
    plt.title(f'CLIP Frame Similarity - {video_name}')
    plt.xlabel('Frame Index')
    plt.ylabel('Frame Index')
    
    # 保存图像
    output_file = os.path.join(output_path, f"{video_name}_clip_similarity.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"相似度矩阵已保存至: {output_file}")
    
    # 保存数值数据
    np_output_file = os.path.join(output_path, f"{video_name}_clip_similarity.npy")
    np.save(np_output_file, sim_matrix_np)
    print(f"相似度矩阵数据已保存至: {np_output_file}")
    
    # 创建归一化后的可视化图像
    visualize_normalized_similarity_matrix(sim_matrix_np, output_path, video_name)

def visualize_normalized_similarity_matrix(
    sim_matrix_np: np.ndarray,
    output_path: str,
    video_name: str
):
    """
    创建归一化的相似度矩阵可视化
    
    Args:
        sim_matrix_np: 相似度矩阵 numpy数组
        output_path: 输出目录
        video_name: 视频名称
    """
    # 获取非对角线元素
    mask = ~np.eye(sim_matrix_np.shape[0], dtype=bool)
    non_diagonal = sim_matrix_np[mask]
    
    # 计算均值和标准差
    mean_sim = np.mean(non_diagonal)
    std_sim = np.std(non_diagonal)
    min_sim = np.min(non_diagonal)
    max_sim = np.max(non_diagonal)
    
    print(f"相似度统计: 均值={mean_sim:.4f}, 标准差={std_sim:.4f}, 最小值={min_sim:.4f}, 最大值={max_sim:.4f}")
    
    # 创建三种不同的归一化图像
    
    # 1. 使用均值±3个标准差作为范围
    plt.figure(figsize=(10, 8))
    vmin = max(0, mean_sim - 3 * std_sim)
    vmax = min(1, mean_sim + 3 * std_sim)
    plt.imshow(sim_matrix_np, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label='CLIP Similarity (Normalized ±3σ)')
    plt.title(f'CLIP Similarity (Normalized ±3σ) - {video_name}')
    plt.xlabel('Frame Index')
    plt.ylabel('Frame Index')
    output_file = os.path.join(output_path, f"{video_name}_clip_similarity_norm_3sigma.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"均值±3标准差归一化相似度矩阵已保存至: {output_file}")
    
    # 2. 使用实际非对角线的最小/最大值作为范围
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix_np, cmap='viridis', vmin=min_sim, vmax=max_sim)
    plt.colorbar(label='CLIP Similarity (Min-Max Normalized)')
    plt.title(f'CLIP Similarity (Min-Max Normalized) - {video_name}')
    plt.xlabel('Frame Index')
    plt.ylabel('Frame Index')
    output_file = os.path.join(output_path, f"{video_name}_clip_similarity_norm_minmax.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"最小-最大值归一化相似度矩阵已保存至: {output_file}")
    
    # 3. 应用对数变换以增强差异（将值映射到[0,1]）
    # 先将相似度映射到[0,1]区间
    normalized_sim = (sim_matrix_np - min_sim) / (max_sim - min_sim + 1e-8)
    # 对角线保持为1
    np.fill_diagonal(normalized_sim, 1.0)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(normalized_sim, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='CLIP Similarity (Normalized [0,1])')
    plt.title(f'CLIP Similarity (Normalized [0,1]) - {video_name}')
    plt.xlabel('Frame Index')
    plt.ylabel('Frame Index')
    output_file = os.path.join(output_path, f"{video_name}_clip_similarity_norm_01.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[0,1]归一化相似度矩阵已保存至: {output_file}")

def plot_frame_to_frame_similarity(
    similarity_matrix: torch.Tensor,
    output_path: str,
    video_name: str
):
    """
    绘制相邻帧之间的相似度曲线
    
    Args:
        similarity_matrix: 相似度矩阵 [F, F]
        output_path: 输出目录
        video_name: 视频名称
    """
    os.makedirs(output_path, exist_ok=True)
    
    # 转换为numpy数组
    sim_matrix_np = similarity_matrix.cpu().numpy()
    num_frames = sim_matrix_np.shape[0]
    
    # 计算相邻帧之间的相似度
    frame_to_frame_sim = [sim_matrix_np[i, i+1] for i in range(num_frames-1)]
    
    # 获取相邻帧相似度的统计信息
    mean_sim = np.mean(frame_to_frame_sim)
    std_sim = np.std(frame_to_frame_sim)
    min_sim = np.min(frame_to_frame_sim)
    max_sim = np.max(frame_to_frame_sim)
    
    # 创建普通图像
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_frames), frame_to_frame_sim, marker='o', linestyle='-', color='blue')
    plt.title(f'Frame-to-Frame CLIP Similarity - {video_name}')
    plt.xlabel('Frame Index')
    plt.ylabel('Similarity to Previous Frame')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=mean_sim, color='r', linestyle='--', label=f'Average: {mean_sim:.4f}')
    plt.legend()
    output_file = os.path.join(output_path, f"{video_name}_frame_to_frame_similarity.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"帧间相似度曲线已保存至: {output_file}")
    
    # 创建归一化版本，y轴范围基于均值±3标准差
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_frames), frame_to_frame_sim, marker='o', linestyle='-', color='blue')
    plt.title(f'Frame-to-Frame CLIP Similarity (Normalized) - {video_name}')
    plt.xlabel('Frame Index')
    plt.ylabel('Similarity to Previous Frame')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=mean_sim, color='r', linestyle='--', label=f'Average: {mean_sim:.4f}')
    plt.ylim([max(0, mean_sim - 3 * std_sim), min(1, mean_sim + 3 * std_sim)])
    plt.legend()
    output_file = os.path.join(output_path, f"{video_name}_frame_to_frame_similarity_normalized.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"归一化帧间相似度曲线已保存至: {output_file}")
    
    # 还创建一个基于实际最小/最大值的版本
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_frames), frame_to_frame_sim, marker='o', linestyle='-', color='blue')
    plt.title(f'Frame-to-Frame CLIP Similarity (Min-Max) - {video_name}')
    plt.xlabel('Frame Index')
    plt.ylabel('Similarity to Previous Frame')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=mean_sim, color='r', linestyle='--', label=f'Average: {mean_sim:.4f}')
    plt.ylim([min_sim, max_sim])
    plt.legend()
    output_file = os.path.join(output_path, f"{video_name}_frame_to_frame_similarity_minmax.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"最小-最大值帧间相似度曲线已保存至: {output_file}")

def select_most_dissimilar_frames(similarity_matrix, k=5):
    """
    选择视频中最不相似的k张帧
    
    Args:
        similarity_matrix: 帧间相似度矩阵
        k: 要选择的帧数量
    
    Returns:
        选中的帧索引列表
    """
    num_frames = similarity_matrix.shape[0]
    if k >= num_frames:
        return list(range(num_frames))
    
    # 先选第一帧
    selected_indices = [0]
    
    # 不断选择与已选帧最大相似度最小的帧
    while len(selected_indices) < k:
        max_min_sim = float('inf')
        next_frame_idx = -1
        
        for i in range(num_frames):
            if i in selected_indices:
                continue
                
            # 计算与所有已选帧的最大相似度
            min_sim = min([similarity_matrix[i, j] for j in selected_indices])
            
            # 如果这个最大相似度比当前找到的更小，更新选择
            if min_sim < max_min_sim:
                max_min_sim = min_sim
                next_frame_idx = i
        
        selected_indices.append(next_frame_idx)
    
    return selected_indices

def select_most_similar_frames(similarity_matrix, k=5):
    """
    选择视频中最相似的k张帧
    
    Args:
        similarity_matrix: 帧间相似度矩阵
        k: 要选择的帧数量
    
    Returns:
        选中的帧索引列表
    """
    num_frames = similarity_matrix.shape[0]
    if k >= num_frames:
        return list(range(num_frames))
    
    # 先选第一帧
    selected_indices = [0]
    
    # 不断选择与已选帧平均相似度最大的帧
    while len(selected_indices) < k:
        max_avg_sim = -float('inf')
        next_frame_idx = -1
        
        for i in range(num_frames):
            if i in selected_indices:
                continue
                
            # 计算与所有已选帧的平均相似度
            avg_sim = sum([similarity_matrix[i, j] for j in selected_indices]) / len(selected_indices)
            
            # 如果这个平均相似度比当前找到的更大，更新选择
            if avg_sim > max_avg_sim:
                max_avg_sim = avg_sim
                next_frame_idx = i
        
        selected_indices.append(next_frame_idx)
    
    return selected_indices

def print_similarity_matrix_statistics(similarity_matrix, selected_indices=None):
    """
    打印相似度矩阵的详细统计信息
    
    Args:
        similarity_matrix: 相似度矩阵
        selected_indices: 选中的帧索引列表
    """
    # 打印整个相似度矩阵
    print("\n相似度矩阵:")
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(similarity_matrix)
    np.set_printoptions()  # 重置打印选项
    
    # 如果有选中的帧，打印它们之间的相似度
    if selected_indices and len(selected_indices) > 1:
        print("\n选中帧之间的相似度矩阵:")
        selected_sim = similarity_matrix[np.ix_(selected_indices, selected_indices)]
        np.set_printoptions(precision=4, suppress=True, linewidth=120)
        print(selected_sim)
        np.set_printoptions()  # 重置打印选项
        
        # 计算选中帧之间的平均相似度
        # 不考虑对角线元素(全为1)
        mask = ~np.eye(len(selected_indices), dtype=bool)
        selected_non_diag = selected_sim[mask]
        
        print(f"\n选中帧之间相似度统计:")
        print(f"  平均相似度: {np.mean(selected_non_diag):.4f}")
        print(f"  最小相似度: {np.min(selected_non_diag):.4f}")
        print(f"  最大相似度: {np.max(selected_non_diag):.4f}")
        print(f"  标准差: {np.std(selected_non_diag):.4f}")
        
        # 打印每对选中帧之间的相似度
        print("\n每对选中帧之间的相似度:")
        for i, idx1 in enumerate(selected_indices):
            for j, idx2 in enumerate(selected_indices):
                if i < j:  # 只打印矩阵的上三角部分
                    print(f"  帧 {idx1} 和帧 {idx2} 的相似度: {similarity_matrix[idx1, idx2]:.4f}")

def main():
    args = get_args()
    device = torch.device(args.device)
    import shutil   
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    
    # 获取视频帧
    print(f"正在读取视频: {args.video_path}")
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
    
    # 计算CLIP相似度
    print("正在计算CLIP相似度...")
    similarity_matrix = compute_clip_similarity(frames, args.device)
    
    # 生成文件名
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    
    # 可视化相似度矩阵
    print("正在生成可视化结果...")
    visualize_similarity_matrix(similarity_matrix, args.output_path, video_name)
    
    # 绘制帧间相似度曲线
    plot_frame_to_frame_similarity(similarity_matrix, args.output_path, video_name)
    
    # 选择帧
    num_frames = frames.shape[0]
    k = args.k
    similarity_matrix_np = similarity_matrix.cpu().numpy()
    
    if args.selection_mode == "dissimilar":
        selected_frames = select_most_dissimilar_frames(similarity_matrix_np, k=k)
        frame_type = "最不相似"
    else:  # similar
        selected_frames = select_most_similar_frames(similarity_matrix_np, k=k)
        frame_type = "最相似"
    
    print(f"选择的{frame_type}帧索引: {selected_frames}")
    
    # 打印相似度矩阵和选中帧之间的相似度
    print_similarity_matrix_statistics(similarity_matrix_np, selected_frames)
            
    # 创建保存选中帧的目录
    frames_dir_name = f"{args.selection_mode}_frames"
    selected_frames_dir = os.path.join(args.output_path, frames_dir_name)
    os.makedirs(selected_frames_dir, exist_ok=True)
    
    for idx in selected_frames:
        frame_image = Image.fromarray((frames[idx].cpu().numpy() * 255).astype(np.uint8))
        frame_path = os.path.join(args.output_path, f"{frames_dir_name}/{video_name}_frame_{idx}.png")
        frame_image.save(frame_path)
        print(f"已保存帧 {idx} 至: {frame_path}")
    
    # 保存一个包含所有选中帧的图像
    plt.figure(figsize=(15, 3 * k))
    for i, idx in enumerate(selected_frames):
        plt.subplot(k, 1, i+1)
        plt.imshow(frames[idx].cpu().numpy())
        plt.title(f"帧 {idx}")
        plt.axis('off')
    plt.tight_layout()
    all_frames_path = os.path.join(args.output_path, f"{frames_dir_name}/{video_name}_all_selected_frames.png")
    plt.savefig(all_frames_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"所有选中{frame_type}帧的组合图已保存至: {all_frames_path}")
    
    print("完成！")
    
    
    with open("frameIndex.txt", "w") as f:
        record = []
        for idx in selected_frames:
            record.append(idx)
        record = sorted(record)
        for idx in record:
            f.write(f"{idx}\n")

if __name__ == "__main__":
    main() 