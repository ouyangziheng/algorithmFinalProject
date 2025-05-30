# 视频帧关键帧获取

此项目包含两个主要组件：

1. Python脚本 `extract_clip_similarity.py`：从视频中提取帧，并使用CLIP模型计算帧间相似度矩阵，将结果保存为npy文件
2. C++程序 `max_similarity_frames`：从相似度矩阵中选择k帧，使得它们之间的两两相似度之和最小（即选择最不相似的帧）

## 依赖项 

### Python部分

注：此部分仅于获取矩阵有关

安装Python依赖：
```bash
pip install -r requirements.txt
```

### C++部分
- C++17兼容的编译器
- zlib

## 编译

使用提供的Makefile编译C++程序：

```bash
make
```

## 使用方法

### 步骤1：提取视频帧并计算CLIP相似度矩阵

```bash
python extract_clip_similarity.py --video_path <视频路径> --output_path <输出目录> [选项]
```

选项：
- `--width`: 调整大小后的视频帧宽度 (默认: 720)
- `--height`: 调整大小后的视频帧高度 (默认: 480)
- `--skip_frames_start`: 从开始跳过的帧数 (默认: 0)
- `--skip_frames_end`: 从结尾跳过的帧数 (默认: 0)
- `--frame_sample_step`: 采样帧的时间步长 (默认: 自动计算)
- `--max_num_frames`: 最大采样帧数 (默认: 101)
- `--device`: 计算设备 (cuda 或 cpu, 默认: cuda)
- `--prefix`: 输出文件前缀 (默认: similarity)

例如：
```bash
python extract_clip_similarity.py --video_path movie.mp4 --output_path output --max_num_frames 100
```

这将生成一个相似度矩阵文件 `output/similarity_matrix.npy`

### 步骤2：使用C++程序选择帧

```bash
./max_similarity_frames --similarity_matrix=<相似度矩阵路径> --k=<选择的帧数> [选项]
```

选项：
- `--k`: 要选择的帧数 (默认: 5)
- `--output_path`: 输出结果的路径 (默认: output)
- `--method`: 使用的算法 (默认: greedy)
  - 可选: backtracking, greedy, uniform, random, first_last, brute_force
- `--time_limit`: 回溯算法的时间限制（秒）(默认: 30)
- `--random_seed`: 随机算法的种子 (默认: 42)
- `--distance_weight`: 帧间距离的权重因子 (默认: 0.5)
- `--help`: 显示帮助信息

例如：
```bash
./max_similarity_frames --similarity_matrix=output/similarity_matrix.npy --k=10 --method=greedy
```

输出将保存在 `output/selected_frames.txt` 文件中，每行包含一个选中的帧索引

## 算法说明

C++程序实现了多种帧选择算法：

1. `greedy`：贪心算法，逐步添加能够最小化总相似度的帧
2. `backtracking`：回溯搜索算法，尝试找到全局最优解，但运行时间较长
3. `uniform`：均匀采样算法，不考虑相似度，只选择均匀分布的帧（对照）
4. `random`：随机选择算法，随机选择k帧，用作基线对比（对照）
5. `first_last`：首尾帧算法，选择第一帧、最后一帧，以及它们之间均匀分布的帧（对照）
6. `brute_force`：暴力穷举算法，尝试所有可能的组合找到全局最优解，适用于小规模问题(k<=10, n<=25)，作为其他算法的准确性基准，注意使用暴力算法的时候需要调整最大组合数目，程序会在超过最大组合数目时候退化成为贪心算法
