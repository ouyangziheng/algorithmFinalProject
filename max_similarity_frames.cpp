/**
 * max_similarity_frames.cpp
 *
 * 此程序用于从帧间相似度矩阵中选择k帧，使得它们之间的两两相似度之和最小（即选择最不相似的帧）。
 * 可以处理的规模：n在100左右，k在5~10之间。
 *
 * 注意：CLIP模型相关功能需要通过Python脚本生成相似度矩阵文件（.npy格式）。
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// 用于处理npy文件的简单库
#include "cnpy.h"

// 命名空间
namespace fs = std::filesystem;

// 类型定义
using SimilarityMatrix = std::vector<std::vector<double>>;
using SelectedFrames = std::vector<int>;
using TimePoint = std::chrono::high_resolution_clock::time_point;

/**
 * 计算两个时间点之间的秒数
 */
double calculate_elapsed_seconds(const TimePoint& start, const TimePoint& end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
               .count() /
           1000.0;
}

/**
 * 从npy文件加载相似度矩阵
 */
SimilarityMatrix load_similarity_matrix(const std::string& file_path) {
    std::cout << "从文件加载相似度矩阵: " << file_path << std::endl;

    // 加载npy文件
    cnpy::NpyArray arr = cnpy::npy_load(file_path);

    // 检查数组是否为2D
    if (arr.shape.size() != 2 || arr.shape[0] != arr.shape[1]) {
        throw std::runtime_error("相似度矩阵必须是方阵!");
    }

    // 获取矩阵维度
    size_t n = arr.shape[0];

    // 创建矩阵
    SimilarityMatrix matrix(n, std::vector<double>(n, 0.0));

    // 根据数据类型填充矩阵
    if (arr.word_size == sizeof(float)) {
        const float* data = arr.data<float>();
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                matrix[i][j] = static_cast<double>(data[i * n + j]);
            }
        }
    } else if (arr.word_size == sizeof(double)) {
        const double* data = arr.data<double>();
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                matrix[i][j] = data[i * n + j];
            }
        }
    } else {
        throw std::runtime_error("不支持的数据类型，仅支持float或double");
    }

    std::cout << "相似度矩阵大小: " << n << "x" << n << std::endl;
    return matrix;
}

/**
 * 将相似度矩阵保存为CSV文件（用于调试）
 */
void save_matrix_as_csv(const SimilarityMatrix& matrix,
                        const std::string& file_path) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return;
    }

    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    std::cout << "矩阵已保存为CSV: " << file_path << std::endl;
}

/**
 * 计算归一化相似度矩阵（非对角线元素）
 */
SimilarityMatrix normalize_similarity_matrix(const SimilarityMatrix& matrix) {
    size_t n = matrix.size();
    SimilarityMatrix normalized = matrix;

    // 找出非对角线元素的最大值和最小值
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i != j) {  // 非对角线元素
                min_val = std::min(min_val, matrix[i][j]);
                max_val = std::max(max_val, matrix[i][j]);
            }
        }
    }

    // 对非对角线元素进行归一化
    if (max_val > min_val) {  // 避免除以零
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i != j) {  // 只归一化非对角线元素
                    normalized[i][j] =
                        (matrix[i][j] - min_val) / (max_val - min_val);
                }
            }
        }
    }

    std::cout << "相似度统计: 最小值=" << min_val << ", 最大值=" << max_val
              << ", 归一化范围=[0,1]" << std::endl;
    return normalized;
}

/**
 * 根据帧间距离调整相似度矩阵
 */
SimilarityMatrix adjust_similarity_with_distance(
    const SimilarityMatrix& similarity_matrix, double distance_weight = 0.5,
    double max_distance_influence = 0.5) {
    (void)max_distance_influence;  // 防止未使用参数警告
    size_t n = similarity_matrix.size();
    SimilarityMatrix adjusted_matrix = similarity_matrix;

    // 创建距离矩阵：帧i和帧j之间的归一化距离
    std::vector<std::vector<double>> distance_matrix(
        n, std::vector<double>(n, 0.0));
    double max_distance = static_cast<double>(n - 1);  // 最大可能距离

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // 计算归一化距离 (0到1之间)
            distance_matrix[i][j] =
                std::abs(static_cast<int>(i) - static_cast<int>(j)) /
                max_distance;
        }
    }

    // 将距离因素整合到相似度矩阵中
    // 距离越大，对相似度的惩罚越大
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i != j) {  // 不调整对角线元素
                // 距离惩罚因子：距离越大，惩罚越大
                double distance_penalty = distance_matrix[i][j];

                // 根据权重调整相似度
                // 当distance_weight=0时，保持原始相似度
                // 当distance_weight=1时，最大程度考虑距离因素
                adjusted_matrix[i][j] =
                    (1 - distance_weight) * similarity_matrix[i][j] +
                    distance_weight *
                        (similarity_matrix[i][j] * (1 - distance_penalty));
            }
        }
    }

    return adjusted_matrix;
}

/**
 * 计算选中帧之间的相似度总和
 */
double calculate_similarity_sum(const SimilarityMatrix& similarity_matrix,
                                const SelectedFrames& indices) {
    double total = 0.0;
    for (size_t i = 0; i < indices.size(); ++i) {
        for (size_t j = i + 1; j < indices.size(); ++j) {
            total += similarity_matrix[indices[i]][indices[j]];
        }
    }
    return total;
}

/**
 * 贪心算法：选择k帧，使得这k帧之间的两两相似度之和最小
 */
SelectedFrames select_min_similarity_frames_greedy(
    const SimilarityMatrix& similarity_matrix, int k) {
    size_t n = similarity_matrix.size();
    if (k >= static_cast<int>(n)) {
        SelectedFrames result(n);
        std::iota(result.begin(), result.end(), 0);  // 0, 1, 2, ..., n-1
        return result;
    }

    // 初始化选择第一个帧
    SelectedFrames selected_indices = {0};

    // 逐步添加能够最小化总相似度的帧
    while (static_cast<int>(selected_indices.size()) < k) {
        double min_gain = std::numeric_limits<double>::max();
        int next_idx = -1;

        for (size_t i = 0; i < n; ++i) {
            // 如果帧已被选中，跳过
            if (std::find(selected_indices.begin(), selected_indices.end(),
                          i) != selected_indices.end()) {
                continue;
            }

            // 计算添加这一帧能带来的相似度增益
            double gain = 0.0;
            for (int j : selected_indices) {
                gain += similarity_matrix[i][j];
            }

            if (gain < min_gain) {
                min_gain = gain;
                next_idx = static_cast<int>(i);
            }
        }

        selected_indices.push_back(next_idx);
    }

    return selected_indices;
}

/**
 * 回溯搜索算法：选择k帧，使得这k帧之间的两两相似度之和最小
 */
SelectedFrames select_min_similarity_frames_backtracking(
    const SimilarityMatrix& similarity_matrix, int k, int time_limit = 30) {
    size_t n = similarity_matrix.size();
    if (k >= static_cast<int>(n)) {
        SelectedFrames result(n);
        std::iota(result.begin(), result.end(), 0);
        return result;
    }

    // 使用贪心算法获取初始解
    SelectedFrames greedy_solution =
        select_min_similarity_frames_greedy(similarity_matrix, k);
    double greedy_similarity =
        calculate_similarity_sum(similarity_matrix, greedy_solution);

    // 存储最佳结果
    SelectedFrames best_frames = greedy_solution;
    double min_similarity = greedy_similarity;

    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();

    // 预计算每个节点与其他节点的平均相似度，用于选择优先搜索的节点顺序
    std::vector<std::pair<int, double>> avg_similarities;
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < n; ++j) {
            if (i != j) {
                sum += similarity_matrix[i][j];
            }
        }
        double avg_sim = sum / (n - 1);  // 不包括自己
        avg_similarities.push_back({static_cast<int>(i), avg_sim});
    }

    // 按平均相似度升序排序，优先考虑相似度低的节点
    std::sort(
        avg_similarities.begin(), avg_similarities.end(),
        [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
            return a.second < b.second;
        });

    std::vector<int> search_order;
    for (const auto& p : avg_similarities) {
        search_order.push_back(p.first);
    }

    // 用于跟踪时间限制
    bool time_limit_reached = false;

    // 回溯函数（使用引用捕获）
    std::function<bool(SelectedFrames&, std::vector<int>&, double)> backtrack =
        [&](SelectedFrames& selected, std::vector<int>& remaining,
            double current_similarity) -> bool {
        // 检查时间限制
        auto current_time = std::chrono::high_resolution_clock::now();
        if (calculate_elapsed_seconds(start_time, current_time) > time_limit) {
            std::cout << "回溯搜索达到时间限制 " << time_limit
                      << " 秒，返回当前最佳结果" << std::endl;
            time_limit_reached = true;
            return true;  // 表示达到时间限制
        }

        // 达到要求的帧数
        if (selected.size() == static_cast<size_t>(k)) {
            if (current_similarity < min_similarity) {
                min_similarity = current_similarity;
                best_frames = selected;
            }
            return false;
        }

        // 剪枝：如果剩余节点不足以满足要求
        if (selected.size() + remaining.size() < static_cast<size_t>(k)) {
            return false;
        }

        // 剪枝：即使选择相似度最小的组合，也无法超过当前最优解
        if (current_similarity >= min_similarity) {
            return false;
        }

        // 遍历剩余节点
        for (size_t i = 0; i < remaining.size(); ++i) {
            int node = remaining[i];

            // 计算添加此节点后的相似度增加量
            double new_similarity = current_similarity;
            for (int idx : selected) {
                new_similarity += similarity_matrix[node][idx];
            }

            // 如果当前相似度已经超过最小值，剪枝
            if (new_similarity >= min_similarity) {
                continue;
            }

            // 选择当前节点
            selected.push_back(node);

            // 创建新的remaining列表（不包含当前节点）
            std::vector<int> new_remaining;
            for (size_t j = 0; j < remaining.size(); ++j) {
                if (j != i) {
                    new_remaining.push_back(remaining[j]);
                }
            }

            // 递归
            bool limit_reached =
                backtrack(selected, new_remaining, new_similarity);
            if (limit_reached) {
                return true;
            }

            // 回溯
            selected.pop_back();
        }

        return false;
    };

    // 开始回溯搜索
    SelectedFrames selected;
    std::vector<int> initial_remaining = search_order;
    backtrack(selected, initial_remaining, 0.0);

    // 如果没有完成搜索，输出一条消息
    if (time_limit_reached) {
        std::cout << "警告：回溯搜索未完成，返回的可能不是最优解" << std::endl;
    }

    return best_frames;
}

/**
 * 均匀采样算法：不考虑相似度，只选择均匀分布的帧
 */
SelectedFrames select_min_similarity_frames_uniform(
    const SimilarityMatrix& similarity_matrix, int k) {
    size_t n = similarity_matrix.size();
    if (k >= static_cast<int>(n)) {
        SelectedFrames result(n);
        std::iota(result.begin(), result.end(), 0);
        return result;
    }

    // 计算步长，使得选择的帧均匀分布
    double step = static_cast<double>(n) / k;

    // 选择帧
    SelectedFrames selected;
    for (int i = 0; i < k; ++i) {
        // 对于每个位置，选择最接近的整数帧索引
        int idx = std::min(static_cast<int>(n) - 1, static_cast<int>(i * step));
        selected.push_back(idx);
    }

    return selected;
}

/**
 * 随机选择算法：随机选择k帧
 */
SelectedFrames select_min_similarity_frames_random(
    const SimilarityMatrix& similarity_matrix, int k, int seed = 42) {
    size_t n = similarity_matrix.size();
    if (k >= static_cast<int>(n)) {
        SelectedFrames result(n);
        std::iota(result.begin(), result.end(), 0);
        return result;
    }

    // 设置随机数生成器
    std::mt19937 gen(seed);

    // 创建所有可能的帧索引
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // 随机打乱
    std::shuffle(indices.begin(), indices.end(), gen);

    // 选择前k个
    SelectedFrames selected(indices.begin(), indices.begin() + k);
    return selected;
}

/**
 * 首尾帧算法：选择第一帧、最后一帧，以及它们之间均匀分布的帧
 */
SelectedFrames select_min_similarity_frames_first_last(
    const SimilarityMatrix& similarity_matrix, int k) {
    size_t n = similarity_matrix.size();
    if (k >= static_cast<int>(n)) {
        SelectedFrames result(n);
        std::iota(result.begin(), result.end(), 0);
        return result;
    }

    if (k == 1) {
        return {0};  // 只选第一帧
    }

    SelectedFrames selected = {0, static_cast<int>(n) - 1};  // 第一帧和最后一帧

    // 如果k>2，在中间均匀选择剩余的帧
    if (k > 2) {
        double step = static_cast<double>(n - 1) / (k - 1);
        for (int i = 1; i < k - 1; ++i) {
            int idx =
                std::min(static_cast<int>(n) - 1, static_cast<int>(i * step));
            selected.push_back(idx);
        }
    }

    // 排序结果
    std::sort(selected.begin(), selected.end());
    return selected;
}

/**
 * 暴力穷举算法：尝试所有可能的组合，保证找到全局最优解
 * 只适用于较小规模的问题，因为复杂度为O(n choose k)
 */
SelectedFrames select_min_similarity_frames_brute_force(
    const SimilarityMatrix& similarity_matrix, int k,
    long long max_combinations = 100000000000) {
    size_t n = similarity_matrix.size();
    if (k >= static_cast<int>(n)) {
        SelectedFrames result(n);
        std::iota(result.begin(), result.end(), 0);
        return result;
    }

    // 计算组合数，避免超出限制
    size_t num_combinations = 1;
    for (int i = n, j = 1; j <= k; i--, j++) {
        num_combinations = num_combinations * i / j;
        if (num_combinations > static_cast<size_t>(max_combinations)) {
            std::cout << "警告: 暴力算法需要评估" << num_combinations
                      << "种组合，超过限制" << max_combinations << std::endl;
            std::cout << "改用贪心算法..." << std::endl;
            return select_min_similarity_frames_greedy(similarity_matrix, k);
        }
    }

    std::cout << "暴力算法正在评估全部" << num_combinations << "种组合..."
              << std::endl;

    // 保存最佳结果
    double min_similarity = std::numeric_limits<double>::max();
    SelectedFrames best_combination(k);

    // 生成用于组合的索引数组 [0,1,2,...,n-1]
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // 创建用于组合的辅助数组
    std::vector<bool> selector(n, false);
    for (int i = 0; i < k; ++i) {
        selector[n - 1 - i] = true;  // 选择最后k个元素
    }

    // 使用std::next_permutation生成所有组合
    do {
        SelectedFrames combination;
        for (size_t i = 0; i < n; ++i) {
            if (selector[i]) {
                combination.push_back(indices[i]);
            }
        }

        double similarity_sum =
            calculate_similarity_sum(similarity_matrix, combination);

        if (similarity_sum < min_similarity) {
            min_similarity = similarity_sum;
            best_combination = combination;
        }
    } while (std::next_permutation(selector.begin(), selector.end()));

    std::cout << "暴力算法完成，找到最优解: 相似度总和 = " << min_similarity
              << std::endl;
    return best_combination;
}

/**
 * 保存选中的帧索引到文件
 */
void save_selected_frames(const SelectedFrames& frames,
                          const std::string& output_path) {
    // 确保输出目录存在
    fs::create_directories(output_path);

    // 排序帧索引
    SelectedFrames sorted_frames = frames;
    std::sort(sorted_frames.begin(), sorted_frames.end());

    // 保存到文件
    std::string file_path = output_path + "/selected_frames.txt";
    std::ofstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return;
    }

    for (int idx : sorted_frames) {
        file << idx << "\n";
    }

    file.close();
    std::cout << "选中的帧索引已保存至: " << file_path << std::endl;
}

/**
 * 帮助信息
 */
void print_help() {
    std::cout << "用法: max_similarity_frames [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --k=NUM                    要选择的帧数 (默认: 5)"
              << std::endl;
    std::cout << "  --output_path=PATH         输出结果的路径 (默认: output)"
              << std::endl;
    std::cout << "  --method=METHOD            使用的算法 (默认: greedy)"
              << std::endl;
    std::cout << "                             可选: backtracking, greedy, "
                 "uniform, random, first_last, brute_force"
              << std::endl;
    std::cout
        << "  --time_limit=SECONDS       回溯算法的时间限制（秒）(默认: 30)"
        << std::endl;
    std::cout << "  --random_seed=NUM          随机算法的种子 (默认: 42)"
              << std::endl;
    std::cout << "  --distance_weight=NUM      帧间距离的权重因子 (默认: 0.5)"
              << std::endl;
    std::cout << "  --similarity_matrix=PATH   预先计算的相似度矩阵文件路径 "
                 "(.npy格式)"
              << std::endl;
    std::cout << "  --help                     显示此帮助信息" << std::endl;
}

/**
 * 命令行参数解析
 */
std::map<std::string, std::string> parse_args(int argc, char* argv[]) {
    std::map<std::string, std::string> args;

    // 设置默认值
    args["k"] = "5";
    args["output_path"] = "output";
    args["method"] = "greedy";
    args["time_limit"] = "30";
    args["random_seed"] = "42";
    args["distance_weight"] = "0.5";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            args["help"] = "true";
            return args;
        }

        size_t pos = arg.find('=');
        if (pos != std::string::npos) {
            std::string key = arg.substr(0, pos);
            std::string value = arg.substr(pos + 1);

            // 删除前导的--
            if (key.substr(0, 2) == "--") {
                key = key.substr(2);
            }

            args[key] = value;
        }
    }

    return args;
}

/**
 * 主函数
 */
int main(int argc, char* argv[]) {
    // 解析命令行参数
    auto args = parse_args(argc, argv);

    // 显示帮助信息
    if (args.find("help") != args.end()) {
        print_help();
        return 0;
    }

    // 检查必需的参数
    if (args.find("similarity_matrix") == args.end()) {
        std::cerr << "错误: 缺少必需的参数 --similarity_matrix" << std::endl;
        print_help();
        return 1;
    }

    try {
        // 获取参数
        int k = std::stoi(args["k"]);
        std::string output_path = args["output_path"];
        std::string method = args["method"];
        int time_limit = std::stoi(args["time_limit"]);
        int random_seed = std::stoi(args["random_seed"]);
        double distance_weight = std::stod(args["distance_weight"]);
        std::string similarity_matrix_path = args["similarity_matrix"];

        // 确保输出目录存在
        fs::create_directories(output_path);

        // 加载相似度矩阵
        SimilarityMatrix similarity_matrix =
            load_similarity_matrix(similarity_matrix_path);

        // 计算归一化矩阵
        SimilarityMatrix normalized_matrix =
            normalize_similarity_matrix(similarity_matrix);

        // 如果指定了距离权重，则调整相似度矩阵
        if (distance_weight > 0) {
            std::cout << "应用帧间距离调整，距离权重: " << distance_weight
                      << std::endl;
            similarity_matrix = adjust_similarity_with_distance(
                similarity_matrix, distance_weight);
        }

        // 记录开始时间
        auto start_time = std::chrono::high_resolution_clock::now();

        // 根据选择的方法执行算法
        SelectedFrames selected_frames;

        if (method == "backtracking") {
            std::cout << "使用回溯算法选择" << k
                      << "帧，目标是找出最不相似的帧，时间限制：" << time_limit
                      << "秒..." << std::endl;
            selected_frames = select_min_similarity_frames_backtracking(
                similarity_matrix, k, time_limit);
        } else if (method == "uniform") {
            std::cout << "使用均匀采样算法选择" << k << "帧..." << std::endl;
            selected_frames =
                select_min_similarity_frames_uniform(similarity_matrix, k);
        } else if (method == "random") {
            std::cout << "使用随机选择算法选择" << k << "帧..." << std::endl;
            selected_frames = select_min_similarity_frames_random(
                similarity_matrix, k, random_seed);
        } else if (method == "first_last") {
            std::cout << "使用首尾帧算法选择" << k << "帧..." << std::endl;
            selected_frames =
                select_min_similarity_frames_first_last(similarity_matrix, k);
        } else if (method == "brute_force") {
            std::cout << "使用暴力穷举算法选择" << k << "帧..." << std::endl;
            selected_frames = select_min_similarity_frames_brute_force(
                similarity_matrix, k, 1000000000);
        } else {
            std::cout << "使用贪心算法选择" << k
                      << "帧，目标是找出最不相似的帧..." << std::endl;
            selected_frames =
                select_min_similarity_frames_greedy(similarity_matrix, k);
        }

        // 计算执行时间
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_time = calculate_elapsed_seconds(start_time, end_time);

        // 计算所选帧之间的相似度总和
        double similarity_sum =
            calculate_similarity_sum(similarity_matrix, selected_frames);

        // 计算所选帧的平均帧间距离
        std::vector<int> sorted_indices = selected_frames;
        std::sort(sorted_indices.begin(), sorted_indices.end());

        std::vector<int> distances;
        for (size_t i = 0; i < sorted_indices.size() - 1; ++i) {
            distances.push_back(sorted_indices[i + 1] - sorted_indices[i]);
        }

        double avg_distance = 0.0;
        if (!distances.empty()) {
            avg_distance =
                std::accumulate(distances.begin(), distances.end(), 0.0) /
                distances.size();
        }

        int min_distance =
            distances.empty()
                ? 0
                : *std::min_element(distances.begin(), distances.end());
        int max_distance =
            distances.empty()
                ? 0
                : *std::max_element(distances.begin(), distances.end());

        // 输出结果
        std::cout << "算法执行时间: " << elapsed_time << "秒" << std::endl;

        std::cout << "选中的帧索引: ";
        for (size_t i = 0; i < selected_frames.size(); ++i) {
            std::cout << selected_frames[i];
            if (i < selected_frames.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;

        std::cout << "选中帧之间的相似度总和: " << similarity_sum << std::endl;

        std::cout << "选中帧的帧间距离统计:" << std::endl;
        std::cout << "  平均距离: " << avg_distance << std::endl;
        std::cout << "  最小距离: " << min_distance << std::endl;
        std::cout << "  最大距离: " << max_distance << std::endl;

        // 保存选中的帧索引
        save_selected_frames(selected_frames, output_path);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
}