"""
ESOL数据集降维和迭代标签溢出聚类分析
"""

import os
import sys


def setup_paths():
    """设置模块路径"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 将当前目录添加到Python路径
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    print(f"已将 {current_dir} 添加到Python路径")


if __name__ == "__main__":
    # 设置路径
    setup_paths()

    # 导入主模块
    from main import run_dimensionality_reduction

    # 设置输入和输出路径
    input_path = r"D:\materproject\all-reps\ESOL\ESOL-table"
    output_path = r"D:\materproject\single-rep-rd\table\ESOL"

    # 允许通过命令行参数修改路径
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")

    # 运行分析
    results_df, clustering_df = run_dimensionality_reduction(input_path, output_path)

    print("\n分析完成！")