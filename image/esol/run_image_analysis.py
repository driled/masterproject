"""
运行ESOL分子图像表示的降维和ILS聚类分析
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
    from main_image_analysis import run_image_dimensionality_reduction

    # 设置输入和输出路径
    image_dir = r"D:\materproject\all-reps\ESOL\ESOL-image"
    metadata_path = r"D:\materproject\all-reps\ESOL\ESOL-table\esol.csv"  # 包含溶解度数据的CSV文件路径
    output_path = r"D:\materproject\single-rep-rd\image\ESOL"

    # 允许通过命令行参数修改路径
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    if len(sys.argv) > 2:
        metadata_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = sys.argv[3]

    # 允许通过命令行参数选择CNN模型
    model_name = 'resnet18'  # 默认使用ResNet18
    if len(sys.argv) > 4:
        model_name = sys.argv[4]

    print(f"图像目录: {image_dir}")
    print(f"元数据路径: {metadata_path}")
    print(f"输出路径: {output_path}")
    print(f"CNN模型: {model_name}")

    # 运行分析
    results_df, clustering_df = run_image_dimensionality_reduction(
        image_dir,
        metadata_path,
        output_path,
        model_name=model_name
    )

    print("\n分析完成！使用CNN特征和ILS聚类方法的结果已保存。")