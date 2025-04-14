"""
运行QM7b图像表示的降维和聚类分析
"""

import os
import sys

def setup_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    print(f"已将 {current_dir} 添加到Python路径")

if __name__ == "__main__":
    setup_paths()

    # 只导入修改后的模块
    import main_image_analysis_modified as main_image_analysis

    # image_dir = r"D:\materproject\all-reps\QM7b\QM7b-image"
    # metadata_path = ""  # QM7b无溶解度，空字符串即可
    # output_path = r"D:\materproject\single-rep-rd\image\QM7b"

    # image_dir = r"D:\materproject\all-reps\ESOL\ESOL-image"
    # metadata_path = ""  # QM7b无溶解度，空字符串即可
    # output_path = r"D:\materproject\single-rep-rd\image\ESOL"

    image_dir = r"D:\materproject\all-reps\GO_qdots\GO_qdots-image"
    # metadata_path = ""  # QM7b无溶解度，空字符串即可
    output_path = r"D:\materproject\single-rep-rd\image\GO_qdots"

    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    if len(sys.argv) > 2:
        metadata_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = sys.argv[3]

    model_name = 'resnet18'
    if len(sys.argv) > 4:
        model_name = sys.argv[4]

    print(f"图像目录: {image_dir}")

    print(f"输出路径: {output_path}")
    print(f"CNN模型: {model_name}")

    results_df = main_image_analysis.run_image_dimensionality_reduction(
        image_dir,
        output_path,
        model_name=model_name
    )

    print("\n分析完成！所有降维和聚类结果已保存。")
