import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
import time


def xyz_to_mol_robust(xyz_file):
    """
    将XYZ文件转换为原子和坐标
    """
    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    try:
        n_atoms = int(lines[0].strip())
        atoms = []
        coords = []

        for i in range(2, 2 + n_atoms):
            if i >= len(lines):
                break
            parts = lines[i].strip().split()
            if len(parts) >= 4:
                atom_symbol = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                atoms.append(atom_symbol)
                coords.append([x, y, z])

        return atoms, coords
    except Exception as e:
        print(f"解析文件 {xyz_file} 时出错: {str(e)}")
        return None, None


def create_simple_2d_image(atoms, coords, output_file, size=(500, 500)):
    """
    创建简单的2D分子图像
    """
    if atoms is None or coords is None:
        return False

    try:
        # 创建一个简单的2D图像
        fig, ax = plt.figure(figsize=(8, 8)), plt.gca()

        # 原子颜色映射
        atom_colors = {
            'C': 'black',
            'H': 'gray',
            'O': 'red',
            'N': 'blue',
            'S': 'gold',
            'P': 'orange',
            'F': 'green',
            'Cl': 'green',
            'Br': 'brown',
            'I': 'purple'
        }

        # 原子大小映射
        atom_sizes = {
            'C': 100,
            'H': 50,
            'O': 120,
            'N': 120,
            'S': 140,
            'P': 140,
            'F': 100,
            'Cl': 120,
            'Br': 140,
            'I': 160
        }

        # 提取x,y坐标（忽略z坐标，因为我们只生成2D图像）
        x = [coord[0] for coord in coords]
        y = [coord[1] for coord in coords]

        # 计算中心和缩放
        center_x, center_y = np.mean(x), np.mean(y)
        x_centered = [xi - center_x for xi in x]
        y_centered = [yi - center_y for yi in y]

        # 计算合适的缩放比例
        max_dim = max(max(abs(val) for val in x_centered), max(abs(val) for val in y_centered))
        scale = 0.9 / max_dim if max_dim > 0 else 1

        # 应用缩放
        x_scaled = [xi * scale for xi in x_centered]
        y_scaled = [yi * scale for yi in y_centered]

        # 绘制原子
        for i, atom in enumerate(atoms):
            # 提取元素符号
            element = ''.join([c for c in atom if c.isalpha()])
            color = atom_colors.get(element, 'gray')
            size = atom_sizes.get(element, 80)
            ax.scatter(x_scaled[i], y_scaled[i], c=color, s=size, edgecolors='black', zorder=10)

            # 添加元素标签
            ax.text(x_scaled[i], y_scaled[i], element, horizontalalignment='center',
                    verticalalignment='center', color='white' if element in ['C', 'Br', 'I'] else 'black',
                    fontweight='bold', fontsize=8, zorder=15)

        # 尝试绘制简单的键（基于距离）
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                # 提取元素符号
                el_i = ''.join([c for c in atoms[i] if c.isalpha()])
                el_j = ''.join([c for c in atoms[j] if c.isalpha()])

                # 计算2D距离
                dist_2d = np.sqrt((x_scaled[i] - x_scaled[j]) ** 2 + (y_scaled[i] - y_scaled[j]) ** 2)

                # 元素对的最大键长阈值（经验值）
                bond_threshold = {
                    ('C', 'C'): 0.3,
                    ('C', 'O'): 0.25,
                    ('C', 'N'): 0.25,
                    ('C', 'H'): 0.2,
                    ('O', 'H'): 0.2,
                    ('N', 'H'): 0.2,
                    ('C', 'S'): 0.3,
                    ('S', 'H'): 0.25,
                    ('O', 'O'): 0.25
                }

                # 获取这对元素的阈值
                key = tuple(sorted([el_i, el_j]))
                threshold = bond_threshold.get(key, 0.35) * scale  # 默认阈值

                # 如果距离小于阈值，绘制键
                if dist_2d < threshold:
                    # 在原子之间绘制线
                    ax.plot([x_scaled[i], x_scaled[j]], [y_scaled[i], y_scaled[j]], 'k-',
                            linewidth=1.5, alpha=0.8, zorder=5)

        # 设置绘图参数
        ax.set_aspect('equal')
        ax.axis('off')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        # 保存图像
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"创建2D图像时出错: {str(e)}")
        return False


def process_single_file(xyz_file, output_dir):
    """
    处理单个XYZ文件并创建2D图像
    """
    try:
        # 提取文件名（不带路径和扩展名）
        file_name = os.path.splitext(os.path.basename(xyz_file))[0]

        # 解析XYZ文件
        atoms, coords = xyz_to_mol_robust(xyz_file)

        # 创建2D图像
        output_file = os.path.join(output_dir, f"{file_name}.png")
        success = create_simple_2d_image(atoms, coords, output_file)

        return (xyz_file, success)
    except Exception as e:
        print(f"处理 {xyz_file} 时出现未预期的错误: {str(e)}")
        return (xyz_file, False)


def process_xyz_files_parallel(input_dir, output_dir, num_processes=None):
    """
    并行处理目录中的所有XYZ文件并创建2D图像
    """
    # 如果未指定进程数，使用CPU核心数减1
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有XYZ文件
    xyz_files = glob.glob(os.path.join(input_dir, '*.xyz'))
    total_files = len(xyz_files)
    print(f"在 {input_dir} 中找到 {total_files} 个XYZ文件")
    print(f"将使用 {num_processes} 个进程并行处理")

    # 创建进程池
    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 创建处理单个文件的偏函数
        process_file = partial(process_single_file, output_dir=output_dir)

        # 使用进程池并行处理文件
        results = []
        for i, result in enumerate(pool.imap_unordered(process_file, xyz_files)):
            results.append(result)
            # 显示进度
            if (i + 1) % 100 == 0 or (i + 1) == total_files:
                elapsed_time = time.time() - start_time
                files_per_second = (i + 1) / elapsed_time
                remaining_time = (total_files - (i + 1)) / files_per_second if files_per_second > 0 else 0
                print(f"进度: {i + 1}/{total_files} ({(i + 1) / total_files * 100:.1f}%) "
                      f"- 速度: {files_per_second:.1f}文件/秒 "
                      f"- 剩余时间: {remaining_time / 60:.1f}分钟")

    # 处理结果
    successful = sum(1 for _, success in results if success)
    failed = total_files - successful

    elapsed_time = time.time() - start_time
    print(f"\n处理完成! ")
    print(f"总计: {total_files} 文件")
    print(f"成功: {successful} 文件")
    print(f"失败: {failed} 文件")
    print(f"总耗时: {elapsed_time:.1f} 秒 (平均 {total_files / elapsed_time:.1f} 文件/秒)")


# 主函数
if __name__ == "__main__":
    input_dir = r"D:\materproject\project\Raw_data_files"
    output_dir = r"D:\materproject\all-reps\GO_qdots\GO_qdots-image"

    # 使用8个进程并行处理 (可以根据您的计算机性能调整)
    process_xyz_files_parallel(input_dir, output_dir, num_processes=8)
    print("完成!")