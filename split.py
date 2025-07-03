import numpy as np
from plyfile import PlyData, PlyElement
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

import random

def split_ply_by_fitted_plane(input_path, output_dir, grid_shape=(5, 3), max_points=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 Loading PLY: {input_path}")
    plydata = PlyData.read(input_path)
    data = plydata.elements[0].data
    total_points = len(data)

    if max_points and total_points > max_points:
        print(f"⚠️ Limiting to first {max_points:,} points (of {total_points:,})")
        data = data[:max_points]
        total_points = max_points

    # 读取坐标
    x = np.asarray(data['x'], dtype=np.float32)
    y = np.asarray(data['y'], dtype=np.float32)
    z = np.asarray(data['z'], dtype=np.float32)
    xyz_all = np.stack([x, y, z], axis=1)

    # 打印异常检查
    print("🔍 Coordinate summary:")
    print(f"  X: {x.min():.2f} ~ {x.max():.2f}")
    print(f"  Y: {y.min():.2f} ~ {y.max():.2f}")
    print(f"  Z: {z.min():.2f} ~ {z.max():.2f}")

    # ✨ 清洗非法点：NaN / Inf / 极大值
    coord_limit = 1e4  # 10 km 限制
    valid_mask = (
        np.all(np.isfinite(xyz_all), axis=1) &
        (np.abs(xyz_all) < coord_limit).all(axis=1)
    )

    valid_indices = np.flatnonzero(valid_mask)
    xyz = xyz_all[valid_mask]
    print(f"✅ Valid points after filtering: {len(xyz):,} / {total_points:,}")

    if len(xyz) < 100:
        raise ValueError("🚫 Too few valid points after filtering! Check coordinate range or input data.")

    # ✨ PCA 平面拟合
    print("📐 Performing PCA...")
    mean = xyz.mean(axis=0)
    centered = xyz - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    sorted_idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sorted_idx]

    u_axis = eigvecs[:, 0]  # 主方向
    v_axis = eigvecs[:, 1]  # 次主方向

    # 投影到平面 (u, v)
    u_coords = centered @ u_axis
    v_coords = centered @ v_axis

    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()
    u_bins = np.linspace(u_min, u_max, grid_shape[0] + 1)
    v_bins = np.linspace(v_min, v_max, grid_shape[1] + 1)

    print("📊 Assigning points to grid...")
    grid_indices = [[[] for _ in range(grid_shape[0])] for _ in range(grid_shape[1])]

    for i in range(len(xyz)):
        u, v = u_coords[i], v_coords[i]
        u_idx = np.searchsorted(u_bins, u, side='right') - 1
        v_idx = np.searchsorted(v_bins, v, side='right') - 1
        u_idx = min(max(u_idx, 0), grid_shape[0] - 1)
        v_idx = min(max(v_idx, 0), grid_shape[1] - 1)
        grid_indices[v_idx][u_idx].append(i)

    print("💾 Exporting sub-ply files...")
    for v in range(grid_shape[1]):
        for u in range(grid_shape[0]):
            local_idxs = grid_indices[v][u]
            if not local_idxs:
                print(f"  ⛔ Skipping empty cell ({u}, {v})")
                continue

            real_idxs = valid_indices[local_idxs]  # 映射到原始 data 中索引
            sub_data = data[real_idxs]
            el = PlyElement.describe(sub_data, 'vertex')
            filename = f"part_u{u}_v{v}.ply"
            out_path = os.path.join(output_dir, filename)
            PlyData([el], text=False).write(out_path)
            print(f"  ✅ Saved {len(local_idxs):,} points to {filename}")

    print("🎉 Done!")

def visualize_split_ply_grid(folder_path, sample_per_part=5000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    part_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".ply")])
    num_parts = len(part_files)
    print(f"Found {num_parts} parts.")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colormap = plt.cm.get_cmap('tab20', num_parts)

    for i, filename in enumerate(part_files):
        full_path = os.path.join(folder_path, filename)
        plydata = PlyData.read(full_path)
        data = plydata.elements[0].data

        # 提取 xyz
        x = np.asarray(data['x'])
        y = np.asarray(data['y'])
        z = np.asarray(data['z'])
        xyz = np.stack([x, y, z], axis=1)

        # 采样部分点
        N = len(xyz)
        sample_n = min(sample_per_part, N)
        idxs = np.random.choice(N, sample_n, replace=False)
        xyz_sampled = xyz[idxs]

        color = colormap(i)

        ax.scatter(
            xyz_sampled[:, 0], xyz_sampled[:, 1], xyz_sampled[:, 2],
            color=color, label=filename.replace(".ply", ""),
            s=2, alpha=0.8
        )

    ax.set_title("3D Visualization of Split PLY Grid")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    split_ply_by_fitted_plane(
        input_path="E:/tengbei220v_ply/point_cloud.ply",
        output_dir="E:/tengbei220v_ply/split_parts100000000",
        grid_shape=(5, 3),
        max_points=100000000,  # 或设置为 20_000_000 测试运行
    )

    visualize_split_ply_grid("E:/tengbei220v_ply/split_parts100000000", sample_per_part=2000)
