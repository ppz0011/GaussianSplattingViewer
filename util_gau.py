import numpy as np
from plyfile import PlyData, PlyElement
from dataclasses import dataclass

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations


def mask_to_indices(mask):
    return np.where(mask)[0]  



@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray
    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-1]


    @staticmethod
    def static_get_bbox_mask(xyz: np.ndarray, 
                        bbox_min: np.ndarray,
                        bbox_max: np.ndarray) -> np.ndarray:
        """
        生成包围盒内点的布尔掩码
        
        参数:
            xyz: 点云坐标 (N,3)
            bbox_min: 包围盒最小值坐标 (3,)
            bbox_max: 包围盒最大值坐标 (3,)
        
        返回:
            布尔数组(True表示点在包围盒内)
        """
        return np.all((xyz >= bbox_min) & (xyz <= bbox_max), axis=1)

    def crop_with_indices(self, indices: np.ndarray):
        """
        基于整数索引裁剪
        """
        return self.__class__(
            xyz=self.xyz[indices],
            rot=self.rot[indices],
            scale=self.scale[indices],
            opacity=self.opacity[indices],
            sh=self.sh[indices]
        )
    
    def crop_with_mask(self, mask: np.ndarray):
        """
        基于布尔索引裁剪
        """
        return self.crop_with_indices(mask_to_indices(mask))


def naive_gaussian():
    gau_xyz = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)
    gau_rot = np.array([
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0
    ]).astype(np.float32).reshape(-1, 4)
    gau_s = np.array([
        0.03, 0.03, 0.03,
        0.2, 0.03, 0.03,
        0.03, 0.2, 0.03,
        0.03, 0.03, 0.2
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = np.array([
        1, 0, 1, 
        1, 0, 0, 
        0, 1, 0, 
        0, 0, 1, 
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = (gau_c - 0.5) / 0.28209
    gau_a = np.array([
        1, 1, 1, 1
    ]).astype(np.float32).reshape(-1, 1)
    return GaussianData(
        gau_xyz,
        gau_rot,
        gau_s,
        gau_a,
        gau_c
    )


def load_ply(path, is_center_at_origin=True):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    
    # ⬇️ 平移点云，将中心移到坐标原点
    if is_center_at_origin:
        centering_offset = np.mean(xyz, axis=0)
        xyz = xyz - centering_offset
        
    else:
        centering_offset = np.zeros(3, dtype=np.float32)
        
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))

    # ⬇️ 推断 SH 阶数
    num_extra_coeffs = len(extra_f_names)
    sh_coeffs_per_channel = (num_extra_coeffs // 3) + 1  # +1 for DC
    max_sh_degree = int(np.sqrt(sh_coeffs_per_channel) - 1)
    assert sh_coeffs_per_channel == (max_sh_degree + 1)**2

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # Reshape and transpose
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    # Process scale and rotation
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # Final processing
    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales).astype(np.float32)
    opacities = 1 / (1 + np.exp(-opacities))  # sigmoid
    opacities = opacities.astype(np.float32)
    
    shs = np.concatenate([features_dc.reshape(-1, 3), features_extra.reshape(len(features_dc), -1)], axis=-1)
    shs = shs.astype(np.float32)

    return GaussianData(xyz, rots, scales, opacities, shs), centering_offset



def plot_point_distribution(gaussian_data: GaussianData, title="Gaussian Point Cloud Position Distribution",
                             sample_size=5000):
    xyz = gaussian_data.xyz
    N = len(xyz)

    # 随机采样点
    if N > sample_size:
        indices = np.random.choice(N, sample_size, replace=False)
        xyz_sampled = xyz[indices]
    else:
        xyz_sampled = xyz

    # 计算边界盒角点
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    corner_points = np.array(list(product(*zip(mins, maxs))))

    # 合并采样点和角点
    all_points = np.concatenate([xyz_sampled, corner_points], axis=0)

    # 绘图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    ax.scatter(
        xyz_sampled[:, 0], xyz_sampled[:, 1], xyz_sampled[:, 2],
        color='blue', marker='o', s=3, alpha=0.6
    )

    # 绘制边界角点
    ax.scatter(
        corner_points[:, 0], corner_points[:, 1], corner_points[:, 2],
        color='red', marker='x', s=30, label='Bounding Box Corners'
    )

    # 添加边界框线段（12条边）
    for start, end in combinations(corner_points, 2):
        diff = np.abs(start - end)
        if np.count_nonzero(diff) == 1:  # 只在一个维度上不同 → 是合法边
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color='black', linewidth=1
            )

    # 轴标签与图例
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    plt.show()




def save_ply(gaussian_data: GaussianData, path: str):
    """
    Save GaussianData to a .ply file.
    
    Args:
        path: Output .ply file path
        gaussian_data: GaussianData object to save
    """
    max_sh_degree = 3
    # Prepare data in the same format as loaded by load_ply
    xyz = gaussian_data.xyz
    rots = gaussian_data.rot
    scales = np.log(gaussian_data.scale)  # Inverse of exp in load_ply
    opacities = -np.log(1.0 / gaussian_data.opacity - 1.0)  # Inverse of sigmoid
    
    # Split SH coefficients into DC and rest
    sh_dim = 3 * ((max_sh_degree + 1) ** 2)
    assert gaussian_data.sh_dim == sh_dim, f"Expected SH dim {sh_dim}, got {gaussian_data.sh_dim}"
    
    features_dc = gaussian_data.sh[:, :3].reshape(-1, 3, 1)
    features_extra = gaussian_data.sh[:, 3:].reshape(-1, (max_sh_degree + 1) ** 2 - 1, 3)
    features_extra = np.transpose(features_extra, [0, 2, 1]).reshape(-1, 3 * ((max_sh_degree + 1) ** 2 - 1))

    # Create the structured array for PLY data
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4')
    ]
    
    # Add scale attributes
    for i in range(3):
        dtype.append((f'scale_{i}', 'f4'))
    
    # Add rotation attributes
    for i in range(4):
        dtype.append((f'rot_{i}', 'f4'))
    
    # Add spherical harmonics rest attributes
    for i in range(3 * ((max_sh_degree + 1) ** 2 - 1)):
        dtype.append((f'f_rest_{i}', 'f4'))
    
    # Create and fill the structured array
    num_points = len(gaussian_data)
    vertex_data = np.zeros(num_points, dtype=dtype)
    
    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    
    vertex_data['f_dc_0'] = features_dc[:, 0, 0]
    vertex_data['f_dc_1'] = features_dc[:, 1, 0]
    vertex_data['f_dc_2'] = features_dc[:, 2, 0]
    
    vertex_data['opacity'] = opacities[:, 0]
    
    for i in range(3):
        vertex_data[f'scale_{i}'] = scales[:, i]
    
    for i in range(4):
        vertex_data[f'rot_{i}'] = rots[:, i]
    
    for i in range(features_extra.shape[1]):
        vertex_data[f'f_rest_{i}'] = features_extra[:, i]
    
    # Create the PlyElement and save
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    PlyData([vertex_element], text=False).write(path)




if __name__ == "__main__":
    gs = load_ply("C:\\Users\\MSI_NB\\Downloads\\viewers\\models\\train\\point_cloud\\iteration_7000\\point_cloud.ply")
    a = gs.flat()
    print(a.shape)
