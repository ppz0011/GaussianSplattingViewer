from OpenGL import GL as gl
import util
import util_gau
import numpy as np

try:
    from OpenGL.raw.WGL.EXT.swap_control import wglSwapIntervalEXT
except:
    wglSwapIntervalEXT = None


_sort_buffer_xyz = None
_sort_buffer_gausid = None  # used to tell whether gaussian is reloaded

def _sort_gaussian_cpu(gaus, view_mat):
    xyz = np.asarray(gaus.xyz)
    view_mat = np.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = np.argsort(depth)
    index = index.astype(np.int32).reshape(-1, 1)
    return index


def _sort_gaussian_cupy(gaus, view_mat):
    import cupy as cp
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = cp.asarray(gaus.xyz)
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = cp.asarray(view_mat)

    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]

    index = cp.argsort(depth)
    index = index.astype(cp.int32).reshape(-1, 1)

    index = cp.asnumpy(index) # convert to numpy
    return index


def _sort_gaussian_torch(gaus, view_mat):
    global _sort_buffer_gausid, _sort_buffer_xyz
    if _sort_buffer_gausid != id(gaus):
        _sort_buffer_xyz = torch.tensor(gaus.xyz).cuda()
        _sort_buffer_gausid = id(gaus)

    xyz = _sort_buffer_xyz
    view_mat = torch.tensor(view_mat).cuda()
    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]
    index = torch.argsort(depth)
    index = index.type(torch.int32).reshape(-1, 1).cpu().numpy()
    return index


# Decide which sort to use
_sort_gaussian = None
try:
    import torch
    if not torch.cuda.is_available():
        raise ImportError
    print("Detect torch cuda installed, will use torch as sorting backend")
    _sort_gaussian = _sort_gaussian_torch
except ImportError:
    try:
        import cupy as cp
        print("Detect cupy installed, will use cupy as sorting backend")
        _sort_gaussian = _sort_gaussian_cupy
    except ImportError:
        _sort_gaussian = _sort_gaussian_cpu


class CropBox:
    """裁剪框类，用于定义和操作3D裁剪框"""
    def __init__(self, min_point=None, max_point=None):
        if min_point is None:
            min_point = np.array([-1.0, -1.0, -1.0])
        if max_point is None:
            max_point = np.array([1.0, 1.0, 1.0])
        
        self.min_point = np.array(min_point, dtype=np.float32)
        self.max_point = np.array(max_point, dtype=np.float32)
        self.start_point = np.copy(self.min_point)  # 初始起点
        self.end_point = np.copy(self.max_point)  # 初始终点
        self.enabled = False
        self.visible = False  # 是否显示裁剪框
        self.show_control_points = False  # 是否显示控制点

    def set_bounds(self, start_point, end_point):
        """设置裁剪框的边界"""
        self.min_point = np.minimum(start_point, end_point)
        self.max_point = np.maximum(start_point, end_point)
        self.start_point = np.copy(start_point)
        self.end_point = np.copy(end_point)

    def enable(self):
        """启用裁剪"""
        self.enabled = True
        
    def disable(self):
        """禁用裁剪"""
        self.enabled = False

    def toggle_enabled(self):
        """切换裁剪框可见性"""
        self.enabled = not self.enabled

    def toggle_visibility(self):
        """切换裁剪框可见性"""
        self.visible = not self.visible

    def toggle_control_points(self):
        """切换控制点可见性"""
        self.show_control_points = not self.show_control_points
        
    def get_wireframe_vertices(self):
        """获取裁剪框的线框顶点"""
        min_p = self.min_point
        max_p = self.max_point
        
        # 定义立方体的8个顶点
        vertices = np.array([
            [min_p[0], min_p[1], min_p[2]],  # 0
            [max_p[0], min_p[1], min_p[2]],  # 1
            [max_p[0], max_p[1], min_p[2]],  # 2
            [min_p[0], max_p[1], min_p[2]],  # 3
            [min_p[0], min_p[1], max_p[2]],  # 4
            [max_p[0], min_p[1], max_p[2]],  # 5
            [max_p[0], max_p[1], max_p[2]],  # 6
            [min_p[0], max_p[1], max_p[2]],  # 7
        ], dtype=np.float32)
        
        return vertices
        
    def get_wireframe_indices(self):
        """获取裁剪框的线框索引"""
        # 定义立方体的12条边
        indices = np.array([
            # 底面
            0, 1, 1, 2, 2, 3, 3, 0,
            # 顶面
            4, 5, 5, 6, 6, 7, 7, 4,
            # 垂直边
            0, 4, 1, 5, 2, 6, 3, 7
        ], dtype=np.uint32)
        
        return indices
    
    def get_control_points(self):
        """获取控制点位置（min_point和max_point）"""
        return np.array([self.start_point, self.end_point], dtype=np.float32)

class GaussianRenderBase:
    def __init__(self):
        self.gaussians = None
        self._reduce_updates = True

    @property
    def reduce_updates(self):
        return self._reduce_updates

    @reduce_updates.setter
    def reduce_updates(self, val):
        self._reduce_updates = val
        self.update_vsync()

    def update_vsync(self):
        print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        raise NotImplementedError()
    
    def sort_and_update(self):
        raise NotImplementedError()

    def set_scale_modifier(self, modifier: float):
        raise NotImplementedError()
    
    def set_render_mod(self, mod: int):
        raise NotImplementedError()
    
    def update_camera_pose(self, camera: util.Camera):
        raise NotImplementedError()

    def update_camera_intrin(self, camera: util.Camera):
        raise NotImplementedError()
    
    def draw(self):
        raise NotImplementedError()
    
    def set_render_reso(self, w, h):
        raise NotImplementedError()

    def set_crop_box(self, start_point, end_point):
        """设置裁剪框"""
        self.crop_box.set_bounds(start_point, end_point)
        #self.crop_box.visible = True  # 默认启用裁剪框可见性
        
    def enable_crop_box(self):
        """启用裁剪框"""
        self.crop_box.enable()

    def disable_crop_box(self):
        """禁用裁剪框"""
        self.crop_box.disable()

    def toggle_crop_box_enabled(self):
        """切换裁剪框可见性"""
        self.crop_box.toggle_enabled()

    def toggle_crop_box_visibility(self):
        """切换裁剪框可见性"""
        self.crop_box.toggle_visibility()

    def toggle_control_points(self):
        """切换控制点可见性"""
        self.crop_box.toggle_control_points()


class OpenGLRenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()
        gl.glViewport(0, 0, w, h)
        self.program = util.load_shaders('shaders/gau_vert.glsl', 'shaders/gau_frag.glsl')

        # 加载裁剪框的着色器程序
        try:
            self.wireframe_program = util.load_shaders('shaders/wireframe_vert.glsl', 'shaders/wireframe_frag.glsl')
        except:
            print("Warning: wireframe shaders not found, crop box will be disabled")
            self.wireframe_program = None

        self.crop_box = CropBox()  # 初始化裁剪框
        # Vertex data for a quad
        self.quad_v = np.array([
            -1,  1,
            1,  1,
            1, -1,
            -1, -1
        ], dtype=np.float32).reshape(4, 2)
        self.quad_f = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32).reshape(2, 3)
        
        # load quad geometry
        vao, buffer_id = util.set_attributes(self.program, ["position"], [self.quad_v])
        util.set_faces_tovao(vao, self.quad_f)
        self.vao = vao
        self.gau_bufferid = None
        self.index_bufferid = None

        # 设置裁剪框的VAO和缓冲区
        if self.wireframe_program:
            self.setup_wireframe_rendering()
            self.setup_control_point_rendering()
            #self.test_basic_rendering()
        else:
            self.wireframe_vao = None
            self.wireframe_vbo = None
            self.wireframe_ebo = None
            self.control_point_vaos = []
            self.control_point_vbos = []
            self.control_point_ebos = []

        # opengl settings
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.update_vsync()

    def setup_wireframe_rendering(self):
        """设置裁剪框线框渲染"""
        # 创建裁剪框的VAO和VBO
        self.wireframe_vao = gl.glGenVertexArrays(1)
        self.wireframe_vbo = gl.glGenBuffers(1)
        self.wireframe_ebo = gl.glGenBuffers(1)
        
        gl.glBindVertexArray(self.wireframe_vao)
        
        # 设置顶点缓冲区
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.wireframe_vbo)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)
        
        # 设置索引缓冲区
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.wireframe_ebo)
        
        gl.glBindVertexArray(0)

    def setup_control_point_rendering(self):
        """设置控制点渲染"""
        self.control_point_vaos = []
        self.control_point_vbos = []
        self.control_point_ebos = []
        
        # 为两个控制点（min_point和max_point）创建VAO
        for i in range(2):
            vao = gl.glGenVertexArrays(1)
            vbo = gl.glGenBuffers(1)
            ebo = gl.glGenBuffers(1)
            
            gl.glBindVertexArray(vao)
            
            # 设置顶点缓冲区
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(0)
            
            # 设置索引缓冲区
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
            
            gl.glBindVertexArray(0)
            
            self.control_point_vaos.append(vao)
            self.control_point_vbos.append(vbo)
            self.control_point_ebos.append(ebo)

    def update_wireframe_geometry(self):
        """更新裁剪框的几何体"""
        vertices = self.crop_box.get_wireframe_vertices()
        indices = self.crop_box.get_wireframe_indices()
        
        gl.glBindVertexArray(self.wireframe_vao)
        
        # 更新顶点数据
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.wireframe_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_DYNAMIC_DRAW)
        
        # 更新索引数据
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.wireframe_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
        
        gl.glBindVertexArray(0)

    def update_vsync(self):
        if wglSwapIntervalEXT is not None:
            wglSwapIntervalEXT(1 if self.reduce_updates else 0)
        else:
            print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.gaussians = gaus
        # load gaussian geometry
        gaussian_data = gaus.flat()
        self.gau_bufferid = util.set_storage_buffer_data(self.program, "gaussian_data", gaussian_data, 
                                                         bind_idx=0,
                                                         buffer_id=self.gau_bufferid)
        util.set_uniform_1int(self.program, gaus.sh_dim, "sh_dim")

        # 更新裁剪框相关的uniform
        self.update_crop_box_uniforms()

    def update_crop_box_uniforms(self):
        """更新裁剪框相关的uniform变量"""
        util.set_uniform_1int(self.program, int(self.crop_box.enabled), "crop_enabled")
        util.set_uniform_v3(self.program, self.crop_box.min_point, "crop_min")
        util.set_uniform_v3(self.program, self.crop_box.max_point, "crop_max")

    def sort_and_update(self, camera: util.Camera):
        index = _sort_gaussian(self.gaussians, camera.get_view_matrix())
        self.index_bufferid = util.set_storage_buffer_data(self.program, "gi", index, 
                                                           bind_idx=1,
                                                           buffer_id=self.index_bufferid)
        return
   
    def set_scale_modifier(self, modifier):
        util.set_uniform_1f(self.program, modifier, "scale_modifier")

    def set_render_mod(self, mod: int):
        util.set_uniform_1int(self.program, mod, "render_mod")

    def set_render_reso(self, w, h):
        gl.glViewport(0, 0, w, h)

    def update_camera_pose(self, camera: util.Camera):
        view_mat = camera.get_view_matrix()
        util.set_uniform_mat4(self.program, view_mat, "view_matrix")
        util.set_uniform_v3(self.program, camera.position, "cam_pos")

        # 同时更新裁剪框着色器的view matrix
        util.set_uniform_mat4(self.wireframe_program, view_mat, "view_matrix")

    def update_camera_intrin(self, camera: util.Camera):
        proj_mat = camera.get_project_matrix()
        util.set_uniform_mat4(self.program, proj_mat, "projection_matrix")
        util.set_uniform_v3(self.program, camera.get_htanfovxy_focal(), "hfovxy_focal")

        # 同时更新裁剪框着色器的projection matrix
        util.set_uniform_mat4(self.wireframe_program, proj_mat, "projection_matrix")

    def draw(self):
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.vao)
        num_gau = len(self.gaussians)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, len(self.quad_f.reshape(-1)), gl.GL_UNSIGNED_INT, None, num_gau)

        # 绘制裁剪框（如果启用且可见）
        if self.crop_box.visible:
            self.draw_crop_box()
            # 绘制控制点
            if self.crop_box.show_control_points:
                self.draw_control_points_simple()

        # 3. 确保恢复到主渲染状态
        # gl.glUseProgram(self.program)
        # gl.glBindVertexArray(self.vao)
        # gl.glEnable(gl.GL_BLEND)
        # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    
    def draw_crop_box(self):
        """绘制裁剪框"""
        if not self.wireframe_program or not self.wireframe_vao:
            return
            
        try:
            # 更新裁剪框几何体
            self.update_wireframe_geometry()
            
            # 保存当前状态
            old_program = gl.glGetInteger(gl.GL_CURRENT_PROGRAM)
            old_line_width = gl.glGetFloat(gl.GL_LINE_WIDTH)
            depth_test_enabled = gl.glIsEnabled(gl.GL_DEPTH_TEST)
            
            # 设置线框渲染状态
            gl.glUseProgram(self.wireframe_program)
            gl.glBindVertexArray(self.wireframe_vao)
            
            # 设置线宽
            gl.glLineWidth(2.0)
            
            # 禁用深度测试，确保裁剪框总是可见
            gl.glDisable(gl.GL_DEPTH_TEST)
            
            # 设置裁剪框颜色（橙色）
            try:
                util.set_uniform_v3(self.wireframe_program, np.array([1.0, 0.5, 0.0]), "color")
            except:
                pass  # 如果着色器没有这个uniform，忽略
            
            # 绘制线框
            indices = self.crop_box.get_wireframe_indices()
            gl.glDrawElements(gl.GL_LINES, len(indices), gl.GL_UNSIGNED_INT, None)
            
            # 恢复状态
            gl.glUseProgram(old_program)
            gl.glLineWidth(old_line_width)
            if depth_test_enabled:
                gl.glEnable(gl.GL_DEPTH_TEST)
            
            gl.glBindVertexArray(0)
        except Exception as e:
            print(f"Error drawing crop box: {e}")

    def draw_control_points_simple(self):
        """简化的控制点绘制 - 使用线段组成的十字，完善状态管理"""
        if not self.wireframe_program:
            return
            
        try:
            # 保存所有相关的OpenGL状态
            old_program = gl.glGetInteger(gl.GL_CURRENT_PROGRAM)
            old_vao = gl.glGetInteger(gl.GL_VERTEX_ARRAY_BINDING)
            old_array_buffer = gl.glGetInteger(gl.GL_ARRAY_BUFFER_BINDING)
            old_element_buffer = gl.glGetInteger(gl.GL_ELEMENT_ARRAY_BUFFER_BINDING)
            old_line_width = gl.glGetFloat(gl.GL_LINE_WIDTH)
            
            # 保存深度测试状态
            depth_test_enabled = gl.glIsEnabled(gl.GL_DEPTH_TEST)
            
            # 保存混合状态
            blend_enabled = gl.glIsEnabled(gl.GL_BLEND)
            if blend_enabled:
                blend_src = gl.glGetInteger(gl.GL_BLEND_SRC_ALPHA)
                blend_dst = gl.glGetInteger(gl.GL_BLEND_DST_ALPHA)
            
            # 设置控制点渲染状态
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glUseProgram(self.wireframe_program)
            
            control_points = self.crop_box.get_control_points()
            colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            
            for i, point in enumerate(control_points):
                # 绘制三维十字
                self.draw_cross_at_point(point, 0.2, colors[i])
            
            # 完全恢复OpenGL状态
            gl.glUseProgram(old_program)
            gl.glBindVertexArray(old_vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, old_array_buffer)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, old_element_buffer)
            gl.glLineWidth(old_line_width)
            
            # 恢复深度测试
            if depth_test_enabled:
                gl.glEnable(gl.GL_DEPTH_TEST)
            
            # 恢复混合状态
            if blend_enabled:
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(blend_src, blend_dst)
            
            # 确保没有遗留的顶点属性启用
            gl.glDisableVertexAttribArray(0)
            
        except Exception as e:
            print(f"Error in simple control points: {e}")
            # 紧急恢复状态
            try:
                gl.glUseProgram(old_program)
                gl.glBindVertexArray(old_vao)
                if depth_test_enabled:
                    gl.glEnable(gl.GL_DEPTH_TEST)
            except:
                pass

    def draw_cross_at_point(self, center, size, color):
        """在指定点绘制三维十字，改进状态管理"""
        half_size = size / 2.0
        x, y, z = center[0], center[1], center[2]
        
        # 三条线段的顶点
        vertices = np.array([
            # X轴线段
            [x - half_size, y, z], [x + half_size, y, z],
            # Y轴线段  
            [x, y - half_size, z], [x, y + half_size, z],
            # Z轴线段
            [x, y, z - half_size], [x, y, z + half_size],
        ], dtype=np.float32)
        
        # 线段索引
        indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
        
        # 创建临时VAO和缓冲区
        temp_vao = gl.glGenVertexArrays(1)
        temp_vbo = gl.glGenBuffers(1)
        temp_ebo = gl.glGenBuffers(1)
        
        try:
            gl.glBindVertexArray(temp_vao)
            
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, temp_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_DYNAMIC_DRAW)
            
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, temp_ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_DYNAMIC_DRAW)
            
            # 设置颜色
            try:
                util.set_uniform_v3(self.wireframe_program, np.array(color), "color")
            except:
                color_location = gl.glGetUniformLocation(self.wireframe_program, "color")
                if color_location != -1:
                    gl.glUniform3f(color_location, color[0], color[1], color[2])
            
            # 设置线宽并绘制
            gl.glLineWidth(5.0)
            gl.glDrawElements(gl.GL_LINES, len(indices), gl.GL_UNSIGNED_INT, None)
            
        finally:
            # 立即清理临时资源，避免状态污染
            gl.glBindVertexArray(0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
            gl.glDisableVertexAttribArray(0)
            
            gl.glDeleteVertexArrays(1, [temp_vao])
            gl.glDeleteBuffers(1, [temp_vbo])
            gl.glDeleteBuffers(1, [temp_ebo])

    def set_crop_box(self, start_point, end_point):
        """设置裁剪框并更新uniform"""
        super().set_crop_box(start_point, end_point)
        self.update_crop_box_uniforms()
