from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
from OpenGL import GL as gl
import numpy as np
import glm
import ctypes

import glfw
import json
import os




class BBoxState:
    def __init__(self):
        self.start_point = None     # 起点坐标
        self.end_point = None       # 终点坐标
        self.is_selecting = False   # 是否正在选择起点
        self.is_finalized = False   # 是否已完成选择
    def reset(self):
        """重置所有状态"""
        self.__init__()



class WireframeRenderer:
    def __init__(self):
        self.program = self._compile_shaders()
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        
    def _compile_shaders(self):
        # 极简着色器（与主渲染着色器互不干扰）
        vertex_shader = """
        #version 330 core
        layout(location=0) in vec3 position;
        uniform mat4 mvp;
        void main() { gl_Position = mvp * vec4(position,1); }
        """
        fragment_shader = """
        #version 330 core
        uniform vec3 color;
        out vec4 FragColor;
        void main() { FragColor = vec4(color,1); }
        """
        return compile_shaders(vertex_shader, fragment_shader)

    def update_geometry(self, corners):
        # 更新线框顶点数据
        indices = [
            0,1, 1,3, 3,2, 2,0,  # 底面
            4,5, 5,7, 7,6, 6,4,  # 顶面
            0,4, 1,5, 2,6, 3,7   # 侧面
        ]
        vertices = np.array([corners[i] for i in indices], dtype=np.float32)
        
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glBindVertexArray(0)

    def render(self, mvp_matrix, is_selected):
        # 保存当前OpenGL状态
        prev_program = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
        prev_vao = gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)
        
        # 设置线框专属状态
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.program, "mvp"), 1, gl.GL_FALSE, mvp_matrix)
        color = [0, 1, 0] if is_selected else [1, 0.5, 0]
        gl.glUniform3f(gl.glGetUniformLocation(self.program, "color"), color[0], color[1], color[2])
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glLineWidth(2.0)
        
        # 绘制线框
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_LINES, 0, 24)
        gl.glBindVertexArray(0)
        
        # 精确恢复原始状态
        gl.glUseProgram(prev_program)
        gl.glBindVertexArray(prev_vao)


class Camera:
    def __init__(self, h, w):
        # 投影参数
        self.znear = 0.01       # 近裁剪平面距离(单位：世界坐标)
        self.zfar = 100         # 远裁剪平面距离(单位：世界坐标)
        self.h = h              # 视口高度(像素)
        self.w = w              # 视口宽度(像素)
        self.fovy = np.pi / 2   # 垂直视野角度(弧度)，π/2=90度

        # 视图参数
        self.position = np.array([3.0, 0.0, 0.0]).astype(np.float32)  # 相机位置(x,y,z)
        self.target = np.array([0.0, 0.0, 0.0]).astype(np.float32)     # 相机注视的目标点
        self.up = np.array([0.0, 0.0, 1.0]).astype(np.float32)        # 相机上方向向量(归一化)


        # 欧拉角
        self.yaw = 0             # 偏航角(绕Y轴旋转，弧度)
        self.pitch = 0           # 俯仰角(绕X轴旋转，弧度)
    

        self.orientation = glm.quat(1.0, 0.0, 0.0, 0.0)  # 单位四元数
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # 状态标志
        self.is_pose_dirty = True     # 表示相机位置/旋转是否变化需要更新视图矩阵
        self.is_intrin_dirty = True   # 表示内参是否变化需要更新投影矩阵
        
        # 鼠标控制
        self.last_x = 640         # 上一次鼠标X位置(像素)
        self.last_y = 360         # 上一次鼠标Y位置(像素)
        self.first_mouse = True   # 是否是第一次收到鼠标移动事件
        
        # 鼠标按键状态
        self.is_leftmouse_pressed = False   # 左键是否按下(用于旋转控制)
        self.is_rightmouse_pressed = False  # 右键是否按下(用于平移控制)
        
        # 移动控制参数
        self.move_sensitivity = 0.8  # 基础移动速度(单位：世界坐标/按键事件)
        self.boost_multiplier = 2.0  # 加速移动时的速度倍率(按住Shift时生效)

        # 各操作灵敏度
        self.rot_sensitivity = 0.01    # 旋转(鼠标左键拖动)灵敏度
        self.trans_sensitivity = 0.03  # 平移(鼠标右键拖动)灵敏度
        self.zoom_sensitivity = 0.08   # 缩放(鼠标滚轮)灵敏度
        self.roll_sensitivity = 0.03   # 滚转(Q/E按键)灵敏度
        
        # 距离参数
        self.target_dist = 3.          # 相机到目标的初始距离(单位：世界坐标)

        # 新增按键状态跟踪
        self.key_states = {
            glfw.KEY_W: False,
            glfw.KEY_A: False,
            glfw.KEY_S: False,
            glfw.KEY_D: False,
            glfw.KEY_SPACE: False,
            glfw.KEY_LEFT_CONTROL: False,
            glfw.KEY_LEFT_SHIFT: False
        }
    

    def _global_rot_mat(self):
        x = np.array([1, 0, 0])
        z = np.cross(x, self.up)
        z = z / np.linalg.norm(z)
        x = np.cross(self.up, z)
        return np.stack([x, self.up, z], axis=-1)



    def get_view_matrix(self):
        return np.array(glm.lookAt(self.position, self.target, self.up))

    def get_project_matrix(self):
        # htanx, htany, focal = self.get_htanfovxy_focal()
        # f_n = self.zfar - self.znear
        # proj_mat = np.array([
        #     1 / htanx, 0, 0, 0,
        #     0, 1 / htany, 0, 0,
        #     0, 0, self.zfar / f_n, - 2 * self.zfar * self.znear / f_n,
        #     0, 0, 1, 0
        # ])
        project_mat = glm.perspective(
            self.fovy,
            self.w / self.h,
            self.znear,
            self.zfar
        )
        return np.array(project_mat).astype(np.float32)

    def get_htanfovxy_focal(self):
        htany = np.tan(self.fovy / 2)
        htanx = htany / self.h * self.w
        focal = self.h / (2 * htany)
        return [htanx, htany, focal]

    def get_focal(self):
        return self.h / (2 * np.tan(self.fovy / 2))

    def process_mouse(self, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos

        if self.is_leftmouse_pressed:
            self.yaw += xoffset * self.rot_sensitivity
            self.yaw = self.yaw % (2 * np.pi)

            self.pitch += yoffset * self.rot_sensitivity
            self.pitch = np.clip(self.pitch, -np.pi/2 + 0.01, np.pi/2 - 0.01)  # 避免完全垂直

            front = np.array([np.cos(self.yaw) * np.cos(self.pitch), 
                            np.sin(self.pitch), np.sin(self.yaw) * 
                            np.cos(self.pitch)])


            """
            front = self._global_rot_mat() @ front.reshape(3, 1)
            front = front[:, 0]
            self.position[:] = - front * np.linalg.norm(self.position - self.target) + self.target

            self.is_pose_dirty = True
            """

            front = self._global_rot_mat() @ front.reshape(3, 1)
            front = front[:, 0]
            # 更新target位置（保持固定距离）
            self.target[:] = front * np.linalg.norm(self.target - self.position) + self.position


            self.is_pose_dirty = True
        
        if self.is_rightmouse_pressed:
            front = self.target - self.position
            front = front / np.linalg.norm(front)
            right = np.cross(self.up, front)
            self.position += right * xoffset * self.trans_sensitivity
            self.target += right * xoffset * self.trans_sensitivity
            cam_up = np.cross(right, front)
            self.position += cam_up * yoffset * self.trans_sensitivity
            self.target += cam_up * yoffset * self.trans_sensitivity
            
            self.is_pose_dirty = True
        
    def process_wheel(self, dx, dy):
        front = self.target - self.position
        front = front / np.linalg.norm(front)
        self.position += front * dy * self.zoom_sensitivity
        self.target += front * dy * self.zoom_sensitivity
        self.is_pose_dirty = True
        
    def process_roll_key(self, d):
        front = self.target - self.position
        right = np.cross(front, self.up)
        new_up = self.up + right * (d * self.roll_sensitivity / np.linalg.norm(right))
        self.up = new_up / np.linalg.norm(new_up)
        self.is_pose_dirty = True


    def reset_pose(self):
        front = self.target - self.position
        front = front / np.linalg.norm(front)
        right  = np.cross(front, self.up)
        right = right / np.linalg.norm(right)
        new_up = np.cross(right, front)
        self.up = new_up / np.linalg.norm(new_up)
        self.yaw = 0
        self.pitch = 0
        self.is_pose_dirty = True

    def get_pose_state(self):
        """获取位姿状态字典"""
        return {
            # 位置和朝向
            'position': self.position.tolist(),
            'target': self.target.tolist(),
            'up': self.up.tolist(),
            
            # 欧拉角
            'yaw': float(self.yaw),
            'pitch': float(self.pitch),
            
            # 距离参数
            'target_dist': float(self.target_dist)
        }
    
    def save_pose_to_file(self, filepath='camera_pose.json'):
        """保存位姿信息到JSON文件"""
        pose_state = self.get_pose_state()
        with open(filepath, 'w') as f:
            json.dump(pose_state, f, indent=4)
        print(f"Camera pose saved to {filepath}")
    
    def load_pose_from_file(self, filepath='camera_pose.json'):
        """从JSON文件加载位姿信息"""
        if not os.path.exists(filepath):
            print(f"Warning: Camera pose file {filepath} not found, using default pose")
            return False
        
        with open(filepath, 'r') as f:
            pose_state = json.load(f)
        
        # 更新位姿相关参数
        self.position = np.array(pose_state['position'], dtype=np.float32)
        self.target = np.array(pose_state['target'], dtype=np.float32)
        self.up = np.array(pose_state['up'], dtype=np.float32)
        
        self.yaw = pose_state['yaw']
        self.pitch = pose_state['pitch']
        self.target_dist = pose_state['target_dist']
        
        # 标记需要更新视图矩阵
        self.is_pose_dirty = True
        
        print(f"Camera pose loaded from {filepath}")
        return True


    def update_camera_movement(self, delta_time=None):
        """持续处理相机移动"""
        # 获取当前是否加速
        is_boosting = self.key_states.get(glfw.KEY_LEFT_SHIFT, False)
        
        # 计算移动方向
        dx, dy, dz = 0, 0, 0
        if self.key_states[glfw.KEY_W]: dz += 1.0    # 前
        if self.key_states[glfw.KEY_S]: dz -= 1.0    # 后
        if self.key_states[glfw.KEY_A]: dx -= 1.0    # 左
        if self.key_states[glfw.KEY_D]: dx += 1.0    # 右
        if self.key_states[glfw.KEY_SPACE]: dy += 1.0    # 上
        if self.key_states[glfw.KEY_LEFT_CONTROL]: dy -= 1.0    # 下
        
        # 如果有移动输入，处理移动
        if dx != 0 or dy != 0 or dz != 0:
            self.process_move_key(dx, dy, dz, is_boosting)


    def process_move_key(self, dx, dy, dz, is_boosting=False):
        """处理相机移动
        Args:
            dx: 左右移动量 (正=右, 负=左)
            dy: 上下移动量 (正=上, 负=下)
            dz: 前后移动量 (正=前, 负=后)
            is_boosting: 是否正在加速
        """
        # 计算相机坐标系下的方向向量
        front = self.target - self.position
        front = front / np.linalg.norm(front)
        right = np.cross(front, self.up)

        right_norm = np.linalg.norm(right)
        if right_norm > 1e-6:  # 添加小的阈值防止除以零
            right  = right / right_norm
        else:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # 默认右向量

        up = np.cross(right, front)
        
        # 计算移动向量
        move_vec = np.zeros(3, dtype=np.float32)
        if dx != 0:
            move_vec += right * dx
        if dy != 0:
            move_vec += up * dy  # 使用计算得到的up向量而不是self.up
        if dz != 0:
            move_vec += front * dz
        
        # 应用移动速度和加速
        speed = self.move_sensitivity
        if is_boosting:
            speed *= self.boost_multiplier
        
        # 更新相机和目标位置
        self.position += move_vec * speed
        self.target += move_vec * speed
        
        # 更新目标距离
        self.target_dist = np.linalg.norm(self.position - self.target)
        self.is_pose_dirty = True


    def flip_ground(self):
        self.up = -self.up
        self.is_pose_dirty = True

    def update_target_distance(self):
        _dir = self.target - self.position
        _dir = _dir / np.linalg.norm(_dir)
        self.target = self.position + _dir * self.target_dist
        
    def update_resolution(self, height, width):
        self.h = max(height, 1)
        self.w = max(width, 1)
        self.is_intrin_dirty = True


def load_shaders(vs, fs):
    vertex_shader = open(vs, 'r').read()        
    fragment_shader = open(fs, 'r').read()

    active_shader = shaders.compileProgram(
        shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )
    return active_shader


def compile_shaders(vertex_shader, fragment_shader):
    active_shader = shaders.compileProgram(
        shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )
    return active_shader


def set_attributes(program, keys, values, vao=None, buffer_ids=None):
    glUseProgram(program)
    if vao is None:
        vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    if buffer_ids is None:
        buffer_ids = [None] * len(keys)
    for i, (key, value, b) in enumerate(zip(keys, values, buffer_ids)):
        if b is None:
            b = glGenBuffers(1)
            buffer_ids[i] = b
        glBindBuffer(GL_ARRAY_BUFFER, b)
        glBufferData(GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
        length = value.shape[-1]
        pos = glGetAttribLocation(program, key)
        glVertexAttribPointer(pos, length, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(pos)
    
    glBindBuffer(GL_ARRAY_BUFFER,0)
    return vao, buffer_ids

def set_attribute(program, key, value, vao=None, buffer_id=None):
    glUseProgram(program)
    if vao is None:
        vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    if buffer_id is None:
        buffer_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id)
    glBufferData(GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
    length = value.shape[-1]
    pos = glGetAttribLocation(program, key)
    glVertexAttribPointer(pos, length, GL_FLOAT, False, 0, None)
    glEnableVertexAttribArray(pos)
    glBindBuffer(GL_ARRAY_BUFFER,0)
    return vao, buffer_id

def set_attribute_instanced(program, key, value, instance_stride=1, vao=None, buffer_id=None):
    glUseProgram(program)
    if vao is None:
        vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    if buffer_id is None:
        buffer_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id)
    glBufferData(GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
    length = value.shape[-1]
    pos = glGetAttribLocation(program, key)
    glVertexAttribPointer(pos, length, GL_FLOAT, False, 0, None)
    glEnableVertexAttribArray(pos)
    glVertexAttribDivisor(pos, instance_stride)
    glBindBuffer(GL_ARRAY_BUFFER,0)
    return vao, buffer_id

def set_storage_buffer_data(program, key, value: np.ndarray, bind_idx, vao=None, buffer_id=None):
    glUseProgram(program)
    # if vao is None:  # TODO: if this is really unnecessary?
    #     vao = glGenVertexArrays(1)
    if vao is not None:
        glBindVertexArray(vao)
    
    if buffer_id is None:
        buffer_id = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id)
    glBufferData(GL_SHADER_STORAGE_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
    # pos = glGetProgramResourceIndex(program, GL_SHADER_STORAGE_BLOCK, key)  # TODO: ???
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind_idx, buffer_id)
    # glShaderStorageBlockBinding(program, pos, pos)  # TODO: ???
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return buffer_id

def set_faces_tovao(vao, faces: np.ndarray):
    # faces
    glBindVertexArray(vao)
    element_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
    return element_buffer

def set_gl_bindings(vertices, faces):
    # vertices
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    # vertex_buffer = glGenVertexArrays(1)
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, None)
    glEnableVertexAttribArray(0)

    # faces
    element_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
    # glVertexAttribPointer(1, 3, GL_FLOAT, False, 36, ctypes.c_void_p(12))
    # glEnableVertexAttribArray(1)
    # glVertexAttribPointer(2, 3, GL_FLOAT, False, 36, ctypes.c_void_p(12))
    # glEnableVertexAttribArray(2)

def set_uniform_mat4(shader, content, name):
    glUseProgram(shader)
    if isinstance(content, glm.mat4):
        content = np.array(content).astype(np.float32)
    else:
        content = content.T
    glUniformMatrix4fv(
        glGetUniformLocation(shader, name), 
        1,
        GL_FALSE,
        content.astype(np.float32)
    )

def set_uniform_1f(shader, content, name):
    glUseProgram(shader)
    glUniform1f(
        glGetUniformLocation(shader, name), 
        content,
    )

def set_uniform_1int(shader, content, name):
    glUseProgram(shader)
    glUniform1i(
        glGetUniformLocation(shader, name), 
        content
    )

def set_uniform_v3f(shader, contents, name):
    glUseProgram(shader)
    glUniform3fv(
        glGetUniformLocation(shader, name),
        len(contents),
        contents
    )

def set_uniform_v3(shader, contents, name):
    glUseProgram(shader)
    glUniform3f(
        glGetUniformLocation(shader, name),
        contents[0], contents[1], contents[2]
    )

def set_uniform_v1f(shader, contents, name):
    glUseProgram(shader)
    glUniform1fv(
        glGetUniformLocation(shader, name),
        len(contents),
        contents
    )
    
def set_uniform_v2(shader, contents, name):
    glUseProgram(shader)
    glUniform2f(
        glGetUniformLocation(shader, name),
        contents[0], contents[1]
    )

def set_texture2d(img, texid=None):
    h, w, c = img.shape
    assert img.dtype == np.uint8
    if texid is None:
        texid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texid)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,   
        GL_RGB, GL_UNSIGNED_BYTE, img
    )
    glActiveTexture(GL_TEXTURE0)  # can be removed
    # glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    return texid

def update_texture2d(img, texid, offset):
    x1, y1 = offset
    h, w = img.shape[:2]
    glBindTexture(GL_TEXTURE_2D, texid)
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, x1, y1, w, h,
        GL_RGB, GL_UNSIGNED_BYTE, img
    )


