import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import imageio
import util_gau
import tkinter as tk
from tkinter import filedialog
import os
import sys
import argparse
from renderer_ogl import OpenGLRenderer, GaussianRenderBase


# Add the directory containing main.py to the Python path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


g_camera = util.Camera(720, 1280)
#g_camera.load_pose_from_file()  # 尝试加载上次位姿

BACKEND_OGL=0
BACKEND_CUDA=1
g_renderer_list = [
    None, # ogl
]
g_renderer_idx = BACKEND_OGL
g_renderer: GaussianRenderBase = g_renderer_list[g_renderer_idx]
g_scale_modifier = 1.
g_auto_sort = False
g_is_center_at_origin = True  # 是否将点云中心移到坐标原点
g_centering_offset = np.zeros(3, dtype=np.float32)  # 平移偏移量，默认为零向量
g_show_control_win = True
g_show_help_win = True
g_show_camera_win = False
g_render_mode_tables = ["Gaussian Ball", "Flat Ball", "Billboard", "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 7

g_bbox_state = util.BBoxState()
g_original_gaussians = None
g_cropped_gaussians = None
g_wireframe_renderer = None

def impl_glfw_init():
    window_name = "NeUVF editor"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    global window
    window = glfw.create_window(
        g_camera.w, g_camera.h, window_name, None, None
    )
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL);
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:
        g_camera.is_leftmouse_pressed = False
        g_camera.is_rightmouse_pressed = False
    g_camera.process_mouse(xpos, ypos)

def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)

def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)

def key_callback(window, key, scancode, action, mods):
    if action == glfw.PRESS or action == glfw.REPEAT:
        # 镜头旋转控制
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)

    # 处理移动模块按键
    if key in g_camera.key_states:
        # 更新按键状态
        if action == glfw.PRESS:
            g_camera.key_states[key] = True
        elif action == glfw.RELEASE:
            g_camera.key_states[key] = False

    # 添加保存/加载快捷键 (F1保存，F5加载)
    if action == glfw.PRESS:
        if key == glfw.KEY_F1:
            g_camera.save_pose_to_file()
        elif key == glfw.KEY_F5:
            g_camera.load_pose_from_file()

def update_camera_pose_lazy():
    if g_camera.is_pose_dirty:
        g_renderer.update_camera_pose(g_camera)
        g_camera.is_pose_dirty = False

def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:
        g_renderer.update_camera_intrin(g_camera)
        g_camera.is_intrin_dirty = False

def update_activated_renderer_state(gaus: util_gau.GaussianData):
    g_renderer.update_gaussian_data(gaus)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)

def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)


def draw_bbox_wireframe(bbox_state, camera):
    if not bbox_state.is_selecting and not bbox_state.is_finalized:
        return

    # 计算当前包围盒坐标
    if bbox_state.is_selecting:
        # 动态跟随相机
        current_min = np.minimum(bbox_state.start_point, camera.position)
        current_max = np.maximum(bbox_state.start_point, camera.position)
    else:
        # 已固定的包围盒
        current_min = np.minimum(bbox_state.start_point, bbox_state.end_point)
        current_max = np.maximum(bbox_state.start_point, bbox_state.end_point)

    # 获取8个角点
    corners = []
    for x in [current_min[0], current_max[0]]:
        for y in [current_min[1], current_max[1]]:
            for z in [current_min[2], current_max[2]]:
                corners.append([x, y, z])


    # 渲染线框（自动状态管理）
    view_mat = g_camera.get_view_matrix()
    proj_mat = g_camera.get_project_matrix()
    # 组合MVP矩阵（假设无模型变换）
    mvp_matrix = proj_mat @ view_mat @ np.eye(4)

    g_wireframe_renderer.update_geometry(corners)
    g_wireframe_renderer.render(mvp_matrix, bbox_state.is_selecting)

def main():
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort, g_is_center_at_origin, g_centering_offset,\
        g_show_control_win, g_show_help_win, g_show_camera_win, \
        g_render_mode, g_render_mode_tables, \
        g_bbox_state, g_original_gaussians, g_cropped_gaussians, \
        g_wireframe_renderer
        
    imgui.create_context()
    if args.hidpi:
        imgui.get_io().font_global_scale = 1.5
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()  # used for file dialog
    root.withdraw()
    
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    
    glfw.set_window_size_callback(window, window_resize_callback)

    # init renderer
    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h)
    try:
        from renderer_cuda import CUDARenderer
        g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]
    except ImportError:
        g_renderer_idx = BACKEND_OGL
    else:
        g_renderer_idx = BACKEND_CUDA

    g_renderer = g_renderer_list[g_renderer_idx]

    g_wireframe_renderer = util.WireframeRenderer()

    # gaussian data
    gaussians = util_gau.naive_gaussian()
    update_activated_renderer_state(gaussians)
    
    # settings
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        g_camera.update_camera_movement()  # 更新相机移动位置


        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose_lazy()
        update_camera_intrin_lazy()
        
        g_renderer.draw()
        #draw_bbox_wireframe(g_bbox_state, g_camera)

        # imgui ui
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, g_show_control_win = imgui.menu_item(
                    "Show Control", None, g_show_control_win
                )
                clicked, g_show_help_win = imgui.menu_item(
                    "Show Help", None, g_show_help_win
                )
                clicked, g_show_camera_win = imgui.menu_item(
                    "Show Camera Control", None, g_show_camera_win
                )
                imgui.end_menu()
            imgui.end_main_menu_bar()
        
        if g_show_control_win:
            if imgui.begin("Control", True):
                # rendering backend
                changed, g_renderer_idx = imgui.combo("backend", g_renderer_idx, ["ogl", "cuda"][:len(g_renderer_list)])
                if changed:
                    g_renderer = g_renderer_list[g_renderer_idx]
                    update_activated_renderer_state(gaussians)

                imgui.text(f"fps = {imgui.get_io().framerate:.1f}")

                changed, g_renderer.reduce_updates = imgui.checkbox(
                        "reduce updates", g_renderer.reduce_updates,
                    )

                changed, g_is_center_at_origin = imgui.checkbox(
                        "center at origin", g_is_center_at_origin,
                    )
                imgui.text(f"# of Gaus = {len(gaussians)}")
                if imgui.button(label = 'plot_distribution'):
                    
                    file_path = filedialog.askopenfilename(title="open ply",
                        initialdir="C:\\Users\\MSI_NB\\Downloads\\viewers",
                        filetypes=[('ply file', '.ply')]
                        )
                    if file_path:
                        try:
                            test_gaussians, offset = util_gau.load_ply(file_path)
                            util_gau.plot_point_distribution(test_gaussians, "test gaussians")
                        except RuntimeError as e:
                            pass
                if imgui.button(label='open ply'):
                    file_path = filedialog.askopenfilename(title="open ply",
                        initialdir="C:\\Users\\MSI_NB\\Downloads\\viewers",
                        filetypes=[('ply file', '.ply')]
                        )
                    if file_path:
                        try:
                            gaussians, g_centering_offset = util_gau.load_ply(file_path, is_center_at_origin=g_is_center_at_origin)
                            g_renderer.update_gaussian_data(gaussians)
                            g_renderer.sort_and_update(g_camera)
                            g_original_gaussians = gaussians
                            g_cropped_gaussians = None
                        except RuntimeError as e:
                            pass
                
                imgui.same_line()
                if imgui.button(label = "save ply"):
                    # 弹出保存文件对话框
                    save_path = filedialog.asksaveasfilename(
                        title="Save PLY File",                   # 对话框标题
                        defaultextension=".ply",             # 默认扩展名
                        filetypes=[("PLY Files", "*.ply"), ("All Files", "*.*")]  # 文件类型过滤
                    )
                    if save_path:
                        util_gau.save_ply(g_renderer.gaussians, save_path)
                        print("PLY File saved to:", save_path)
                    else:
                        print("Save cancelled.")

                # camera fov
                changed, g_camera.fovy = imgui.slider_float(
                    "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f"
                )
                g_camera.is_intrin_dirty = changed
                update_camera_intrin_lazy()
                
                # scale modifier
                changed, g_scale_modifier = imgui.slider_float(
                    "", g_scale_modifier, 0.1, 10, "scale modifier = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset"):
                    g_scale_modifier = 1.
                    changed = True
                    
                if changed:
                    g_renderer.set_scale_modifier(g_scale_modifier)
                
                # render mode
                changed, g_render_mode = imgui.combo("shading", g_render_mode, g_render_mode_tables)
                if changed:
                    g_renderer.set_render_mod(g_render_mode - 4)
                
                # sort button
                if imgui.button(label='sort Gaussians'):
                    g_renderer.sort_and_update(g_camera)
                imgui.same_line()
                changed, g_auto_sort = imgui.checkbox(
                        "auto sort", g_auto_sort,
                    )
                if g_auto_sort:
                    g_renderer.sort_and_update(g_camera)
                
                if imgui.button(label='save image'):
                    width, height = glfw.get_framebuffer_size(window)
                    nrChannels = 3;
                    stride = nrChannels * width;
                    stride += (4 - stride % 4) if stride % 4 else 0
                    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                    gl.glReadBuffer(gl.GL_FRONT)
                    bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                    img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
                    imageio.imwrite("save.png", img[::-1])
                    # save intermediate information
                    # np.savez(
                    #     "save.npz",
                    #     gau_xyz=gaussians.xyz,
                    #     gau_s=gaussians.scale,
                    #     gau_rot=gaussians.rot,
                    #     gau_c=gaussians.sh,
                    #     gau_a=gaussians.opacity,
                    #     viewmat=g_camera.get_view_matrix(),
                    #     projmat=g_camera.get_project_matrix(),
                    #     hfovxyfocal=g_camera.get_htanfovxy_focal()
                    # )


                if imgui.button(label = "save camera pose"):
                    g_camera.save_pose_to_file()
                imgui.same_line()
                if imgui.button(label = "load camera pose"):
                    g_camera.load_pose_from_file()
                if imgui.button(label = "reset camera pose"):
                    g_camera.reset_pose()

                # start select BBOX按钮逻辑
                if imgui.button(label = "start select BBOX"):
                    if not g_bbox_state.is_selecting:
                        g_bbox_state.start_point = g_camera.position.copy()
                        g_bbox_state.is_selecting = True
                        g_bbox_state.is_finalized = False
                    else:
                        g_bbox_state.reset()  # 取消选择

                # select BBox over按钮逻辑（仅在已选择起点时可用）
                imgui.same_line()
                if imgui.button(label = "select BBox over") and g_bbox_state.is_selecting:
                    g_bbox_state.end_point = g_camera.position.copy()
                    g_bbox_state.is_finalized = True
                    g_bbox_state.is_selecting = False

                # crop BBox按钮逻辑（仅在完成选择时可用）
                if imgui.button(label = "crop BBox") and g_bbox_state.is_finalized:
                    # 执行裁剪
                    bbox_min = np.minimum(g_bbox_state.start_point, g_bbox_state.end_point)
                    bbox_max = np.maximum(g_bbox_state.start_point, g_bbox_state.end_point)
                    
                    mask = util_gau.GaussianData.static_get_bbox_mask(g_original_gaussians.xyz, bbox_min, bbox_max)
                    g_cropped_gaussians = g_original_gaussians.crop_with_mask(mask)

                    g_renderer.update_gaussian_data(g_cropped_gaussians)
                    g_renderer.sort_and_update(g_camera)

                imgui.same_line()
                if imgui.button(label = "undo crop"):
                    #撤销裁剪
                    g_renderer.update_gaussian_data(g_original_gaussians)
                    g_renderer.sort_and_update(g_camera)
                    g_cropped_gaussians = None

                imgui.end()

        if g_show_camera_win:
            if imgui.button(label='rot 180'):
                g_camera.flip_ground()

            changed, g_camera.target_dist = imgui.slider_float(
                    "t", g_camera.target_dist, 1., 8., "target dist = %.3f"
                )
            if changed:
                g_camera.update_target_distance()

            changed, g_camera.rot_sensitivity = imgui.slider_float(
                    "r", g_camera.rot_sensitivity, 0.002, 0.1, "rotate speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset r"):
                g_camera.rot_sensitivity = 0.02

            changed, g_camera.trans_sensitivity = imgui.slider_float(
                    "m", g_camera.trans_sensitivity, 0.001, 0.03, "move speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset m"):
                g_camera.trans_sensitivity = 0.01

            changed, g_camera.zoom_sensitivity = imgui.slider_float(
                    "z", g_camera.zoom_sensitivity, 0.001, 0.05, "zoom speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset z"):
                g_camera.zoom_sensitivity = 0.01

            changed, g_camera.roll_sensitivity = imgui.slider_float(
                    "ro", g_camera.roll_sensitivity, 0.003, 0.1, "roll speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset ro"):
                g_camera.roll_sensitivity = 0.03

        if g_show_help_win:
            imgui.begin("Help", True)
            imgui.text("Open Gaussian Splatting PLY file \n  by click 'open ply' button")
            imgui.text("Use left click & move to rotate camera")
            imgui.text("Use right click & move to translate camera")
            imgui.text("Press Q/E to roll camera")
            imgui.text("Use scroll to zoom in/out")
            imgui.text("Use control panel to change setting")
            imgui.end()
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()

    #g_camera.save_pose_to_file()  # save camera pose

    glfw.terminate()


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description="NeUVF editor with optional HiDPI support.")
    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
    args = parser.parse_args()

    main()
