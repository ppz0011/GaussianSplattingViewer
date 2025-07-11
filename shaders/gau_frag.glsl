#version 430 core

in vec3 color;
in float alpha;
in vec3 conic;
in vec2 coordxy;  // local coordinate in quad, unit in pixel
in vec3 world_pos;  // 从顶点着色器传来的世界坐标

uniform int render_mod;  // > 0 render 0-ith SH dim, -1 depth, -2 bill board, -3 flat ball, -4 gaussian ball

// 裁剪框相关uniform
uniform int crop_enabled;
uniform vec3 crop_min;
uniform vec3 crop_max;

out vec4 FragColor;

void main() {
    // 裁剪框检查 - 在片段着色器中进行精确裁剪
    if (crop_enabled != 0) {
        // 检查当前片段的世界坐标是否在裁剪框内
        if (world_pos.x < crop_min.x || world_pos.x > crop_max.x ||
            world_pos.y < crop_min.y || world_pos.y > crop_max.y ||
            world_pos.z < crop_min.z || world_pos.z > crop_max.z) {
            discard;  // 丢弃裁剪框外的片段
        }
    }

    if (render_mod == -2)
    {
        FragColor = vec4(color, 1.f);
        return;
    }

    float power = -0.5f * (conic.x * coordxy.x * coordxy.x + conic.z * coordxy.y * coordxy.y) - conic.y * coordxy.x * coordxy.y;
    if (power > 0.f)
        discard;
    float opacity = min(0.99f, alpha * exp(power));
    if (opacity < 1.f / 255.f)
        discard;
    FragColor = vec4(color, opacity);

    // handling special shading effect
    if (render_mod == -3)
        FragColor.a = FragColor.a > 0.22 ? 1 : 0;
    else if (render_mod == -4)
    {
        FragColor.a = FragColor.a > 0.22 ? 1 : 0;
        FragColor.rgb = FragColor.rgb * exp(power);
    }
}