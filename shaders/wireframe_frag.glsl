// wireframe_frag.glsl - 裁剪框片段着色器
#version 330 core

uniform vec3 color;
out vec4 FragColor;

void main()
{
    FragColor = vec4(color, 1.0);
}