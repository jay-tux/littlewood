#version 330 core
uniform sampler2D tex;
out vec4 FragColor;
in vec2 pass;

void main()
{
  FragColor = texture(tex, pass).bgra;
  //FragColor = vec4(pass.xy, 0.0f, 1.0f);
}
