#version 140
uniform sampler1D gauss;

in vec3 velocity;
in vec3 param_pos;

out vec4 fragColor;
void
main()
{
  float r = length(param_pos);
  float g = texture( gauss, r ).a;
  fragColor = vec4( -100*g*normalize(param_pos), g );
}
