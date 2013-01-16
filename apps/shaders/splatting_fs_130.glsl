#version 130
uniform sampler1D gauss;
varying in vec3 velocity;
varying in vec3 param_pos;
void
main()
{
  float r = length(param_pos);
  float g = texture( gauss, r ).a;
  gl_FragColor = vec4( -100*g*normalize(param_pos), g );
}
