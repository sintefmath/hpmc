#version 130
uniform samplerBuffer positions;
varying out vec3 velocity;
varying out vec3 param_pos;
const float r = 0.15;
uniform int active;
uniform float slice_z;
void
main()
{
  vec3 v = texelFetch( positions, 2*gl_InstanceID+0 ).xyz;
  vec3 p = texelFetch( positions, 2*gl_InstanceID+1 ).xyz;
  bool kill = (p.z+r < slice_z ) ||
    (slice_z < p.z-r ) ||
    (active < gl_InstanceID);
  velocity = v;
  param_pos = vec3( gl_Vertex.xy, (1.0/r)*(slice_z-p.z) );
  gl_Position = vec4( 2.0*(p.xy + r*gl_Vertex.xy)-vec3(1.0), kill ? 100.0 : 0.0, 1.0 );
}
