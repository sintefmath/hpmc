#version 140

in vec3 inPosition;
in vec3 texCoords;

out vec4 vel;
out vec4 pos;

uniform sampler3D density;
uniform int active;
uniform vec3 gravity;
uniform float dt;
uniform mat4 modelViewProjection;
void
main()
{
  if( active < gl_VertexID ) {
    vel = vec4( texCoords, 1.0);
    pos = vec4(inPosition, 1.0);
    gl_Position = vec4(-5,0,0,1);
    return;
  }
  vec4 field = texture( density, inPosition );
  vec3 foo = clamp(dot(field.xyz,field.xyz)-3.0,0.0,100.0)*field.xyz/dot(field.xyz,field.xyz);
  vec3 g = 0.01*field.xyz;//ec3(d_x - e_x, d_y-e_y, d_z-e_z);
  vec3 v = texCoords;
  vec3 avoidance = -0.02*pow((max(dot(g,g)-10*dot(v,v),0.0)),2.0)*normalize(g);
  vec3 drag = -0.5*dot(v,v)*normalize(v);
  vec3 p = inPosition;
  float friction = min(0.9+abs(p.z)/0.01, 1.0);
    v = vec3(friction,friction,1.0)*v + dt*(
             gravity + 
             avoidance + 
             drag );
    p = p + dt*v;
    float i = 0.1;
    if( p.z < 0.0 ) {
       v = vec3(v.xy,  -0.1*v.z );
       p = vec3(p.xy,  -p.z );
    }
    else if( p.z > 1.0 ) {
       v = vec3(v.xy,  -i*v.z );
       p = vec3(p.xy, 2.0-p.z );
    }
    if( p.y < 0.0 ) {
       v = vec3(v.x, -i*v.y, v.z);
       p = vec3(p.x,   -p.y, p.z);
    }
    else if( p.y > 1.0 ) {
       v = vec3(v.x,  -i*v.y, v.z);
       p = vec3(p.x, 2.0-p.y, p.z);
    }
    if( p.x < 0.0  ) {
       v = vec3( -i*v.x, v.yz );
       p = vec3( -p.x,  p.yz );
    }
    else if( p.x > 1.0 ) {
       v = vec3(  -i*v.x, v.yz );
       p = vec3( 2.0-p.x, p.yz );
    }
    vel = vec4(v,1);
    pos = vec4(p,1);
    gl_FrontColor = vec4(1.0, 0.5, 0.1, 0.5 );
    gl_Position = modelViewProjection * vec4(p,1);
}
