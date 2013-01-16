#version 150
// interleaved arrays with GL_N3F_V3F is assumed
in vec3 vbo_normal;
in vec3 vbo_position;

out vec3 normal;
out vec3 position;

void
main()
{
    normal = vbo_normal;
    position = vbo_position;
    //gl_Position = vec4( vbo_position, 1.0 );
};
