#version 150

in vec2 vbo_texcoord;
in vec3 vbo_normal;
in vec3 vbo_position;

// input from interleaved GL_T2F_N3F_V3F buffer
// pass output to GS, position in gl_Position
out vec3 invel;
out vec2 ininfo;
out vec3 inpos;

void
main()
{
    invel       = vbo_normal;
    ininfo      = vbo_texcoord;
    inpos       = vbo_position;
    //gl_Position = vec4(vbo_position,1.0);
};
