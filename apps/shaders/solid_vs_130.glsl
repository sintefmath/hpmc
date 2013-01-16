#version 130

// prototype for function provided by HPMC
void
extractVertex( out vec3 p, out vec3 n);

// used for transform feedback
out vec3 position;

uniform mat4 PM;

void
main()
{
    vec3 p, n;
    extractVertex( p, n );
    position = p;
    gl_Position = PM * vec4( p, 1.0 );
};
