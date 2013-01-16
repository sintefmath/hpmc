#version 110

// prototype for function provided by HPMC
void
extractVertex( out vec3 p, out vec3 n);

// used for transform feedback
varying vec3 position;

void
main()
{
    vec3 p, n;
    extractVertex( p, n );
    position = p;
    gl_Position = gl_ModelViewProjectionMatrix * vec4( p, 1.0 );
    gl_FrontColor = gl_Color;
}

