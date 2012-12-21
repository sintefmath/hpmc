#version 130

out vec3 normal;
uniform mat4 PM;
uniform mat3 NM;

// prototype for function provided by HPMC
void
extractVertex( out vec3 p, out vec3 n);

void
main()
{
    vec3 p, n;
    extractVertex( p, n );
    gl_Position = PM * vec4( p, 1.0 );
    normal = NM * n;
}
