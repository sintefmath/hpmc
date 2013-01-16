#version 150

out vec3 normal_cs;
out vec3 position_cs;

uniform mat4 P;     // projection matrix
uniform mat4 M;     // modelview matrix
uniform mat3 NM;    // normal matrix

// prototype for function provided by HPMC
void
extractVertex( out vec3 p, out vec3 n);

void
main()
{
    vec3 p, n;
    extractVertex( p, n );
    vec4 pp = M * vec4( p, 1.0 );
    vec3 cn = normalize( NM * n );
    normal_cs = cn;
    position_cs = (1.0/pp.w)*pp.xyz;
    gl_Position = P * pp;
};
