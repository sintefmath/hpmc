layout(points, invocations=1) in;
layout(triangle_strip, max_vertices=3) out;

uniform mat4            modelviewprojection;
uniform mat3            normalmatrix;



out GO {
    vec3                normal;
} out_g;

void
hp5_downtraverse( out uint pos, out uint key_remainder, in uint key );

void
mc_extract( out vec3 P, out vec3 N, in uint pos, in uint remainder );

void
main()
{
    uint pos;
    uint remainder;
    hp5_downtraverse( pos, remainder, 3*gl_PrimitiveIDIn );


    for(int i=0; i<3; i++ ) {
        vec3 P;
        vec3 N;
        mc_extract( P, N, pos, remainder + i );
        out_g.normal = normalmatrix * N;
        gl_Position = modelviewprojection * vec4( P, 1.0 );
        EmitVertex();
    }
}
