#version 150

varying in vec3 normal_cs;
varying in float grad_length;
void
main()
{
    const vec3 v = vec3(0.0, 0.0, 1.0 );
    vec3 l = normalize(vec3(1.0, 1.0, 1.0));
    vec3 h = normalize( v + l );
    vec3 cn = normalize( normal_cs );
    float diff = max(0.0,dot( cn, l ) )
               + max(0.0,dot(-cn, l ) );
    float spec = pow( max( 0.0, dot( cn, h) ), 30.0 )
               + pow( max( 0.0, dot(-cn, h) ), 30.0 );
    gl_FragColor = vec4( 0.1, 0.2, 0.7, 0.0) * diff
                 + vec4( 1.0, 1.0, 1.0, 0.0) * spec;
};
