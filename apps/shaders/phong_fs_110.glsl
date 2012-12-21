#version 110

varying vec3 normal;

void
main()
{
    const vec3 v = vec3( 0.0, 0.0, 1.0 );
    vec3 l = normalize( vec3( 1.0, 1.0, 1.0 ) );
    vec3 h = normalize( v+l );
    vec3 n = normalize( normal );
    float diff = max( 0.1, dot( n, l ) );
    float spec = pow( max( 0.0, dot(n, h)), 20.0);
    gl_FragColor = diff * gl_Color + spec * vec4(1.0);
}

