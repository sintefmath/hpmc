#version 110

varying vec3 normal;

void
main()
{
    vec3 v = vec3( 0.0, 0.0, 1.0 );
    vec3 n = normalize( normal );
    vec3 r = reflect( v, n );
    vec3 h = 0.5*(v+n);
    vec3 c_r = vec3(0.4, 1.3, 2.0) * max( 0.0, -r.y )
             + vec3(0.5, 0.4, 0.2) * pow( max( 0.0, r.y), 3.0 );
    vec3 c_s = vec3(0.7, 0.9, 1.0) * pow( max( 0.0, dot( v, h ) ), 50.0 );
    vec3 c_f = vec3(0.8, 0.9, 1.0) * pow( 1.0-abs(n.z), 5.0 );
    gl_FragColor = vec4( c_r + c_s + c_f, 1.0 );
}


