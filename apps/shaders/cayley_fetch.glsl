// evaluates the scalar field
float
HPMC_fetch( vec3 p )
{
    p *= 2.0;
    p -= 1.0;
    return 1.0 - 16.0*p.x*p.y*p.z - 4.0*p.x*p.x - 4.0*p.y*p.y - 4.0*p.z*p.z;
}

// evaluates the gradient as well as the scalar field
vec4
HPMC_fetchGrad( vec3 p )
{
    p *= 2.0;
    p -= 1.0;
    return vec4( -16.0*p.y*p.z - 8.0*p.x,
                 -16.0*p.x*p.z - 8.0*p.y,
                 -16.0*p.x*p.y - 8.0*p.z,
                 1.0 - 16.0*p.x*p.y*p.z - 4.0*p.x*p.x - 4.0*p.y*p.y - 4.0*p.z*p.z );
}
