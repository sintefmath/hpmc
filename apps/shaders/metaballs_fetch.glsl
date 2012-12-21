uniform float twist;
uniform vec3 centers[8];

float
HPMC_fetch( vec3 p )
{
    p = 2.0*p - 1.0;
    float rot1 = twist*p.z;
    float rot2 = 0.7*twist*p.y;
    p = mat3( cos(rot1), -sin(rot1), 0,
              sin(rot1),  cos(rot1), 0,
              0,          0, 1)*p;
    p = mat3( cos(rot2), 0, -sin(rot2),
              0, 1,          0,
              sin(rot2), 0,  cos(rot2) )*p;
    p = 0.5*p + vec3(0.5);
    float s = 0.0;
    for(int i=0; i<8; i++) {
        vec3 r = p-centers[i];
        s += 0.05/dot(r,r);
    }
    return s;
}
