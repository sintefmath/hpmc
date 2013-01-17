#version 150
layout(triangles) in;
layout(points, max_vertices=1) out;

// geometry shader is run once per triangle and emits one or nil points

// an offset we use to randomize which primitive that generates points
uniform int offset;

// governs how likely it is that a triangle will produce a point
uniform int threshold;
uniform mat4 P;

in vec3 normal[3];
in vec3 position[3];

// varyings that will be recorded
out vec2 info;
out vec3 vel;
out vec3 pos;

void
main()
{
    if( int(offset + gl_PrimitiveIDIn) % threshold == 0 ) {
        int side = (gl_PrimitiveIDIn / threshold) %2;
        info = vec2( 1.0, 1.0 );
//       position new particle on center of triangle
        pos = (1.0/3.0)*( position[0].xyz +
                          position[1].xyz +
                          position[2].xyz )
//       and push it slightly off the surface along the normal direction
            + (side!=0?0.02:-0.02)*normalize( normal[0] +
                                              normal[1] +
                                              normal[2] );
//       initial velocity is zero
        vel = vec3(0.0);
        gl_Position = P * vec4(pos, 1.0);
        EmitVertex();
    }
};
