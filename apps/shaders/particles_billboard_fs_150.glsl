#version 150

in vec2 tp;
in float depth;

out vec4 fragment;

uniform vec3 color;

void
main()
{
    fragment = pow((max(1.0-length(tp),0.0)),2.0)*vec4(color,1.0);
// for some reason the depth test doesn't work as expected if the depth
// isn't written... at least on my setup.
    gl_FragDepth = depth;
}
