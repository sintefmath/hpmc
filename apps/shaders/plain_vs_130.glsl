#version 130

in vec3 position;

uniform mat4 PM;

void
main()
{
    gl_Position = PM * vec4(position, 1.0 );
}
