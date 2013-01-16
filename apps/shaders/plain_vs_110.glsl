#version 110
// Plain vertex shader used to render geometry fetched by transform feedback.

void
main()
{
    gl_FrontColor = gl_Color;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
