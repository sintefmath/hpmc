#include <iostream>
#include <string>
#include <vector>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>

void
compileShader( GLuint shader, const std::string& what )
{
    glCompileShader( shader );

    GLint compile_status;
    glGetShaderiv( shader, GL_COMPILE_STATUS, &compile_status );
    if( compile_status != GL_TRUE ) {
        std::cerr << "Compilation of " << what << " failed, infolog:" << std::endl;

        GLint logsize;
        glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &logsize );

        if( logsize > 0 ) {
            std::vector<GLchar> infolog( logsize+1 );
            glGetShaderInfoLog( shader, logsize, NULL, &infolog[0] );
            std::cerr << std::string( infolog.begin(), infolog.end() ) << std::endl;
        }
        else {
            std::cerr << "Empty log message" << std::endl;
        }
        std::cerr << "Exiting." << std::endl;
        exit( EXIT_FAILURE );
    }
}

// --- compile program and check for errors ------------------------------------
void
linkProgram( GLuint program, const std::string& what )
{
    glLinkProgram( program );

    GLint linkstatus;
    glGetProgramiv( program, GL_LINK_STATUS, &linkstatus );
    if( linkstatus != GL_TRUE ) {
        std::cerr << "Linking of " << what << " failed, infolog:" << std::endl;

        GLint logsize;
        glGetProgramiv( program, GL_INFO_LOG_LENGTH, &logsize );

        if( logsize > 0 ) {
            std::vector<GLchar> infolog( logsize+1 );
            glGetProgramInfoLog( program, logsize, NULL, &infolog[0] );
            std::cerr << std::string( infolog.begin(), infolog.end() ) << std::endl;
        }
        else {
            std::cerr << "Empty log message" << std::endl;
        }
        std::cerr << "Exiting." << std::endl;
        exit( EXIT_FAILURE );
    }
}


std::string gpgpu_vs_110 =
        "void\n"
        "main()\n"
        "{\n"
        "    gl_TexCoord[0] = 0.5*gl_Vertex+vec4(0.5);\n"
        "    gl_Position    = gl_Vertex;\n"
        "}\n";

std::string base_fs_110 =
        "uniform float foo;\n"
        "void\n"
        "main(){\n"
        "  float r = distance( gl_TexCoord[0].xy, vec2(0.5,0.5) ) < 0.5 ? 1.0 : 0.0;\n"
        "  gl_FragColor = vec4( r, 0.0, foo, 1.0 );\n"
        "}\n";

std::string reduce_fs_110 =
        "uniform sampler2D  tex;\n"
        "uniform vec2       delta;\n"
        "void\n"
        "main(){\n"
        "    vec4 sums = vec4( \n"
        "        texture2D( tex, gl_TexCoord[0].xy+delta.xx )+"
        "        texture2D( tex, gl_TexCoord[0].xy+delta.yx )+"
        "        texture2D( tex, gl_TexCoord[0].xy+delta.xy )+"
        "        texture2D( tex, gl_TexCoord[0].xy+delta.yy ) );"
        "    gl_FragColor = sums;\n"
        "}\n";

GLuint base_p;
GLuint reduce_p;

GLsizei size_l2 = 4;
GLuint tex;
std::vector<GLuint> fbos;

void
init()
{
    const char* gpgpu_src_v[1] = { gpgpu_vs_110.c_str() };
    const char* base_src_f[1] = { base_fs_110.c_str() };
    const char* reduce_src_f[1] = { reduce_fs_110.c_str() };

    GLuint gpgpu_v = glCreateShader( GL_VERTEX_SHADER );
    glShaderSource( gpgpu_v, 1, gpgpu_src_v, NULL );
    compileShader( gpgpu_v, "base_v" );

    GLuint base_f = glCreateShader( GL_FRAGMENT_SHADER );
    glShaderSource( base_f, 1, base_src_f, NULL );
    compileShader( base_f, "base_f" );

    GLuint reduce_f = glCreateShader( GL_FRAGMENT_SHADER );
    glShaderSource( reduce_f, 1, reduce_src_f, NULL );
    compileShader( base_f, "reduce_f" );

    base_p = glCreateProgram();
    glAttachShader( base_p, gpgpu_v );
    glAttachShader( base_p, base_f );
    linkProgram( base_p, "base_p" );

    reduce_p = glCreateProgram();
    glAttachShader( reduce_p, gpgpu_v );
    glAttachShader( reduce_p, reduce_f );
    linkProgram( reduce_p, "reduce_p" );


    // Texture
    glGenTextures( 1, &tex );
    glBindTexture( GL_TEXTURE_2D, tex );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, size_l2 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    for( GLsizei i=0; i<=size_l2; i++ ) {
        glTexImage2D( GL_TEXTURE_2D, i,
                      GL_RGBA32F_ARB,
                      1<<(size_l2-i), 1<<(size_l2-i), 0,
                      GL_RGBA, GL_FLOAT,
                      NULL );
    }
    glBindTexture( GL_TEXTURE_2D, 0 );

    // fbo's
    fbos.resize( size_l2+1 );
    glGenFramebuffersEXT( size_l2+1, fbos.data() );
    for( GLsizei i=0; i<=size_l2; i++ ) {
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, fbos[i] );
        glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT,
                                   GL_COLOR_ATTACHMENT0_EXT,
                                   GL_TEXTURE_2D,
                                   tex,
                                   i );
        glDrawBuffer( GL_COLOR_ATTACHMENT0_EXT );
        GLenum status = glCheckFramebufferStatusEXT( GL_FRAMEBUFFER_EXT );
        if( status != GL_FRAMEBUFFER_COMPLETE_EXT ) {
            std::cerr << "Framebuffer (level=" << i << ") is incomplete: " << status << std::endl;
            exit( EXIT_FAILURE );
        }
    }
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
}

void
display()
{
    static int foo = 0;
    foo = foo ^ 1;

    // build base level
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, fbos[0] );
    glViewport( 0, 0, (1<<size_l2), (1<<size_l2) );
    glUseProgram( base_p );
    glUniform1f( glGetUniformLocation( base_p, "foo" ), (float)foo );
    glBegin( GL_TRIANGLE_STRIP );
    glVertex2f(-1.f, -1.f );
    glVertex2f( 1.f, -1.f );
    glVertex2f(-1.f,  1.f );
    glVertex2f( 1.f,  1.f );
    glEnd();



    glUseProgram( reduce_p );
    glUniform1i( glGetUniformLocation( reduce_p, "tex" ), 0 );
    glBindTexture( GL_TEXTURE_2D, tex );
    for(GLsizei k=1; k<=size_l2; k++ ) {
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, k-1 );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, k-1 );

        glUniform2f( glGetUniformLocation( reduce_p, "delta" ),
                    -0.5f/(1<<(size_l2+1-k)),
                     0.5f/(1<<(size_l2+1-k)) );

        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, fbos[k] );
        glViewport( 0, 0, (1<<(size_l2-k)), (1<<(size_l2-k)) );
        glBegin( GL_TRIANGLE_STRIP );
        glVertex2f(-1.f, -1.f );
        glVertex2f( 1.f, -1.f );
        glVertex2f(-1.f,  1.f );
        glVertex2f( 1.f,  1.f );
        glEnd();
    }
    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, size_l2 );

    glBindTexture( GL_TEXTURE_2D, tex );
    std::vector<GLfloat> buf(4);
    glGetTexImage( GL_TEXTURE_2D, size_l2, GL_RGBA, GL_FLOAT, buf.data() );

    bool success =
            (buf[0] == 208.f ) &&
            (buf[1] == 0.f ) &&
            (buf[2] == (foo?256.f:0.f) ) &&
            (buf[3] == 256.f );
    if( ! success ) {
        std::cerr << "Failure, dump of texture:" << std::endl;

        for( GLsizei k=0; k<=size_l2; k++ ) {
            buf.resize( 4<<((size_l2-k)<<1) );
            glGetTexImage( GL_TEXTURE_2D, k, GL_RGBA, GL_FLOAT, buf.data() );
            std::cerr << "Level " << k << std::endl;
            for( GLsizei j=0; j<(1<<(size_l2-k)); j++ ) {
                for( GLsizei i=0; i<(1<<(size_l2-k)); i++ ) {
                    std::cerr << "[" << buf[ 4*((1<<(size_l2-k))*j+i)+0 ]
                              << ","<< buf[ 4*((1<<(size_l2-k))*j+i)+1 ]
                              << ","<< buf[ 4*((1<<(size_l2-k))*j+i)+2 ]
                              << ","<< buf[ 4*((1<<(size_l2-k))*j+i)+3 ]
                              << "] ";

                    //                if( ((i&1) == 1)) {
                    //                    std::cerr << " ";
                    //               }
                }
                std::cerr << std::endl;
                //            if( ((j&1) == 1)) {
                //                std::cerr << std::endl;
                //           }
            }
        }
        exit( EXIT_FAILURE );
    }
    else {
        std::cerr << "success" << std::endl;
    }
    glBindTexture( GL_TEXTURE_2D, 0 );
    glutSwapBuffers();
}

#ifndef APIENTRY
#define APIENTRY
#endif

static void APIENTRY debugLogger( GLenum source,
                                  GLenum type,
                                  GLuint id,
                                  GLenum severity,
                                  GLsizei length,
                                  const GLchar* message,
                                  void* data )
{
    std::cerr << "src=" << source
              << ", type=" << type
              << ", id=" << id
              << ", severity=" << severity
              << ": " << message
              << std::endl;
}


int
main( int argc, char** argv )
{
  glutInit( &argc, argv );
  glutInitContextFlags( GLUT_DEBUG );
//  glutInitContextFlags( GLUT_CORE_PROFILE | GLUT_DEBUG );
//  glutInitContextVersion( 2, 0 );
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
  glutInitWindowSize( 1280, 720 );
  glutCreateWindow( argv[0] );
  glewExperimental = GL_TRUE;
  GLenum glew_error = glewInit();

  if( glewIsSupported( "GL_KHR_debug" ) ) {
      glEnable( GL_DEBUG_OUTPUT_SYNCHRONOUS );
      glDebugMessageCallback( debugLogger, NULL );
      glDebugMessageControl( GL_DONT_CARE,
                             GL_DONT_CARE,
                             GL_DEBUG_SEVERITY_LOW,
                             0, NULL, GL_TRUE );
  }
  //  glutReshapeFunc( reshape );
  glutDisplayFunc( display );
  //glutKeyboardFunc( keyboard );
  glutIdleFunc( glutPostRedisplay );
  init();
  glutMainLoop();
  return EXIT_SUCCESS;
}
