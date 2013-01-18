#include <cmath>
#include <hpmc_internal.h>
#include <iostream>
#include <sstream>
#include "HistoPyramid.hpp"
#include "GPGPUQuad.hpp"
#include "Constants.hpp"
#include "Logger.hpp"

using namespace HPMC;
using std::endl;

static const std::string package = "HPMC.HistoPyramid";

namespace hpmc {
    namespace resources {
        extern std::string reduction_first_fs_110;
        extern std::string reduction_first_fs_130;
        extern std::string reduction_upper_fs_110;
        extern std::string reduction_upper_fs_130;
    } // of namespace resources
} // of namespace hpmc


HPMCHistoPyramid::HPMCHistoPyramid(HPMCConstants *constants )
    : m_constants( constants ),
      m_size(0),
      m_size_l2(0),
      m_tex(0),
      m_top_pbo(0)
{
    Logger log( m_constants, package + ".constructor" );

    glGenTextures( 1, &m_tex );
    log.setObjectLabel( GL_TEXTURE, m_tex, "histopyramid" );
    glGenBuffers( 1, &m_top_pbo );
    log.setObjectLabel( GL_BUFFER, m_top_pbo, "histopyramid top readback" );
    glBindBuffer( GL_PIXEL_PACK_BUFFER, m_top_pbo );
    glBufferData( GL_PIXEL_PACK_BUFFER,
                  sizeof(GLfloat)*4,
                  NULL,
                  GL_DYNAMIC_READ );
    glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
}

bool
HPMCHistoPyramid::init()
{
    Logger log( m_constants, package + ".init" );

    bool retval = true;

    if( m_constants->target() < HPMC_TARGET_GL30_GLSL130 ) {

        GLuint fs_0 = HPMCcompileShader( m_constants->versionString() +
                                         hpmc::resources::reduction_first_fs_110,
                                         GL_FRAGMENT_SHADER );
        if( fs_0 == 0 ) {
            log.errorMessage( "Failed to compile first reduction fragment shader" );
            retval = false;
        }
        else {
            m_reduce1_program = glCreateProgram();
            glAttachShader( m_reduce1_program, m_constants->gpgpuQuad().passThroughVertexShader() );
            glAttachShader( m_reduce1_program, fs_0 );
            m_constants->gpgpuQuad().configurePassThroughVertexShader( m_reduce1_program );
            if( m_constants->target() >= HPMC_TARGET_GL30_GLSL130 ) {
                glBindFragDataLocation(  m_reduce1_program, 0, "fragment" );
            }
            if( !HPMClinkProgram( m_reduce1_program ) ) {
                log.errorMessage( "Failed to link first reduction program" );
                retval = false;
            }
            else {
                m_reduce1_loc_delta = glGetUniformLocation( m_reduce1_program, "HPMC_delta" );
                m_reduce1_loc_level = glGetUniformLocation( m_reduce1_program, "HPMC_src_level" );
                m_reduce1_loc_hp_tex = glGetUniformLocation( m_reduce1_program, "HPMC_histopyramid" );

                GLuint fs_n = HPMCcompileShader( m_constants->versionString() +
                                                 hpmc::resources::reduction_upper_fs_110,
                                                 GL_FRAGMENT_SHADER );
                if( fs_n == 0 ) {
                    log.errorMessage( "Failed to compile subsequent reduction fragment shader" );
                    retval = false;
                }
                else {
                    m_reducen_program = glCreateProgram();
                    glAttachShader( m_reducen_program, m_constants->gpgpuQuad().passThroughVertexShader() );
                    glAttachShader( m_reducen_program, fs_0 );
                    m_constants->gpgpuQuad().configurePassThroughVertexShader( m_reducen_program );
                    if( m_constants->target() >= HPMC_TARGET_GL30_GLSL130 ) {
                        glBindFragDataLocation(  m_reducen_program, 0, "fragment" );
                    }
                    if( !HPMClinkProgram( m_reducen_program ) ) {
                        log.errorMessage( "Failed to link subsequent reduction program" );
                        retval = false;
                    }
                    else {
                        m_reducen_loc_delta = glGetUniformLocation( m_reducen_program, "HPMC_delta" );
                        m_reducen_loc_level = glGetUniformLocation( m_reducen_program, "HPMC_src_level" );
                        m_reducen_loc_hp_tex = glGetUniformLocation( m_reducen_program, "HPMC_histopyramid" );
                    }
                    glDeleteShader( fs_n );
                }
            }
            glDeleteShader( fs_0 );
        }
    }
    else {

        GLuint fs_0 = HPMCcompileShader( m_constants->versionString() +
                                         hpmc::resources::reduction_first_fs_130,
                                         GL_FRAGMENT_SHADER );
        if( fs_0 == 0 ) {
            log.errorMessage( "Failed to compile first reduction fragment shader" );
            retval = false;
        }
        else {
            m_reduce1_program = glCreateProgram();
            glAttachShader( m_reduce1_program, m_constants->gpgpuQuad().passThroughVertexShader() );
            glAttachShader( m_reduce1_program, fs_0 );
            m_constants->gpgpuQuad().configurePassThroughVertexShader( m_reduce1_program );
            if( m_constants->target() >= HPMC_TARGET_GL30_GLSL130 ) {
                glBindFragDataLocation(  m_reduce1_program, 0, "fragment" );
            }
            if( !HPMClinkProgram( m_reduce1_program ) ) {
                log.errorMessage( "Failed to link first reduction program" );
                retval = false;
            }
            else {
                m_reduce1_loc_delta = glGetUniformLocation( m_reduce1_program, "HPMC_delta" );
                m_reduce1_loc_level = glGetUniformLocation( m_reduce1_program, "HPMC_src_level" );
                m_reduce1_loc_hp_tex = glGetUniformLocation( m_reduce1_program, "HPMC_histopyramid" );

                GLuint fs_n = HPMCcompileShader( m_constants->versionString() +
                                                 hpmc::resources::reduction_upper_fs_130,
                                                 GL_FRAGMENT_SHADER );
                if( fs_n == 0 ) {
                    log.errorMessage( "Failed to compile subsequent reduction fragment shader" );
                    retval = false;
                }
                else {
                    m_reducen_program = glCreateProgram();
                    glAttachShader( m_reducen_program, m_constants->gpgpuQuad().passThroughVertexShader() );
                    glAttachShader( m_reducen_program, fs_0 );
                    m_constants->gpgpuQuad().configurePassThroughVertexShader( m_reducen_program );
                    if( m_constants->target() >= HPMC_TARGET_GL30_GLSL130 ) {
                        glBindFragDataLocation(  m_reducen_program, 0, "fragment" );
                    }
                    if( !HPMClinkProgram( m_reducen_program ) ) {
                        log.errorMessage( "Failed to link subsequent reduction program" );
                        retval = false;
                    }
                    else {
                        m_reducen_loc_delta = glGetUniformLocation( m_reducen_program, "HPMC_delta" );
                        m_reducen_loc_level = glGetUniformLocation( m_reducen_program, "HPMC_src_level" );
                        m_reducen_loc_hp_tex = glGetUniformLocation( m_reducen_program, "HPMC_histopyramid" );
                    }
                    glDeleteShader( fs_n );
                }
            }
            glDeleteShader( fs_0 );
        }
    }
    return retval;
}

bool
HPMCHistoPyramid::build( GLint tex_unit_a )
{
    bool retval = true;
    Logger log( m_constants, package + ".build" );


    // level 1
    if( m_size_l2 >= 1 ) {
        if( m_reduce1_program == 0 ) {
            log.errorMessage( "Failed to build first reduction program" );
            retval = false;
        }
        else {
            glActiveTexture( GL_TEXTURE0 + tex_unit_a );
            glBindTexture( GL_TEXTURE_2D, m_tex );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0 );

            glUseProgram( m_reduce1_program );
            glUniform1i( m_reduce1_loc_hp_tex, tex_unit_a );
            if( m_constants->target() < HPMC_TARGET_GL30_GLSL130 ) {
                glUniform2f( m_reduce1_loc_delta, -0.5f/m_size, 0.5f/m_size );
                glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, m_fbos[1] );
            }
            else {
                glUniform1i( m_reduce1_loc_level, 0 );
                glBindFramebuffer( GL_FRAMEBUFFER, m_fbos[1] );
            }
            glViewport( 0, 0, m_size/2, m_size/2 );
            m_constants->gpgpuQuad().render();

            // levels 2 and up
            if( m_size_l2 >= 2 ) {
                if( m_reducen_program == 0 ) {
                    log.errorMessage( "Subsequent reduction program has not been successfully built." );
                    retval = false;
                }
                else {
                    glUseProgram( m_reducen_program );
                    glUniform1i( m_reducen_loc_hp_tex, tex_unit_a );
                    if( m_constants->target() < HPMC_TARGET_GL30_GLSL130 ) {
                        for( GLsizei m=2; m<= m_size_l2; m++ ) {
                            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, m-1 );
                            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, m-1 );

                            glUniform2f( m_reducen_loc_delta,
                                         -0.5f/(1<<(m_size_l2+1-m)),
                                         0.5f/(1<<(m_size_l2+1-m)) );

                            glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, m_fbos[m] );
                            glViewport( 0, 0, 1<<(m_size_l2-m), 1<<(m_size_l2-m) );
                            m_constants->gpgpuQuad().render();
                        }
                    }
                    else {
                        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
                        for( GLsizei m=2; m<= m_size_l2; m++ ) {
                            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, m-1 );
                            glUniform1i( m_reduce1_loc_level, m-1 );
                            glBindFramebuffer( GL_FRAMEBUFFER, m_fbos[m] );
                            glViewport( 0, 0, 1<<(m_size_l2-m), 1<<(m_size_l2-m) );
                            m_constants->gpgpuQuad().render();
                        }
                    }
                }
            }
        }
    }

    // readback of top element
    glBindBuffer( GL_PIXEL_PACK_BUFFER, m_top_pbo );
    glBindTexture( GL_TEXTURE_2D, m_tex );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, m_size_l2 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, m_size_l2 );
    glGetTexImage( GL_TEXTURE_2D, m_size_l2,GL_RGBA, GL_FLOAT, NULL );
    glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
    m_top_count_updated = false;

    return retval;
}


HPMCHistoPyramid::~HPMCHistoPyramid()
{
    Logger log( m_constants, package + ".destructor" );
    glDeleteBuffers( 1, &m_top_pbo );
    glDeleteTextures( 1, &m_tex );
    if( !m_fbos.empty() ) {
        if( m_constants->target() < HPMC_TARGET_GL30_GLSL130 ) {
            glDeleteFramebuffersEXT( static_cast<GLsizei>( m_fbos.size() ), m_fbos.data() );
        }
        else {
            glDeleteFramebuffers( static_cast<GLsizei>( m_fbos.size() ), m_fbos.data() );
        }
    }
}

GLsizei
HPMCHistoPyramid::count()
{
    if( !m_top_count_updated ) {
        Logger log( m_constants, package + ".count" );
        GLfloat mem[4];
        glBindBuffer( GL_PIXEL_PACK_BUFFER, m_top_pbo );
        glGetBufferSubData( GL_PIXEL_PACK_BUFFER, 0, sizeof(GLfloat)*4, mem );
        glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
        m_top_count = static_cast<GLsizei>( floorf( mem[0] ) )
                    + static_cast<GLsizei>( floorf( mem[1] ) )
                    + static_cast<GLsizei>( floorf( mem[2] ) )
                    + static_cast<GLsizei>( floorf( mem[3] ) );
        m_top_count_updated = true;
    }
    return m_top_count;
}

const std::string
HPMCHistoPyramid::fragmentSource( bool first ) const
{
    bool gl30 = m_constants->target() >= HPMC_TARGET_GL30_GLSL130;
    std::string filter = first ? "floor" : "";

    std::string src;
    if( m_constants->target() < HPMC_TARGET_GL30_GLSL130 ) {
        src = "// generated by HPMCgenerateReductionShader with filter=\"" + filter + "\"\n" +
              "uniform sampler2D  HPMC_histopyramid;\n" +
              "uniform vec2       HPMC_delta;\n" +
              "void\n" +
              "main()\n" +
              "{\n" +
              "    vec4 sums = vec4(\n" +
              "        dot( vec4(1.0), " + filter + "( texture2D( HPMC_histopyramid, gl_TexCoord[0].xy+HPMC_delta.xx ) ) ),\n" +
              "        dot( vec4(1.0), " + filter + "( texture2D( HPMC_histopyramid, gl_TexCoord[0].xy+HPMC_delta.yx ) ) ),\n" +
              "        dot( vec4(1.0), " + filter + "( texture2D( HPMC_histopyramid, gl_TexCoord[0].xy+HPMC_delta.xy ) ) ),\n" +
              "        dot( vec4(1.0), " + filter + "( texture2D( HPMC_histopyramid, gl_TexCoord[0].xy+HPMC_delta.yy ) ) )\n" +
              "    );\n" +
              "    gl_FragColor = sums;\n" +
              "}\n";
    }
    else {
        src = "// generated by HPMCgenerateReductionShader with filter=\"" + filter + "\"\n" +
              "uniform sampler2D  HPMC_histopyramid;\n" +
              "uniform int        HPMC_src_level;\n" +
              "void\n" +
              "main()\n" +
              "{\n" +
              "    ivec2 tp = 2*ivec2( gl_FragCoord.xy );\n" +
              "    vec4 sums = vec4(\n" +
              "        dot( vec4(1.0), " + filter + "( texelFetch( HPMC_histopyramid, tp + ivec2(0,0), HPMC_src_level ) ) ),\n" +
              "        dot( vec4(1.0), " + filter + "( texelFetch( HPMC_histopyramid, tp + ivec2(1,0), HPMC_src_level ) ) ),\n" +
              "        dot( vec4(1.0), " + filter + "( texelFetch( HPMC_histopyramid, tp + ivec2(0,1), HPMC_src_level ) ) ),\n" +
              "        dot( vec4(1.0), " + filter + "( texelFetch( HPMC_histopyramid, tp + ivec2(1,1), HPMC_src_level ) ) )\n" +
              "    );\n" +
              "    gl_FragColor = sums;\n" +
              "}\n";
    }
    return src;
}

bool
HPMCHistoPyramid::configure( GLsizei size_l2 )
{
    Logger log( m_constants, package + ".configure");


    bool retval = true;

    m_top_count = 0;
    m_top_count_updated = false;
    m_size_l2 = size_l2;
    m_size = 1<<m_size_l2;

    // resize texture
    glBindTexture( GL_TEXTURE_2D, m_tex );
    glTexImage2D( GL_TEXTURE_2D, 0,
                  GL_RGBA32F_ARB,
                  m_size, m_size, 0,
                  GL_RGBA, GL_FLOAT,
                  NULL );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, m_size_l2 );
    glGenerateMipmap( GL_TEXTURE_2D );

    // release old fbos and set up new
    if( m_constants->target() < HPMC_TARGET_GL30_GLSL130 ) {
        if( !m_fbos.empty() ) {
            glDeleteFramebuffersEXT( static_cast<GLsizei>( m_fbos.size() ), m_fbos.data() );
        }
        m_fbos.resize( m_size_l2 + 1 );
        glGenFramebuffersEXT( static_cast<GLsizei>( m_fbos.size() ), m_fbos.data() );

        for( GLuint m=0; m<m_fbos.size(); m++) {
            glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, m_fbos[m] );
            glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT,
                                       GL_COLOR_ATTACHMENT0_EXT,
                                       GL_TEXTURE_2D,
                                       m_tex,
                                       m );
            glDrawBuffer( GL_COLOR_ATTACHMENT0_EXT );
            GLenum status = glCheckFramebufferStatus( GL_FRAMEBUFFER_EXT );
            if( status != GL_FRAMEBUFFER_COMPLETE_EXT ) {
                std::stringstream o;
                o << "Framebuffer for HP level " << m << " of " << m_size_l2 << " is incomplete";
                log.errorMessage( o.str() );
                retval = false;
            }
        }
    }
    else {
        if( !m_fbos.empty() ) {
            glDeleteFramebuffers( static_cast<GLsizei>( m_fbos.size() ), m_fbos.data() );
        }
        m_fbos.resize( m_size_l2 + 1 );
        glGenFramebuffers( static_cast<GLsizei>( m_fbos.size() ), m_fbos.data() );

        for( GLuint m=0; m<m_fbos.size(); m++) {
            glBindFramebuffer( GL_FRAMEBUFFER, m_fbos[m] );
            glFramebufferTexture2D( GL_FRAMEBUFFER,
                                    GL_COLOR_ATTACHMENT0,
                                    GL_TEXTURE_2D,
                                    m_tex,
                                    m );
            glDrawBuffer( GL_COLOR_ATTACHMENT0 );
            GLenum status = glCheckFramebufferStatus( GL_FRAMEBUFFER );
            if( status != GL_FRAMEBUFFER_COMPLETE ) {
                std::stringstream o;
                o << "Framebuffer for HP level " << m << " of " << m_size_l2 << " is incomplete";
                log.errorMessage( o.str() );
                retval = false;
            }
        }
    }

    if( log.doDebug() ) {
        std::stringstream o;
        o << "histopyramid.size = 2^" << m_size_l2 << " = " << m_size;
        log.debugMessage( o.str() );
    }
    return retval;
}


const std::string
HPMCHistoPyramid::downTraversalSource() const
{
    using std::endl;

    Logger log( m_constants, package + ".downTraversalSource" );


    HPMCTarget target = m_constants->target();
    std::stringstream src;

    std::string texture2DLod = (target < HPMC_TARGET_GL30_GLSL130?"texture2DLod":"textureLod");


    //        Start traversal in the center of the top element texel.
    //        Texel shift offsets, one element per interval, updated during traversal
    //        Sums for the four sub-pyramids below, from the HP tex
    //        Histograms: [sum.x, sum.x+sum.y, sum.x+sum.y+sum.z, sum.x+sum.y+sum.z+sums.w]
    //        Texel shift offset mask: key < hist
    //              Fetch sub-pyramid sums for the four sub-pyramids.
    //              Determine accummulative sums, refer to key intervals in paper.
    //              hist.x = sums.x
    //              hist.y = sums.x + sums.y
    //              hist.z = sums.x + sums.y + sums.z
    //              hist.w = sums.x + sums.y + hist.z + sums.w
    //              Build a mask for next step.
    //              0      <= key < hist.x  ->  mask = (1,1,1,1)
    //              hist.x <= key < hist.y  ->  mask = (0,1,1,1)
    //              hist.y <= key < hist.z  ->  mask = (0,0,1,1)
    //              hist.z <= key < hist.w  ->  mask = (0,0,0,1)
    //              Combine mask with delta_x and delta_y to shift texcoord
    //              0      <= key < hist.x  ->  tp += ( -0.25, -0.25 )
    //              hist.x <= key < hist.y  ->  tp += (  0.25, -0.25 )
    //              hist.y <= key < hist.z  ->  tp += ( -0.25,  0.25 )
    //              hist.z <= key < hist.w  ->  tp += (  0.25,  0.25 )
    //          MC codes are stored in the fractional part, floor extracts the vertex count.
    //          The final traversal step determines which of the four elements in the
    //          baselevel that we want to descend into

    if( log.doDebug() ) {
        src << "// >>> Generated by " << log.where() << endl;
    }
    src << "void" << endl;
    src << "HPMC_traverseDown( out vec2  base_texcoord," << endl;
    src << "                   out float base_value," << endl;
    src << "                   out float key_remainder," << endl;
    src << "                   in  float key_ix )" << endl;
    src << "{" << endl;
    src << "    vec2 texpos = vec2(0.5);" << endl;
    src << "    vec4 delta_x = vec4( -0.5,  0.5, -0.5, 0.25 );" << endl;
    src << "    vec4 delta_y = vec4(  0.0, -0.5,  0.0, 0.25 );" << endl;
    src << "    for( int i = " << m_size_l2 << "; i>0; i-- ) {" << endl;
    src << "        vec4 sums = " << texture2DLod << "( HPMC_histopyramid, texpos, float(i) );" << endl;
    src << "        vec4 hist = sums;" << endl;
    src << "        hist.w   += hist.z;" << endl;
    src << "        hist.zw  += hist.yy;" << endl;
    src << "        hist.yzw += hist.xxx;" << endl;
    src << "        vec4 mask = vec4( lessThan( vec4(key_ix), hist ) );" << endl;
    src << "        texpos   += vec2( dot( mask, delta_x ), dot( mask, delta_y ) );" << endl;
    src << "        key_ix   -= dot( sums.xyz, vec3(1.0)-mask.xyz );" << endl;
    src << "        delta_x  *= 0.5;" << endl;
    src << "        delta_y  *= 0.5;" << endl;
    src << "    }" << endl;
    src << "    vec4 raw = " << texture2DLod << "( HPMC_histopyramid, texpos, 0.0 );" << endl;
    src << "    vec4 sums = floor( raw );" << endl;
    src << "    vec4 hist = sums;" << endl;
    src << "    hist.w   += hist.z;" << endl;
    src << "    hist.zw  += hist.yy;" << endl;
    src << "    hist.yzw += hist.xxx;" << endl;
    src << "    vec4 mask = vec4( lessThan( vec4(key_ix), hist ) );" << endl;
    src << "    float nib = dot(vec4(mask), vec4(-1.0,-1.0,-1.0, 3.0));" << endl;
    src << "    base_texcoord = texpos + vec2( dot( mask, delta_x ), dot( mask, delta_y ) );" << endl;
    src << "    key_remainder = key_ix - dot( sums.xyz, vec3(1.0)-mask.xyz );" << endl;
    src << "    base_value = fract( dot( raw, vec4(equal(vec4(nib),vec4(0,1,2,3))) ) );" << endl;
    src << "}" << endl;

    return src.str();
}
