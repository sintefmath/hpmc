/* Copyright STIFTELSEN SINTEF 2012
 *
 * This file is part of the HPMC Library.
 *
 * Author(s): Christopher Dyken, <christopher.dyken@sintef.no>
 *
 * HPMC is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * HPMC is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * HPMC.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <GL/glew.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cuhpmc/Constants.hpp>
#include <cuhpmc/FieldGLBufferUChar.hpp>
#include <cuhpmc/EmitterTriVtxGL.hpp>
#include <cuhpmc/IsoSurfaceGL.hpp>

namespace cuhpmc {
    namespace resources {
        extern std::string hp5_downtraversal_430;
        extern std::string mc_extract_430;
        extern std::string mc_per_cell_430;
        extern std::string gl_direct_vs_430;
        extern std::string gl_direct_gs_430;
        extern std::string gl_direct_fs_430;
    } // of namespace resources

GLWriter::GLWriter( IsoSurfaceGL* iso_surface )
    : EmitterTriVtx( iso_surface ),
      m_conf_constmem_apex( true ),
      m_program( 0 )
{
    std::stringstream defines;
    defines << "#define CUHPMC_LEVELS            " << m_iso_surface->hp5Levels() << "\n";
    defines << "#define CUHPMC_CHUNKS_X          " << m_iso_surface->hp5Chunks().x << "\n";
    defines << "#define CUHPMC_CHUNKS_Y          " << m_iso_surface->hp5Chunks().y << "\n";
    defines << "#define CUHPMC_CHUNKS_Z          " << m_iso_surface->hp5Chunks().z << "\n";
    defines << "#define CUHPMC_FIELD_ROW_PITCH   " << m_field->width() << "\n";
    defines << "#define CUHPMC_FIELD_SLICE_PITCH " << (m_field->width()*m_field->height()) << "\n";
    defines << "#define CUHPMC_SCALE_X           " << (1.f/m_field->width()) << "\n";
    defines << "#define CUHPMC_SCALE_Y           " << (1.f/m_field->height()) << "\n";
    defines << "#define CUHPMC_SCALE_Z           " << (1.f/m_field->depth()) << "\n";
    defines << "#define CUHPMC_SIZE              " << m_iso_surface->hp5Size() << "\n";
    if( m_conf_constmem_apex ) {
        defines << "#define CUHMPC_CONF_CONSTMEM_APEX\n";
    }

    const std::string vs_src = "#version 430\n"
                             + defines.str()
                             + resources::gl_direct_vs_430
                             + resources::mc_per_cell_430
                             + resources::hp5_downtraversal_430;
    const std::string gs_src = "#version 430\n"
                             + defines.str()
                             + resources::gl_direct_gs_430
                             + resources::mc_extract_430;

    const std::string fs_src = resources::gl_direct_fs_430;
    std::stringstream out;

    GLint vs = compileShader( out, vs_src, GL_VERTEX_SHADER );
    if( vs != 0 ) {
        GLint gs = compileShader( out, gs_src, GL_GEOMETRY_SHADER );
        if( gs != 0 ) {
            GLint fs = compileShader( out, fs_src, GL_FRAGMENT_SHADER );
            if( fs != 0 ) {
                m_program = glCreateProgram();
                glAttachShader( m_program, vs );
                glAttachShader( m_program, gs );
                glAttachShader( m_program, fs );
                if( linkShaderProgram( out, m_program ) ) {
                    GLint loc = glGetUniformLocation( m_program, "hp5_offsets" );
                    glProgramUniform1uiv( m_program, loc, m_iso_surface->hp5Levels(), m_iso_surface->hp5Offsets().data() );
                    std::cerr << "loc=" << loc << "\n";
                    for(int i=0; i<m_iso_surface->hp5Levels(); i++ ) {
                        std::cerr << m_iso_surface->hp5Offsets()[i] << "\n";
                    }
                    m_loc_iso = glGetUniformLocation( m_program, "iso" );
                    m_loc_mvp = glGetUniformLocation( m_program, "modelviewprojection" );
                    m_loc_nm = glGetUniformLocation( m_program, "normalmatrix" );
                    if( m_conf_constmem_apex ) {
                        m_block_ix_apex = glGetUniformBlockIndex( m_program, "HP5Apex" );
                    }
                    else {
                        m_block_ix_apex = -1;
                    }
                }
                else {
                    glDeleteProgram( m_program );
                    m_program = 0;
                }
                glDeleteShader( fs );
            }
            glDeleteShader( gs );
        }
        glDeleteShader( vs );
    }
    if( out.tellp() != 0 ) {
        out << "--- vertex source --------------------------------------------------------------\n";
        dumpSource( out, vs_src );
        out << "--- geometry source ------------------------------------------------------------\n";
        dumpSource( out, gs_src );
        out << "--- fragment source ------------------------------------------------------------\n";
        dumpSource( out, fs_src );
        std::cerr << out.str() << "\n";

    }
    if( m_program == 0 ) {
        throw std::runtime_error( "Failed to build shader program" );
    }

}

GLWriter::~GLWriter()
{

}

void
GLWriter::render( const GLfloat* modelview_projection,
                        const GLfloat* normal_matrix,
                        cudaStream_t stream )
{

    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, m_constants->caseIntersectEdgeGL() );
    if( FieldGLBufferUChar* f = dynamic_cast<FieldGLBufferUChar*>( m_field ) ) {
        glActiveTexture( GL_TEXTURE1 );
        glBindTexture( GL_TEXTURE_BUFFER, f->fieldGLTex() );


        if( IsoSurfaceGL* i = dynamic_cast<IsoSurfaceGL*>( m_iso_surface ) ) {
            glActiveTexture( GL_TEXTURE2 );
            glBindTexture( GL_TEXTURE_BUFFER, i->hp5GLTex() );
            glActiveTexture( GL_TEXTURE3 );
            glBindTexture( GL_TEXTURE_BUFFER, i->caseGLTex() );

            if( m_conf_constmem_apex ) {
                glBindBufferBase( GL_UNIFORM_BUFFER, m_block_ix_apex, i->hp5GLBuf() );
            }

            glUseProgram( m_program );

            glUniform1f( m_loc_iso, m_iso_surface->iso() );
            glUniformMatrix4fv( m_loc_mvp, 1, GL_FALSE, modelview_projection );
            glUniformMatrix3fv( m_loc_nm, 1, GL_FALSE, normal_matrix );

            glBindBuffer( GL_DRAW_INDIRECT_BUFFER, i->hp5GLBuf() );
            glDrawArraysIndirect( GL_POINTS, NULL );
            glBindBuffer( GL_DRAW_INDIRECT_BUFFER, 0 );

        }
    }
    if( m_conf_constmem_apex ) {
        glBindBufferBase( GL_UNIFORM_BUFFER, m_block_ix_apex, 0 );
    }
    glUseProgram( 0 );
    glActiveTexture( GL_TEXTURE3 );
    glBindTexture( GL_TEXTURE_BUFFER, 0);
    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, 0 );
    glActiveTexture( GL_TEXTURE2 );
    glBindTexture( GL_TEXTURE_BUFFER, 0 );
    glActiveTexture( GL_TEXTURE1 );
    glBindTexture( GL_TEXTURE_BUFFER,0 );

#if 0
    glEnable( GL_DEPTH_TEST );
    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_BUFFER, field_tex );
    glActiveTexture( GL_TEXTURE1 );
    glBindTexture( GL_TEXTURE_BUFFER, hp5_hp_tex );
    glActiveTexture( GL_TEXTURE2 );
    glBindTexture( GL_TEXTURE_2D, isec_tex );
    glActiveTexture( GL_TEXTURE3 );
    glBindTexture( GL_TEXTURE_BUFFER, case_tex );
    glBindVertexArray( extract_vao );

    if( wireframe ) {
        glUseProgram( extract_solid_p );
        if( extract_apex_constmem ) {
            glBindBufferBase( GL_UNIFORM_BUFFER, extract_solid_apex_ub_ix, hp5_hp_buf );
        }
        glUniformMatrix4fv( extract_solid_modelviewprojection_loc, 1, GL_FALSE, PM );
        glUniformMatrix3fv( extract_solid_normalmatrix_loc, 1, GL_FALSE, NM );
        glUniform1f( extract_solid_iso_loc, iso );

        glPolygonOffset( 1.f, 1.f );
        glEnable( GL_POLYGON_OFFSET_FILL );
        glUniform4f( extract_solid_color_loc, 0.0f, 0.3f, 0.0f, 1.f );
        if( readback ) {
            glDrawArrays( GL_TRIANGLES, 0, N );
        }
        else {
            glBindBuffer( GL_DRAW_INDIRECT_BUFFER, hp5_hp_buf );
            glDrawArraysIndirect( GL_TRIANGLES, NULL );
            glBindBuffer( GL_DRAW_INDIRECT_BUFFER, 0 );
        }
        glDisable( GL_POLYGON_OFFSET_FILL );

        glUniform4f( extract_solid_color_loc, 0.3f, 1.0f, 0.3f, 1.f );
//        glUniform4f( extract_solid_color_loc, 1.0f, 1.0f, 1.0f, 1.f );
        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
        if( readback ) {
            glDrawArrays( GL_TRIANGLES, 0, N );
        }
        else {
            glBindBuffer( GL_DRAW_INDIRECT_BUFFER, hp5_hp_buf );
            glDrawArraysIndirect( GL_TRIANGLES, NULL );
            glBindBuffer( GL_DRAW_INDIRECT_BUFFER, 0 );
        }
        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
    }
    else {
        glUseProgram( extract_p );


        if( extract_apex_constmem ) {
            glBindBufferBase( GL_UNIFORM_BUFFER, extract_apex_ub_ix, hp5_hp_buf );
        }
        glUniformMatrix4fv( extract_modelviewprojection_loc, 1, GL_FALSE, PM );
        glUniformMatrix3fv( extract_normalmatrix_loc, 1, GL_FALSE, NM );
        glUniform1f( extract_iso_loc, iso );

        if( appearance == APPEARANCE_TRANSPARENT || appearance == APPEARANCE_GOO ) {
            glEnable( GL_BLEND );
            glDepthFunc( GL_ALWAYS );
//            glDepthMask( GL_TRUE );
            glBlendFunc( GL_ONE, GL_ONE_MINUS_SRC_COLOR );
        }

        if( readback ) {
            glDrawArrays( GL_TRIANGLES, 0, N );
        }
        else {
            glBindBuffer( GL_DRAW_INDIRECT_BUFFER, hp5_hp_buf );
            glDrawArraysIndirect( GL_TRIANGLES, NULL );
            glBindBuffer( GL_DRAW_INDIRECT_BUFFER, 0 );
        }

        if( appearance == APPEARANCE_TRANSPARENT || appearance == APPEARANCE_GOO  ) {
            glDepthFunc( GL_LESS );
//            glDepthMask( GL_TRUE );
            glDisable( GL_BLEND );
        }


    }
    glUseProgram( 0 );

#endif

}

void
GLWriter::dumpSource( std::stringstream& out, const std::string& input )
{
    int line = 1;
    for( std::string::const_iterator it = input.begin(); it!=input.end(); ++it ) {
        std::string::const_iterator jt = it;
        int c=0;
        out << std::setw(3) << line << ": ";
        for(; *jt != '\n' && jt != input.end(); jt++) {
            out << *jt;
            c++;
        }
        out << "\n";
        line ++;
        it = jt;
        if(jt == input.end() )
            break;
    }
}

GLuint
GLWriter::compileShader( std::stringstream& out, const std::string& src, GLenum type ) const
{
    GLuint shader = glCreateShader( type );
    if( type == 0 ) {
        return 0;
    }
    const GLchar* srcs[1] = {
        src.c_str()
    };
    glShaderSource( shader, 1, srcs, NULL );
    glCompileShader( shader );

    GLint logsize;
    glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &logsize );
    if( logsize > 1 ) {
        std::vector<GLchar> infolog( logsize+1 );
        glGetShaderInfoLog( shader, logsize, NULL, infolog.data() );
        switch( type ) {
        case GL_VERTEX_SHADER:
            out << "--- GLSL vertex shader compiler ------------------------------------------------\n";
            break;
        case GL_TESS_CONTROL_SHADER:
            out << "--- GLSL tessellation control shader compiler ----------------------------------\n";
            break;
        case GL_TESS_EVALUATION_SHADER:
            out << "--- GLSL tessellation evaluation shader compiler -------------------------------\n";
            break;
        case GL_GEOMETRY_SHADER:
            out << "--- GLSL geometry shader compiler ----------------------------------------------\n";
            break;
        case GL_FRAGMENT_SHADER:
            out << "--- GLSL fragment shader compiler ----------------------------------------------\n";
            break;
        }
        out << std::string( infolog.data() );
    }

    GLint status;
    glGetShaderiv( shader, GL_COMPILE_STATUS, &status );
    if( status != GL_TRUE ) {
        glDeleteShader( shader );
        shader = 0;
    }
    return shader;
}

bool
GLWriter::linkShaderProgram( std::stringstream& out, GLuint program ) const
{
    glLinkProgram( program );

    GLint logsize;
    glGetProgramiv( program, GL_INFO_LOG_LENGTH, &logsize );
    if( logsize > 1 ) {
        std::vector<GLchar> infolog( logsize+1 );
        glGetProgramInfoLog( program, logsize, NULL, infolog.data() );
        out << "--- GLSL linker ----------------------------------------------------------------\n";
        out << std::string( infolog.data() );
    }

    GLint status;
    glGetProgramiv( program, GL_LINK_STATUS, &status );
    return status == GL_TRUE;
}




} // of namespace cuhpmc
