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
#include <glhpmc/IsoSurface.hpp>
#include <glhpmc/IsoSurfaceRenderer.hpp>
#include <glhpmc/Constants.hpp>
#include <glhpmc/Logger.hpp>

static const std::string package = "HPMC.IsoSurfaceRenderer";

HPMCIsoSurfaceRenderer::HPMCIsoSurfaceRenderer(HPMCIsoSurface *iso_surface)
    : m_handle( iso_surface ),
      m_program( 0 )
{
}

HPMCIsoSurfaceRenderer::~HPMCIsoSurfaceRenderer()
{
}

const std::string
HPMCIsoSurfaceRenderer::extractionSource() const
{
    return HPMCgenerateDefines( m_handle )
            + m_handle->field().fetcherSource( true )
            + HPMCgenerateExtractVertexFunction( m_handle );
}

bool
HPMCIsoSurfaceRenderer::setProgram( GLuint program,
                                    GLuint tex_unit_work1,
                                    GLuint tex_unit_work2,
                                    GLuint tex_unit_work3 )
{
    Logger log( m_handle->constants(), package + "setProgram" );
    if( program == 0 ) {
        log.errorMessage( "Program == 0" );
        return false;
    }
    HPMCTarget target = m_handle->constants()->target();
    GLint program_link_status;
    GLint scalar_field_sampler_loc  = -1;
    GLint normal_table_sampler_loc  = -1;
    glGetProgramiv( program, GL_LINK_STATUS, &program_link_status );
    if( program_link_status != GL_TRUE ) {
        log.errorMessage( "Passed unsuccessfully linked program");
        return false;
    }
    if( tex_unit_work1 == tex_unit_work2 ) {
        log.errorMessage( "Tex unit 1 == tex unit 2" );
        return false;
    }
    GLint et_loc = glGetUniformLocation( program, "HPMC_edge_table" );
    if( et_loc == -1 ) {
        log.errorMessage( "Unable to find uniform location of edge table sampler" );
        return false;
    }
    GLint hp_loc = glGetUniformLocation( program, "HPMC_histopyramid" );
    if( hp_loc == -1 ) {
        log.errorMessage( "Unable to find uniform location of histopyramid sampler" );
        return false;
    }

    if( m_handle->field().isBinary() ) {
        // Is this used anymore?
        normal_table_sampler_loc = glGetUniformLocation( program, "HPMC_normal_table" );
        //if( normal_table_sampler_loc == -1 ) {
        //    log.errorMessage( "Unable to find uniform location of normal table sampler" );
        //    return false;
        //}
    }
    else {
        if( m_handle->field().m_mode != HPMC_VOLUME_LAYOUT_CUSTOM ) {
            if( tex_unit_work1 == tex_unit_work3 ) {
                log.errorMessage( "Tex unit 1 == tex unit 3" );
                return false;
            }
            if( tex_unit_work2 == tex_unit_work3 ) {
                log.errorMessage( "Tex unit 2 == tex unit 3" );
                return false;
            }
            scalar_field_sampler_loc = glGetUniformLocation( program, "HPMC_scalarfield" );
            if( scalar_field_sampler_loc == -1 ) {
                log.errorMessage( "Unable to find uniform location of field sampler" );
                return false;
            }
        }
    }

    // --- get locations of uniform variables ----------------------------------
    GLint offset_uniform_loc = -1;
    if( target < HPMC_TARGET_GL30_GLSL130 ) {
        offset_uniform_loc = glGetUniformLocation( program, "HPMC_key_offset" );
        if( offset_uniform_loc == -1 ) {
            log.errorMessage( "Unable to find uniform location of offset variable" );
            return false;
        }
    }

    GLint threshold_uniform_loc = -1;
    threshold_uniform_loc = glGetUniformLocation( program, "HPMC_threshold" );
    if( !m_handle->field().m_binary && threshold_uniform_loc == -1 ) {
        log.errorMessage( "Unable to find uniform location of threshold variable" );
        return false;
    }

    // --- store info in handle ------------------------------------------------
    m_program = program;
    m_histopyramid_unit = tex_unit_work1;
    m_edge_decode_unit = tex_unit_work2;
    m_scalarfield_unit = tex_unit_work3;
    m_offset_loc = offset_uniform_loc;
    m_threshold_loc = threshold_uniform_loc;

    glUseProgram( m_program );
    glUniform1i( et_loc, m_edge_decode_unit );
    glUniform1i( hp_loc, m_histopyramid_unit );
    if( m_handle->field().isBinary() ) {
        glUniform1i( normal_table_sampler_loc, m_scalarfield_unit );
    }
    else {
        if( m_handle->field().m_mode != HPMC_VOLUME_LAYOUT_CUSTOM ) {
            glUniform1i( scalar_field_sampler_loc, m_scalarfield_unit );
        }
    }
    return true;
}

bool
HPMCIsoSurfaceRenderer::draw( int transform_feedback_mode, bool flip_orientation )
{
    Logger log( m_handle->constants(), package + ".draw" );
    if( m_program == 0 ) {
        log.errorMessage( "Invalid configuraton: Program == 0" );
        return false;
    }

    GLsizei N = m_handle->vertexCount();

    // --- setup state ---------------------------------------------------------
    glActiveTextureARB( GL_TEXTURE0_ARB + m_histopyramid_unit );
    glBindTexture( GL_TEXTURE_2D, m_handle->histoPyramid().texture() );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL,
                                    m_handle->baseLevelBuilder().log2Size() );

    if( m_handle->field().m_mode != HPMC_VOLUME_LAYOUT_CUSTOM ) {
        glActiveTextureARB( GL_TEXTURE0_ARB + m_scalarfield_unit );
        glBindTexture( GL_TEXTURE_3D, m_handle->field().m_tex );
    }

    glActiveTextureARB( GL_TEXTURE0_ARB + m_edge_decode_unit );

    if( m_handle->field().isBinary() ) {
        if( flip_orientation ) {
            glBindTexture( GL_TEXTURE_2D, m_handle->constants()->edgeTable().textureNormalFlip() );
        }
        else {
            glBindTexture( GL_TEXTURE_2D, m_handle->constants()->edgeTable().textureNormal() );
        }
    }
    else {
        if( flip_orientation ) {
            glBindTexture( GL_TEXTURE_2D, m_handle->constants()->edgeTable().textureFlip() );
        }
        else {
            glBindTexture( GL_TEXTURE_2D, m_handle->constants()->edgeTable().texture() );
        }
    }
    m_handle->constants()->sequenceRenderer().bindVertexInputs();
    glUseProgram( m_program );
    glUniform1f( m_threshold_loc, m_handle->threshold() );

    // --- render triangles ----------------------------------------------------
    if( transform_feedback_mode == 1 ) {
        glBeginTransformFeedback( GL_TRIANGLES );
    }
    else if( transform_feedback_mode == 2 ) {
        glBeginTransformFeedbackNV( GL_TRIANGLES );
    }
    else if( transform_feedback_mode == 3 ) {
        glBeginTransformFeedbackEXT( GL_TRIANGLES );
    }


    m_handle->constants()->sequenceRenderer().render( m_offset_loc, N );

    if( transform_feedback_mode == 1 ) {
        glEndTransformFeedback( );
    }
    else if( transform_feedback_mode == 2 ) {
        glEndTransformFeedbackNV( );
    }
    else if( transform_feedback_mode == 3 ) {
        glEndTransformFeedbackEXT( );
    }
    m_handle->constants()->sequenceRenderer().unbindVertexInputs();

    return true;
}

