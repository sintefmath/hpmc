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
#include <glhpmc/glhpmc.hpp>
#include <glhpmc/glhpmc_internal.hpp>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <glhpmc/BaseLevelBuilder.hpp>
#include <glhpmc/IsoSurface.hpp>
#include <glhpmc/Constants.hpp>
#include <glhpmc/Field.hpp>
#include <glhpmc/Logger.hpp>

#ifdef _WIN32
#define log2f(x) (logf(x)*1.4426950408889634f)
#endif

namespace glhpmc {

static const std::string package = "HPMC.BaseLevelBuilder";

HPMCBaseLevelBuilder::HPMCBaseLevelBuilder( const HPMCIsoSurface* iso_surface )
    : m_iso_surface( iso_surface ),
      m_field_context( NULL ),
      m_program(0),
      m_loc_threshold(-1)
{
    m_tile_size[0] = 0;
    m_tile_size[1] = 0;
    m_layout[0] = 0;
    m_layout[1] = 0;
}

HPMCBaseLevelBuilder::~HPMCBaseLevelBuilder()
{

}

bool
HPMCBaseLevelBuilder::configure()
{
    Logger log( m_iso_surface->constants(), package + ".configure" );

    m_tile_size[0] = 1u<<(GLsizei)ceilf( log2f( static_cast<float>(m_iso_surface->cellsX())/2.0f ) );
    m_tile_size[1] = 1u<<(GLsizei)ceilf( log2f( static_cast<float>(m_iso_surface->cellsY())/2.0f ) );

    float aspect = static_cast<float>(m_tile_size[0]) / static_cast<float>(m_tile_size[1]);

    m_layout[0] = 1u<<(GLsizei)std::max( 0.0f, ceilf( log2f( sqrt(static_cast<float>(m_iso_surface->cellsZ())/aspect ) ) ) );
    m_layout[1] = (m_iso_surface->cellsX() + m_layout[0]-1)/m_layout[0];

    m_size_l2 = (GLsizei)ceilf( log2f( static_cast<float>( std::max( m_tile_size[0]*m_layout[0],
                                                                     m_tile_size[1]*m_layout[1] ) ) ) );

    m_layout[0] = (1u<<m_size_l2) / m_tile_size[0];
    m_layout[1] = (1u<<m_size_l2) / m_tile_size[1];

    if( m_iso_surface->constants()->debugBehaviour() != HPMC_DEBUG_NONE ) {
        std::stringstream o;
        o << "tiling.tile_size = [ " << m_tile_size[0] << " x " << m_tile_size[1] << " ], "
             "tiling.layout = [ " << m_layout[0] << " x " << m_layout[1] << " ]";
        log.debugMessage( o.str() );
    }

    // rebuild program
    if( m_program != 0 ) {
        if( m_field_context != NULL ) {
            delete m_field_context;
            m_field_context = NULL;
        }
        glDeleteProgram( m_program );
        m_program = 0;
    }

    // fragment shader
    GLuint fs = HPMCcompileShader( m_iso_surface->constants()->versionString() +
                                   m_iso_surface->field()->fetcherFieldSource() +
//                                   m_iso_surface->oldField().fetcherSource( false ) +
                                   HPMCgenerateDefines( m_iso_surface ) +
                                   fragmentSource(),
                                   GL_FRAGMENT_SHADER );
    if( fs == 0 ) {
        log.errorMessage( "Failed to build base level fragment shader" );
        return false;
    }

    // attach and link
    m_program = glCreateProgram();
    glAttachShader( m_program, m_iso_surface->constants()->gpgpuQuad().passThroughVertexShader() );
    glAttachShader( m_program, fs );
    glDeleteShader( fs );

    if( m_iso_surface->constants()->target() >= HPMC_TARGET_GL30_GLSL130 ) {
        glBindFragDataLocation( m_program, 0, "fragment" );
    }
    m_iso_surface->constants()->gpgpuQuad().configurePassThroughVertexShader( m_program );
    if(!HPMClinkProgram( m_program ) ) {
        log.errorMessage( "Failed to link base level shader program" );
        return false;
    }

    // set up uniforms
    glUseProgram( m_program );
    m_field_context = m_iso_surface->field()->createContext( m_program );
    m_loc_threshold = glGetUniformLocation( m_program, "HPMC_threshold" );
    m_loc_vertex_count_table = glGetUniformLocation( m_program, "HPMC_vertex_count" );

    glUniform1i( m_loc_vertex_count_table, m_iso_surface->m_hp_build.m_tex_unit_1 );

    return true;
}

bool
HPMCBaseLevelBuilder::build( GLuint vertex_table_sampler, GLuint field_sampler )
{
    Logger log( m_iso_surface->constants(), package + ".build" );
    if( m_iso_surface->constants()->debugBehaviour() != HPMC_DEBUG_NONE ) {
        std::stringstream o;
        o << "vtx unit=" << vertex_table_sampler
          << ", field unit=" << field_sampler;
        log.debugMessage( o.str() );
    }


    // --- build base level ----------------------------------------------------
    glUseProgram( m_program );
    m_iso_surface->constants()->gpgpuQuad().bindVertexInputs();
    m_iso_surface->field()->bind( m_field_context );

    // Switch to texture unit given by h->m_hp_build.m_tex_unit_1.
    glUniform1i( m_loc_vertex_count_table, vertex_table_sampler );
    glActiveTextureARB( GL_TEXTURE0_ARB + vertex_table_sampler );

    // To avoid getting GL errors when we bind base level FBOs, we set mipmap
    // levels of the HP texture to zero.
    glBindTexture( GL_TEXTURE_2D, m_iso_surface->histoPyramid().texture() );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    glBindTexture( GL_TEXTURE_2D, 0 );

    // Then bind the vertex count texture to unit h->m_hp_build.m_tex_unit_1.
    glBindTexture( GL_TEXTURE_1D, m_iso_surface->constants()->caseVertexCounts().texture() );

    // Update the threshold uniform
    if( !m_iso_surface->binary() ) {
        glUniform1f( m_loc_threshold, m_iso_surface->threshold() );
    }

    // And trigger computation.
    if( m_iso_surface->constants()->target() < HPMC_TARGET_GL30_GLSL130 ) {
        glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, m_iso_surface->histoPyramid().baseFramebufferObject() );
    }
    else {
        glBindFramebuffer( GL_FRAMEBUFFER, m_iso_surface->histoPyramid().baseFramebufferObject() );
    }

    glViewport( 0,
                0,
                m_iso_surface->histoPyramid().size(),
                m_iso_surface->histoPyramid().size() );
    m_iso_surface->constants()->gpgpuQuad().render();
    m_iso_surface->field()->unbind( m_field_context );
    return true;
}


const std::string
HPMCBaseLevelBuilder::fragmentSource() const
{
    std::stringstream src;
    using std::endl;

    src << "// generated by HPMCgenerateBaselevelShader" << endl;
    std::string texcoord = "gl_TexCoord[0].xy";
    std::string fragment = "gl_FragColor";
    std::string texture1D = "texture1D";
    std::string texture1D_channel = "a";
    if( HPMC_TARGET_GL30_GLSL130 <= m_iso_surface->constants()->target() ) {
        src << "in vec2 texcoord;" << endl;
        src << "out vec4 fragment;" << endl;
        texcoord = "texcoord";
        fragment = "fragment";
        texture1D = "texture";
        texture1D_channel = "r";
    }
    src << "float" << endl;
    src << "HPMC_sample( vec3 p )" << endl;
    src << "{" << endl;
    src << "    p.z = (p.z+0.5)*(1.0/float( HPMC_FUNC_Z) );" << endl;
    src << "    return HPMC_fetch( p );" << endl;
    src << "}" << endl;

    src << "uniform sampler1D  HPMC_vertex_count;" << endl;
    src << "uniform float      HPMC_threshold;" << endl;
    src << "void" << endl;
    src << "main()" << endl;
    src << "{" << endl;
    //          determine which tile we're in, and thus which slice
    src << "    vec2 stp = vec2( HPMC_TILES_X, HPMC_TILES_Y ) * " << texcoord << ";"<< endl;
    src << "    float slice = dot( vec2( 1.0, HPMC_TILES_X ), floor( stp ) );"<<endl;
    //          skip slices that don't contain cells
    src << "    if( slice < float(HPMC_CELLS_Z) ) {"<<endl;
    src << "        vec3 tp = vec3( fract(stp), slice );"<<endl;
    //              scale texcoord from tile parameterization to func parameterization
    src << "        tp.xy *= vec2( 2.0 * HPMC_TILE_SIZE_X_F / HPMC_FUNC_X_F,"   << endl;
    src << "                       2.0 * HPMC_TILE_SIZE_Y_F / HPMC_FUNC_Y_F );" << endl;
    //              cell extent in func parameterization
    src << "        vec2 xr = vec2( (HPMC_CELLS_X_F+0.5) / HPMC_FUNC_X_F,"   << endl;
    src << "                        (HPMC_CELLS_X_F-0.5) / HPMC_FUNC_X_F );" << endl;
    src << "        vec2 yb = vec2( (HPMC_CELLS_Y_F+0.5) / HPMC_FUNC_Y_F,"   << endl;
    src << "                        (HPMC_CELLS_Y_F-0.5) / HPMC_FUNC_Y_F );" << endl;
    //              mask out cells outside specified range
    //              (we process 2x2x1 cells in a fragment)
    src << "        bvec2 xmask = bvec2( tp.x < xr.x,"  << endl;
    src << "                             tp.x < xr.y );"<< endl;
    src << "        bvec2 ymask = bvec2( tp.y < yb.x,"  << endl;
    src << "                             tp.y < yb.y );"<< endl;
    src << "        vec4 mask = vec4( xmask.x && ymask.x,"  << endl;
    src << "                          xmask.y && ymask.x,"  << endl;
    src << "                          xmask.x && ymask.y,"  << endl;
    src << "                          xmask.y && ymask.y );"<< endl;
    //              shift distance between voxels in func parameterization
    src << "        const vec3 delta = vec3( 1.0/HPMC_FUNC_X_F," << endl;
    src << "                                 1.0/HPMC_FUNC_Y_F," << endl;
    src << "                                 1.0 );" << endl;
    //              fetch 3x3x2 neighbourhood from scalar field
    //              and build partial MC codes
    for(int c=0; c<3; c++) {
        src << "        vec3 l"
            << c
            << " = vec3( " << endl;
        for(int i=0; i<6; i++) {
            src << "            (HPMC_sample( tp + delta*vec3( "
                << ((i>>1)-0.5) << ", "
                << (c-0.5) << ", ";
            if( m_iso_surface->binary() ) {
                src << (float)(i&1) << ") ) < 0.5 ? ";
            }
            else {
                src << (float)(i&1) << ") ) < HPMC_threshold ? ";
            }
            src << ((i&1)==0?" 1.0":"16.0") << " : 0.0 )"
                << ((i&1)==0?" +":(i<5?",":"")) << endl;
        }
        src << "        );" << endl;
    }
    //              build codes for 2x2x1 set of voxels,
    //              store code in fractional part
    src << "        vec4 codes = (1.0/256.0)*vec4(" << endl;
    src << "            l0.x+2.0*l0.y+4.0*l1.x +8.0*l1.y+0.5," << endl;
    src << "            l0.y+2.0*l0.z+4.0*l1.y +8.0*l1.z+0.5," << endl;
    src << "            l1.x+2.0*l1.y+4.0*l2.x +8.0*l2.y+0.5," << endl;
    src << "            l1.y+2.0*l1.z+4.0*l2.y +8.0*l2.z+0.5" << endl;
    src << "        );" << endl;
    //              fetch the triangle count for the 2x2x1 set of voxels
    src << "        vec4 counts = vec4(" << endl;
    src << "            " << texture1D << "( HPMC_vertex_count, codes.x )." << texture1D_channel <<"," << endl;
    src << "            " << texture1D << "( HPMC_vertex_count, codes.y )." << texture1D_channel <<"," << endl;
    src << "            " << texture1D << "( HPMC_vertex_count, codes.z )." << texture1D_channel <<"," << endl;
    src << "            " << texture1D << "( HPMC_vertex_count, codes.w )." << texture1D_channel <<"" << endl;
    src << "        );" << endl;

    // encode the vertex count in the integer part and the code in the fractional part.
    src << "        " << fragment << " = mask*( counts + codes);" << endl;
    src << "    } " << endl;
    src << "    else {" << endl;
    src << "        " << fragment << " = vec4(0.0, 0.0, 0.4, 0.0);" << endl;
    src << "    }" << endl;
    src << "}" << endl;

    return src.str();
}

} // of namespace glhpmc
