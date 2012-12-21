#include <hpmc.h>
#include <hpmc_internal.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include "BaseLevelBuilder.hpp"
#include "IsoSurface.hpp"
#include "Constants.hpp"
#include "Field.hpp"
#include "Logger.hpp"

#ifdef _WIN32
#define log2f(x) (logf(x)*1.4426950408889634f)
#endif

using namespace HPMC;
static const std::string package = "HPMC.BaseLevelBuilder";

HPMCBaseLevelBuilder::HPMCBaseLevelBuilder( const HPMCIsoSurface* iso_surface )
    : m_iso_surface( iso_surface ),
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

    m_tile_size[0] = 1u<<(GLsizei)ceilf( log2f( static_cast<float>(m_iso_surface->field().cellsX())/2.0f ) );
    m_tile_size[1] = 1u<<(GLsizei)ceilf( log2f( static_cast<float>(m_iso_surface->field().cellsY())/2.0f ) );

    float aspect = static_cast<float>(m_tile_size[0]) / static_cast<float>(m_tile_size[1]);

    m_layout[0] = 1u<<(GLsizei)std::max( 0.0f, ceilf( log2f( sqrt(static_cast<float>(m_iso_surface->field().cellsZ())/aspect ) ) ) );
    m_layout[1] = (m_iso_surface->field().cellsX() + m_layout[0]-1)/m_layout[0];

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
        glDeleteProgram( m_program );
        m_program = 0;
    }

    // fragment shader
    GLuint fs = HPMCcompileShader( m_iso_surface->constants()->versionString() +
                                   m_iso_surface->field().fetcherSource( false ) +
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
    m_iso_surface->field().setupProgram( m_field_context, m_program );
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
    m_iso_surface->field().bind( m_field_context, field_sampler );

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
    if( !m_iso_surface->field().isBinary() ) {
        glUniform1f( m_loc_threshold, m_iso_surface->threshold() );
    }

    // And trigger computation.
    glBindFramebuffer( GL_FRAMEBUFFER, m_iso_surface->histoPyramid().baseFramebufferObject() );
    glViewport( 0,
                0,
                m_iso_surface->histoPyramid().size(),
                m_iso_surface->histoPyramid().size() );
    m_iso_surface->constants()->gpgpuQuad().render();
    return true;
}

const std::string
HPMCBaseLevelBuilder::extractSource() const
{
    using std::endl;
    Logger log( m_iso_surface->constants(), package + ".extractSource" );

    HPMCTarget target = m_iso_surface->constants()->target();
    std::stringstream src;

    if( log.doDebug() ) {
        src << "// >>> Generated by " << log.where() << endl;
    }

    //          Scale tp from tile parameterization to scalar field parameterization
    //          Now we have found the MC cell, next find which edge that this vertex lies on
    // Output sample positions, if needed.

    src << "void" << endl;
    src << "HPMC_baseLevelExtract( out vec3 a," << endl;
    src << "                       out vec3 b," << endl;
    src << "                       out vec3 p," << endl;
    src << "                       out vec3 n," << endl;
    src << "                       in  float key )" << endl;
    src << "{" << endl;
    src << "    vec2 base_texcoord;\n";
    src << "    float key_ix;\n";
    src << "    float val;\n";
    src << "    HPMC_traverseDown( base_texcoord, val, key_ix, key );\n";
    src << "    vec2 foo = vec2( HPMC_TILES_X_F,HPMC_TILES_Y_F)*base_texcoord;" << endl;
    src << "    vec2 tp = vec2( (2.0*HPMC_TILE_SIZE_X_F)/HPMC_FUNC_X_F," << endl;
    src << "                    (2.0*HPMC_TILE_SIZE_Y_F)/HPMC_FUNC_Y_F ) * fract(foo);" << endl;
    src << "    float slice = dot( vec2(1.0,HPMC_TILES_X_F), floor(foo));" << endl;
    src << "    vec4 edge = texture2D( HPMC_edge_table, vec2((1.0/16.0)*(key_ix+0.5), val ) );" << endl;
    if( m_iso_surface->field().m_binary ) {
        src << "    n = 2.0*fract(edge.xyz)-vec3(1.0);" << endl;
        src << "    edge = floor(edge);\n";
    }
    src << "    vec3 shift = edge.xyz;" << endl;
    src << "    vec3 axis = vec3( equal(vec3(0.0, 1.0, 2.0), vec3(edge.w)) );" << endl;
    //          Calculate sample positions of the two end-points of the edge.
    src << "    vec3 pa = vec3(tp, slice)" << endl;
    src << "            + vec3(1.0/HPMC_FUNC_X_F, 1.0/HPMC_FUNC_Y_F, 1.0)*shift;" << endl;
    src << "    vec3 pb = pa" << endl;
    src << "            + vec3(1.0/HPMC_FUNC_X_F, 1.0/HPMC_FUNC_Y_F, 1.0)*axis;" << endl;
    src << "    a = vec3(pa.x, pa.y, (pa.z+0.5)*(1.0/float(HPMC_FUNC_Z)) );" << endl;
    src << "    b = vec3(pb.x, pb.y, (pb.z+0.5)*(1.0/float(HPMC_FUNC_Z)) );" << endl;
    if( m_iso_surface->field().m_binary ) {
        src << "    p = 0.5*(pa+pb);\n";
    }
    else {
        if( !m_iso_surface->field().hasGradient() ) {
            //          If we don't have gradient info, we approximate the gradient using forward
            //          differences. The sample at pb is one of the forward samples at pa, so we
            //          save one texture lookup.
            src << "    float va = HPMC_sample( pa );" << endl;
            src << "    vec3 na = vec3( HPMC_sample( pa + vec3( 1.0/HPMC_FUNC_X_F, 0.0, 0.0 ) )," << endl;
            src << "                    HPMC_sample( pa + vec3( 0.0, 1.0/HPMC_FUNC_Y_F, 0.0 ) )," << endl;
            src << "                    HPMC_sample( pa + vec3( 0.0, 0.0, 1.0 ) ) );" << endl;
            src << "    vec3 nb = vec3( HPMC_sample( pb + vec3( 1.0/HPMC_FUNC_X_F, 0.0, 0.0 ) )," << endl;
            src << "                    HPMC_sample( pb + vec3( 0.0, 1.0/HPMC_FUNC_Y_F, 0.0 ) )," << endl;
            src << "                    HPMC_sample( pb + vec3( 0.0, 0.0, 1.0 ) ) );" << endl;
            //          Solve linear equation to approximate point that edge pierces iso-surface.
            src << "    float t = (va-HPMC_threshold)/(va-dot(na,axis));" << endl;
        }
        else {
            //          If we have gradient info, sample pa and pb.
            src << "    vec4 fa = HPMC_sampleGrad( pa );" << endl;
            src << "    vec3 na = fa.xyz;" << endl;
            src << "    float va = fa.w;" << endl;
            src << "    vec4 fb = HPMC_sampleGrad( pb );" << endl;
            src << "    vec3 nb = fb.xyz;" << endl;
            src << "    float vb = fb.w;" << endl;
            //          Solve linear equation to approximate point that edge pierces iso-surface.
            src << "    float t = (va-HPMC_threshold)/(va-vb);" << endl;
        }
        src << "    p = mix(pa, pb, t );" << endl;
        src << "    n = vec3(HPMC_threshold)-mix(na, nb,t);" << endl;
    }
    //          p.xy is in normalized texture coordinates, but z is an integer slice number.
    //          First, remove texel center offset
    src << "    p.xy -= vec2(0.5/HPMC_FUNC_X_F, 0.5/HPMC_FUNC_Y_F );" << endl;
    //          And rescale such that domain fits extent.
    src << "    p *= vec3( HPMC_GRID_EXT_X_F * HPMC_FUNC_X_F/(HPMC_CELLS_X_F-0.0)," << endl;
    src << "               HPMC_GRID_EXT_Y_F * HPMC_FUNC_Y_F/(HPMC_CELLS_Y_F-0.0)," << endl;
    src << "               HPMC_GRID_EXT_Z_F * 1.0/(HPMC_CELLS_Z_F) );" << endl;
    src << "    n *= vec3( HPMC_GRID_EXT_X_F/HPMC_CELLS_X_F,"  << endl;
    src << "               HPMC_GRID_EXT_Y_F/HPMC_CELLS_Y_F,"  << endl;
    src << "               HPMC_GRID_EXT_Z_F/HPMC_CELLS_Z_F );"<< endl;
    src << "}" << endl;

    if( log.doDebug() ) {
        src << "// <<< Generated by " << log.where() << endl;
    }
    return src.str();
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
            if( m_iso_surface->field().isBinary() ) {
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
