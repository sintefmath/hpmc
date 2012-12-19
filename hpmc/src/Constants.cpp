#include <vector>
#include <cmath>
#include <hpmc_internal.h>
#include <sstream>
#include "Constants.hpp"

HPMCConstants::HPMCConstants( HPMCTarget target, HPMCDebugBehaviour debug )
    : m_target( target ),
      m_debug( debug ),
      m_vertex_counts( this),
      m_gpgpu_quad( this ),
      m_sequence_renderer( this ),
      m_edge_table( this )
{
    switch( target ) {
    case HPMC_TARGET_GL20_GLSL110: m_version_string = "#version 110\n"; break;
    case HPMC_TARGET_GL21_GLSL120: m_version_string = "#version 120\n"; break;
    case HPMC_TARGET_GL30_GLSL130: m_version_string = "#version 130\n"; break;
    case HPMC_TARGET_GL31_GLSL140: m_version_string = "#version 140\n"; break;
    case HPMC_TARGET_GL32_GLSL150: m_version_string = "#version 150\n"; break;
    case HPMC_TARGET_GL33_GLSL330: m_version_string = "#version 330\n"; break;
    case HPMC_TARGET_GL40_GLSL400: m_version_string = "#version 400\n"; break;
    case HPMC_TARGET_GL41_GLSL410: m_version_string = "#version 410\n"; break;
    case HPMC_TARGET_GL42_GLSL420: m_version_string = "#version 420\n"; break;
    case HPMC_TARGET_GL43_GLSL430: m_version_string = "#version 430\n"; break;
    }
}

void
HPMCConstants::init()
{
    m_vertex_counts.init();
    m_gpgpu_quad.init();
    m_sequence_renderer.init();
    m_edge_table.init();
}


HPMCConstants::~HPMCConstants()
{
}
