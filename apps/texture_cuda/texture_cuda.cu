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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <builtin_types.h>
#include <sstream>
#include <vector>
#include <iomanip>
#include <iostream>
#include "../common/common.hpp"
#include <cuhpmc/Constants.hpp>
#include <cuhpmc/FieldGlobalMemUChar.hpp>
#include <cuhpmc/FieldGLBufferUChar.hpp>
#include <cuhpmc/IsoSurfaceCUDA.hpp>
#include <cuhpmc/IsoSurfaceGL.hpp>
#include <cuhpmc/EmitterTriVtxCUDA.hpp>
#include <cuhpmc/EmitterTriVtxGL.hpp>
#include <cuhpmc/IsoSurfaceIndexed.hpp>
#include <cuhpmc/EmitterTriIdx.hpp>

namespace resources {
    extern std::string phong_vbo_vs_420;
    extern std::string phong_fs_130;
    extern std::string cayley_fetch;
}

#define CHECK_CUDA do { \
    cudaError_t error = cudaGetLastError(); \
    if( error != cudaSuccess ) { \
        std::cerr << __FILE__ << '@' << __LINE__ \
                  << ": " << cudaGetErrorString( error )  \
                  << std::endl; \
        exit( EXIT_FAILURE ); \
    } \
} while(0)

#define CUDA_CHECKED(a) do { \
    cudaError_t error = (a); \
    if( error != cudaSuccess ) { \
        std::cerr << __FILE__ << '@' << __LINE__ \
                  << ": " << cudaGetErrorString( error )  \
                  << std::endl; \
        exit( EXIT_FAILURE ); \
    } \
} while(0)

using std::cerr;
using std::endl;

int                             volume_size_x       = 128;
int                             volume_size_y       = 128;
int                             volume_size_z       = 128;
float                           iso                 = 0.5f;

enum Mode {
    OPENGL_NON_INDEXED,
    CUDA_NON_INDEXED,
    CUDA_INDEXED

};
Mode                            mode                = CUDA_INDEXED;

unsigned char*                  field_data_dev      = NULL;

// mode OpenGL non-indexed
cuhpmc::Constants*              gl_n_constants           = NULL;
cuhpmc::FieldGLBufferUChar*     gl_n_field               = NULL;
cuhpmc::IsoSurfaceGL*           gl_n_isurf         = NULL;
cuhpmc::GLWriter*               gl_n_writer              = NULL;


// mode CUDA non-indexed
cuhpmc::Constants*              cu_n_constants           = NULL;
cuhpmc::FieldGlobalMemUChar*    cu_n_field               = NULL;
cuhpmc::IsoSurfaceCUDA*         cu_n_isurf         = NULL;
cuhpmc::EmitterTriVtxCUDA*      cu_n_writer              = NULL;


// mode CUDA indexed
cuhpmc::Constants*              cu_i_constants           = NULL;
cuhpmc::FieldGlobalMemUChar*    cu_i_field               = NULL;
cuhpmc::IsoSurfaceIndexed*      cu_i_isurf           = NULL;
cuhpmc::EmitterTriIdx*          cu_i_writer         = NULL;


GLuint                          vertices_vao         = 0;
GLuint                          vertices_vbo         = 0;
GLsizei                         vertices_vbo_n       = 0;
cudaGraphicsResource*           vertices_resource    = NULL;

GLuint                          triangles_vao         = 0;
GLuint                          triangles_vbo         = 0;
GLsizei                         triangles_vbo_n       = 0;
cudaGraphicsResource*           triangles_resource    = NULL;

cudaStream_t                    stream              = 0;
float*                          surface_cuda_d      = NULL;
cudaEvent_t                     pre_buildup         = 0;
cudaEvent_t                     post_buildup        = 0;
float                           buildup_ms          = 0.f;

cudaEvent_t                     pre_write           = 0;
cudaEvent_t                     post_write          = 0;
float                           write_ms            = 0.f;

uint                            runs                = 0;

bool                            profile             = false;


GLuint                          gl_field_buffer     = 0;


GLuint                          vbo_render_prog     = 0;
GLint                           vbo_render_loc_pm   = -1;
GLint                           vbo_render_loc_nm   = -1;
GLint                           vbo_render_loc_col  = -1;

template<class type, bool clamp, bool half_float>
__global__
void
bumpyCayley( type* __restrict__ output,
             const uint field_x,
             const uint field_y,
             const uint field_z,
             const uint field_row_pitch,
             const uint field_slice_pitch,
             const uint by )
{
    uint ix = blockIdx.x*blockDim.x + threadIdx.x;
    uint iy = (blockIdx.y%by)*blockDim.y + threadIdx.y;
    uint iz = blockIdx.y/by;
    if( ix < field_x && iy < field_y && iz < field_z ) {
        float x = 2.f*(float)ix/(float)field_x-1.f;
        float y = 2.f*(float)iy/(float)field_y-1.f;
        float z = 2.f*(float)iz/(float)field_z-1.f;
        float f = 16.f*x*y*z + 4.f*x*x + 4.f*y*y + 4.f*z*z - 1.f;
//                  + 0.2f*sinf(33.f*x)*cosf(39.1f*y)*sinf(37.3f*z)
//                  + 0.1f*sinf(75.f*x)*cosf(75.1f*y)*sinf(71.3f*z);
/*
        - 0.6*sinf(25.1*x)*cosf(23.2*y)*sinf(21*z)
                  + 0.4*sinf(41.1*x*y)*cosf(47.2*y)*sinf(45*z)
                  - 0.2*sinf(111.1*x*y)*cosf(117.2*y)*sinf(115*z);
*/
//        f = sin(f);
        if( clamp ) {
            f = 255.f*f;
            if( f > 255 ) f = 255.f;
            if( f < 0 ) f = 0.f;
        }
        if( half_float ) {
            output[ ix + iy*field_row_pitch + iz*field_slice_pitch ] = __float2half_rn( f );
        }
        else {
            output[ ix + iy*field_row_pitch + iz*field_slice_pitch ] = f;
        }
    }
}


void
printHelp( const std::string& appname )
{
    //       --------------------------------------------------------------------------------
    cerr << "HPMC demo application that visualizes 1-16xyz-4x^2-4y^2-4z^2=iso."<<endl<<endl;
    cerr << "Usage: " << appname << " [options] xsize [ysize zsize] "<<endl<<endl;
    cerr << "where: xsize    The number of samples in the x-direction."<<endl;
    cerr << "       ysize    The number of samples in the y-direction."<<endl;
    cerr << "       zsize    The number of samples in the z-direction."<<endl;
    cerr << "Example usage:"<<endl;
    cerr << "    " << appname << " 64"<< endl;
    cerr << "    " << appname << " 64 128 64"<< endl;
    cerr << endl;
    cerr << "Options specific for this app:" << std::endl;
    cerr << "    --device <int>  Specify which CUDA device to use." << endl;
    cerr << "    --opengl-nonindexed  Direct OpenGL rendering, extracting 3 x tris vertices." << endl;
    cerr << "    --cuda-nonindexed    Use CUDA to extract 3 x tris vertices." << endl;
    cerr << "    --cuda-indexed       Use CUDA to extract vertices and triangle indices." << endl;
    cerr << "    --profile       Enable profiling of CUDA passes." << endl;
    cerr << "    --no-profile    Disable profiling of CUDA passes." << endl;
    cerr << endl;
    printOptions();
}


void
init( int argc, char** argv )
{
    int device = -1;
    for( int i=1; i<argc; ) {
        int eat = 0;
        std::string arg( argv[i] );
        if( (arg == "--device") && (i+1)<argc ) {
            device = atoi( argv[i+1] );
            eat = 2;
        }
        else if( (arg == "--opengl-nonindexed" ) ) {
            mode = OPENGL_NON_INDEXED;
            eat = 1;
        }
        else if( (arg == "--cuda-nonindexed" ) ) {
            mode = CUDA_NON_INDEXED;
            eat = 1;
        }
        else if( (arg == "--cuda-indexed" ) ) {
            mode = CUDA_INDEXED;
            eat = 1;
        }
        else if( (arg == "--profile" ) ) {
            profile = true;
            eat = 1;
        }
        else if( (arg == "--no-profile" ) ) {
            profile = false;
            eat = 1;
        }

        if( eat ) {
            argc = argc - eat;
            for( int k=i; k<argc; k++ ) {
                argv[k] = argv[k+eat];
            }
        }
        else {
            i++;
        }
    }
    if( argc > 1 ) {
        volume_size_x = atoi( argv[1] );
    }
    if( argc > 3 ) {
        volume_size_y = atoi( argv[2] );
        volume_size_z = atoi( argv[3] );
    }
    else {
        volume_size_y = volume_size_x;
        volume_size_z = volume_size_x;
    }
    if( volume_size_x < 4 ) {
        cerr << "Volume size x < 4" << endl;
        exit( EXIT_FAILURE );
    }
    if( volume_size_y < 4 ) {
        cerr << "Volume size y < 4" << endl;
        exit( EXIT_FAILURE );
    }
    if( volume_size_z < 4 ) {
        cerr << "Volume size z < 4" << endl;
        exit( EXIT_FAILURE );
    }


    int device_n = 0;
    cudaGetDeviceCount( &device_n );
    if( device_n == 0 ) {
        std::cerr << "Found no CUDA capable devices." << endl;
        exit( EXIT_FAILURE );
    }
    std::cerr << "Found " << device_n << " CUDA enabled devices:" << endl;
    int best_device = -1;
    int best_device_major = -1;
    int best_device_minor = -1;
    for(int i=0; i<device_n; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties( &prop, i );
        if( (prop.major > best_device_major) || ( (prop.major==best_device_major)&&(prop.minor>best_device_minor) ) ) {
            best_device = i;
            best_device_major = prop.major;
            best_device_minor = prop.minor;
        }
        std::cerr << "    device " << i
                  << ": compute cap=" << prop.major << "." << prop.minor
                  << endl;
    }
    if( device < 0 ) {
        std::cerr << "No CUDA device specified, using device " << best_device << endl;
        device = best_device;
    }
    if( (device < 0) || (device_n <= device) ) {
        std::cerr << "Illegal CUDA device " << device << endl;
        exit( EXIT_FAILURE );
    }
    //    cudaSetDevice( device );
    cudaGLSetGLDevice( device );
    switch( mode ) {
    case OPENGL_NON_INDEXED:
        std::cerr << "Mode is OpenGL non-indexed." << std::endl;
        break;
    case CUDA_NON_INDEXED:
        std::cerr << "Mode is CUDA non-indexed." << std::endl;
        break;
    case CUDA_INDEXED:
        std::cerr << "Mode is CUDA indexed." << std::endl;
        break;
    }


    // --- create field --------------------------------------------------------
    cudaMalloc( (void**)&field_data_dev, sizeof(unsigned char)*volume_size_x*volume_size_y*volume_size_z );
    bumpyCayley<unsigned char, true, false>
            <<< dim3( (volume_size_x+31)/32, volume_size_z*((volume_size_y+31)/32)), dim3(32,32) >>>
            ( field_data_dev,
              volume_size_x,
              volume_size_y,
              volume_size_z,
              volume_size_x,
              volume_size_x*volume_size_y,
              (volume_size_y+31)/32 );
    if( mode == OPENGL_NON_INDEXED ) {
        // we need the field as an OpenGL buffer
        // copy scalar field generated by cuda into host mem
        std::vector<unsigned char> foo( volume_size_x*volume_size_y*volume_size_z );
        cudaMemcpy( foo.data(), field_data_dev, foo.size(), cudaMemcpyDeviceToHost );
        cudaFree( field_data_dev );
        field_data_dev = NULL;

        // copy scalar field into an opengl buffer
        glGenBuffers( 1, &gl_field_buffer );
        glBindBuffer( GL_TEXTURE_BUFFER, gl_field_buffer );
        glBufferData( GL_TEXTURE_BUFFER,
                      foo.size(),
                      foo.data(),
                      GL_STATIC_DRAW );
        glBindBuffer( GL_TEXTURE_BUFFER, 0 );
    }

    // --- If CUDA generates geometry, create GL buffers to write into
    if( (mode == CUDA_NON_INDEXED) || (mode== CUDA_INDEXED ) ) {

        vertices_vbo_n = 100;
        glGenBuffers( 1, &vertices_vbo );
        glBindBuffer( GL_ARRAY_BUFFER, vertices_vbo );
        glBufferData( GL_ARRAY_BUFFER, 3*2*sizeof(GLfloat)*vertices_vbo_n, NULL, GL_DYNAMIC_COPY );
        glGenVertexArrays( 1, &vertices_vao );
        glBindVertexArray( vertices_vao );
        glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*6, (void*)(3*sizeof(GLfloat)) );
        glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*6, NULL );
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glBindVertexArray( 0);
        glBindBuffer( GL_ARRAY_BUFFER, 0 );

        CUDA_CHECKED( cudaGraphicsGLRegisterBuffer( &vertices_resource, vertices_vbo, cudaGraphicsRegisterFlagsWriteDiscard ) );
    }

    if( mode == CUDA_INDEXED ) {
        triangles_vbo_n = 100;

        glGenBuffers( 1, &triangles_vbo );
        glBindBuffer( GL_ARRAY_BUFFER, triangles_vbo );
        glBufferData( GL_ARRAY_BUFFER, 3*2*sizeof(GLfloat)*triangles_vbo_n, NULL, GL_DYNAMIC_COPY );
        glGenVertexArrays( 1, &triangles_vao );
        glBindVertexArray( triangles_vao );
        glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*6, (void*)(3*sizeof(GLfloat)) );
        glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat)*6, NULL );
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glBindVertexArray( 0);
        glBindBuffer( GL_ARRAY_BUFFER, 0 );

        CUDA_CHECKED( cudaGraphicsGLRegisterBuffer( &triangles_resource, triangles_vbo, cudaGraphicsRegisterFlagsWriteDiscard ) );
    }

    // --- Create shader program to render the VBO results ---------------------
    if( (mode == CUDA_NON_INDEXED) || (mode == CUDA_INDEXED) ) {
        GLuint vs = compileShader( resources::phong_vbo_vs_420, GL_VERTEX_SHADER );
        GLuint fs = compileShader( resources::phong_fs_130, GL_FRAGMENT_SHADER );
        vbo_render_prog = glCreateProgram();
        glAttachShader( vbo_render_prog, vs );
        glAttachShader( vbo_render_prog, fs );
        linkProgram( vbo_render_prog, "phong_vbo" );
        vbo_render_loc_pm = glGetUniformLocation( vbo_render_prog, "PM" );
        vbo_render_loc_nm = glGetUniformLocation( vbo_render_prog, "NM" );
        vbo_render_loc_col = glGetUniformLocation( vbo_render_prog, "color" );
    }

    // --- create CUHPMC objectes ----------------------------------------------
    switch( mode ) {
    case OPENGL_NON_INDEXED:
        // create CUHPMC objects
        gl_n_constants = new cuhpmc::Constants();
        gl_n_field = new cuhpmc::FieldGLBufferUChar( gl_n_constants,
                                                     gl_field_buffer,
                                                     volume_size_x,
                                                     volume_size_y,
                                                     volume_size_z );
        gl_n_isurf = new cuhpmc::IsoSurfaceGL( gl_n_field );
        gl_n_writer = new cuhpmc::GLWriter( gl_n_isurf );
        break;
    case CUDA_NON_INDEXED:
        // create CUHPMC objects
        cu_n_constants = new cuhpmc::Constants();
        cu_n_field = new cuhpmc::FieldGlobalMemUChar( cu_n_constants,
                                                      field_data_dev,
                                                      volume_size_x,
                                                      volume_size_y,
                                                      volume_size_z );
        cu_n_isurf = new cuhpmc::IsoSurfaceCUDA( cu_n_field );
        cu_n_writer = new cuhpmc::EmitterTriVtxCUDA( cu_n_isurf );
        break;

    case CUDA_INDEXED:
        // create CUHPMC objects
        cu_i_constants = new cuhpmc::Constants();
        cu_i_field = new cuhpmc::FieldGlobalMemUChar( cu_i_constants,
                                                 field_data_dev,
                                                 volume_size_x,
                                                 volume_size_y,
                                                 volume_size_z );

        cu_i_isurf = new cuhpmc::IsoSurfaceIndexed( cu_i_field );
        cu_i_writer = new cuhpmc::EmitterTriIdx( cu_i_isurf );
        break;
    }

    cudaStreamCreate( &stream );

    // Create profiling events if needed
    if( profile ) {
        cudaEventCreate( &pre_buildup );
        cudaEventCreate( &post_buildup );
        cudaEventCreate( &pre_write );
        cudaEventCreate( &post_write );
    }


    cudaError_t error = cudaGetLastError();
    if( error != cudaSuccess ) {
        std::cerr << "CUDA error: " << cudaGetErrorString( error ) << endl;
        exit( EXIT_FAILURE );
    }
}



void
render( float t,
        float dt,
        float fps,
        const GLfloat* P,
        const GLfloat* MV,
        const GLfloat* PM,
        const GLfloat *NM,
        const GLfloat* MV_inv )
{
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glEnable( GL_DEPTH_TEST );

    iso = 0.5f;//0.48f*(sin(t)+1.f)+0.01f;

#if 0
    static int foobar = 0;
    if( foobar >= 10 ) {
        exit( EXIT_SUCCESS );
    }
    foobar++;
#endif

    // build histopyramid
    if( profile ) {
        cudaEventRecord( pre_buildup, stream );
    }
    switch( mode ) {
    case OPENGL_NON_INDEXED:
        gl_n_isurf->build( iso, stream );
        break;
    case CUDA_NON_INDEXED:
        cu_n_isurf->build( iso, stream );
        break;

    case CUDA_INDEXED:
        cu_i_isurf->build( iso, stream );
        break;
    }
    if( profile ) {
        cudaEventRecord( post_buildup, stream );
    }

    // resize buffers if we run unless we do direct GL rendering
    uint vertices = 0;
    uint triangles = 0;
    switch( mode ) {
    case OPENGL_NON_INDEXED:
        break;

    case CUDA_NON_INDEXED:
        vertices = cu_n_isurf->vertices();
#if 0
        std::cerr << vertices << " vertices \n";
        std::cerr << triangles << " triangles\n";
        exit( EXIT_SUCCESS );
#endif
        if( vertices_vbo_n < vertices ) {
            CUDA_CHECKED( cudaStreamSynchronize( stream ) );
            CUDA_CHECKED( cudaGraphicsUnregisterResource( vertices_resource ) );

            vertices_vbo_n = 1.1f*vertices;
            GLsizei vbuf_size = 2*3*sizeof(GLfloat)*vertices_vbo_n;
            glBindBuffer( GL_ARRAY_BUFFER, vertices_vbo );
            glBufferData( GL_ARRAY_BUFFER, vbuf_size, NULL, GL_DYNAMIC_COPY );
            glBindBuffer( GL_ARRAY_BUFFER, 0);

            CUDA_CHECKED( cudaGraphicsGLRegisterBuffer( &vertices_resource, vertices_vbo, cudaGraphicsRegisterFlagsWriteDiscard ) );

            std::cerr << "Resized VBO to hold " << vertices << " vertices (" << vbuf_size << " bytes)\n";
        }
        break;

    case CUDA_INDEXED:
        vertices  = cu_i_isurf->vertices();
        triangles = cu_i_isurf->triangles();

        if( vertices_vbo_n < triangles ) {
            CUDA_CHECKED( cudaStreamSynchronize( stream ) );
            CUDA_CHECKED( cudaGraphicsUnregisterResource( vertices_resource ) );
            CUDA_CHECKED( cudaGraphicsUnregisterResource( triangles_resource ) );

            vertices_vbo_n = 1.1f*3*triangles;
            triangles_vbo_n = 1.1f*3*triangles;
            GLsizei vbuf_size = 2*3*sizeof(GLfloat)*vertices_vbo_n;
            GLsizei tbuf_size = 2*3*sizeof(GLfloat)*triangles_vbo_n;

            glBindBuffer( GL_ARRAY_BUFFER, vertices_vbo );
            glBufferData( GL_ARRAY_BUFFER, vbuf_size, NULL, GL_DYNAMIC_COPY );

            glBindBuffer( GL_ARRAY_BUFFER, triangles_vbo );
            glBufferData( GL_ARRAY_BUFFER, tbuf_size, NULL, GL_DYNAMIC_COPY );
            glBindBuffer( GL_ARRAY_BUFFER, 0 );

            CUDA_CHECKED( cudaGraphicsGLRegisterBuffer( &vertices_resource, vertices_vbo, cudaGraphicsRegisterFlagsWriteDiscard ) );
            CUDA_CHECKED( cudaGraphicsGLRegisterBuffer( &triangles_resource, triangles_vbo, cudaGraphicsRegisterFlagsWriteDiscard ) );

            std::cerr << "Resized VBO to hold " << vertices << " vertices (" << vbuf_size << " bytes)\n";
            std::cerr << "Resized VBO to hold " << triangles << " triangles (" << tbuf_size << " bytes)\n";
        }
        break;
    }

    if( profile ) {
        cudaEventRecord( pre_write );
    }

    // extract surfaces and render
    switch( mode ) {
    case OPENGL_NON_INDEXED:
        gl_n_writer->render( PM, NM, stream );
        break;
    case CUDA_NON_INDEXED:
    {
        float* vertices_d = NULL;
        size_t vertices_s = 0;
        CUDA_CHECKED( cudaGraphicsMapResources( 1, &vertices_resource, stream ) );
        CUDA_CHECKED( cudaGraphicsResourceGetMappedPointer( (void**)&vertices_d, &vertices_s, vertices_resource ) );

        cu_n_writer->writeInterleavedNormalPosition( vertices_d, vertices, stream );

        CUDA_CHECKED( cudaGraphicsUnmapResources( 1, &vertices_resource, stream ) );

        glUseProgram( vbo_render_prog );
        glUniformMatrix4fv( vbo_render_loc_pm, 1, GL_FALSE, PM );
        glUniformMatrix3fv( vbo_render_loc_nm, 1, GL_FALSE, NM );
        glUniform4f( vbo_render_loc_col, 0.8f, 0.8f, 1.f, 1.f );

        glBindVertexArray( vertices_vao );
        glDrawArrays( GL_TRIANGLES, 0, vertices );
        glBindVertexArray( 0 );
        glUseProgram( 0 );
    }
        break;

    case CUDA_INDEXED:
    {
        float* vertices_d = NULL;
        unsigned int* triangles_d = NULL;
        size_t vertices_s = 0;
        size_t triangles_s = 0;
        cudaGraphicsResource* resources[2] = {
            vertices_resource,
            triangles_resource
        };
        CUDA_CHECKED( cudaGraphicsMapResources( 2, resources, stream ) );
        CUDA_CHECKED( cudaGraphicsResourceGetMappedPointer( (void**)&vertices_d, &vertices_s, vertices_resource ) );
        CUDA_CHECKED( cudaGraphicsResourceGetMappedPointer( (void**)&triangles_d, &triangles_s, triangles_resource ) );

        cu_i_writer->writeVerticesInterleavedN3FV3F( vertices_d, vertices, stream );
        cu_i_writer->writeTriangleIndices( triangles_d, triangles, stream );

        CUDA_CHECKED( cudaGraphicsUnmapResources( 2, resources, stream ) );

        glUseProgram( vbo_render_prog );
        glUniformMatrix4fv( vbo_render_loc_pm, 1, GL_FALSE, PM );
        glUniformMatrix3fv( vbo_render_loc_nm, 1, GL_FALSE, NM );
        glUniform4f( vbo_render_loc_col, 0.8f, 0.8f, 1.f, 1.f );

        /*
        glBindVertexArray( triangles_vao );
        glDrawArrays( GL_TRIANGLES, 0, 3*triangles );
        */

        glPointSize( 1.f );
        glBindVertexArray( vertices_vao );

        glPolygonOffset( 1.f, 1.f );
        glEnable( GL_POLYGON_OFFSET_FILL );
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, triangles_vbo );
        glDrawElements( GL_TRIANGLES, 3*triangles, GL_UNSIGNED_INT, NULL );
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
        glDisable( GL_POLYGON_OFFSET_FILL );

        glUniform4f( vbo_render_loc_col, 1.f, 0.f, 0.f, 1.f );
        glDrawArrays( GL_POINTS, 0, vertices );

        glBindVertexArray( 0 );
        glUseProgram( 0 );



    }
        break;
    }

    if( profile ) {
        cudaEventRecord( post_write );
        cudaEventSynchronize( post_write );

        float ms;
        cudaEventElapsedTime( &ms, pre_buildup, post_buildup );
        buildup_ms += ms;
        cudaEventElapsedTime( &ms, pre_write, post_write );
        write_ms += ms;
        runs++;
    }

    glMatrixMode( GL_PROJECTION );
    glLoadMatrixf( P );
    glMatrixMode( GL_MODELVIEW );
    glLoadMatrixf( MV );
    glBegin( GL_LINES );
    glColor3f( 0.f, 0.f, 0.f ); glVertex3f( 0.f, 0.f, 0.f );
    glColor3f( 1.f, 0.f, 0.f ); glVertex3f( 1.f, 0.f, 0.f );

    glColor3f( 1.f, 0.f, 0.f ); glVertex3f( 1.f, 0.f, 0.f );
    glColor3f( 1.f, 1.f, 0.f ); glVertex3f( 1.f, 1.f, 0.f );

    glColor3f( 1.f, 1.f, 0.f ); glVertex3f( 1.f, 1.f, 0.f );
    glColor3f( 0.f, 1.f, 0.f ); glVertex3f( 0.f, 1.f, 0.f );

    glColor3f( 0.f, 1.f, 0.f ); glVertex3f( 0.f, 1.f, 0.f );
    glColor3f( 0.f, 0.f, 0.f ); glVertex3f( 0.f, 0.f, 0.f );

    glColor3f( 0.f, 0.f, 1.f ); glVertex3f( 0.f, 0.f, 1.f );
    glColor3f( 1.f, 0.f, 1.f ); glVertex3f( 1.f, 0.f, 1.f );

    glColor3f( 1.f, 0.f, 1.f ); glVertex3f( 1.f, 0.f, 1.f );
    glColor3f( 1.f, 1.f, 1.f ); glVertex3f( 1.f, 1.f, 1.f );

    glColor3f( 1.f, 1.f, 1.f ); glVertex3f( 1.f, 1.f, 1.f );
    glColor3f( 0.f, 1.f, 1.f ); glVertex3f( 0.f, 1.f, 1.f );

    glColor3f( 0.f, 1.f, 1.f ); glVertex3f( 0.f, 1.f, 1.f );
    glColor3f( 0.f, 0.f, 1.f ); glVertex3f( 0.f, 0.f, 1.f );


    glColor3f( 0.f, 0.f, 0.f ); glVertex3f( 0.f, 0.f, 0.f );
    glColor3f( 0.f, 0.f, 1.f ); glVertex3f( 0.f, 0.f, 1.f );

    glColor3f( 1.f, 0.f, 0.f ); glVertex3f( 1.f, 0.f, 0.f );
    glColor3f( 1.f, 0.f, 1.f ); glVertex3f( 1.f, 0.f, 1.f );

    glColor3f( 1.f, 1.f, 0.f ); glVertex3f( 1.f, 1.f, 0.f );
    glColor3f( 1.f, 1.f, 1.f ); glVertex3f( 1.f, 1.f, 1.f );

    glColor3f( 0.f, 1.f, 0.f ); glVertex3f( 0.f, 1.f, 0.f );
    glColor3f( 0.f, 1.f, 1.f ); glVertex3f( 0.f, 1.f, 1.f );

    glEnd();


    cudaError_t error = cudaGetLastError();
    if( error != cudaSuccess ) {
        std::cerr << "CUDA error: " << cudaGetErrorString( error ) << endl;
        exit( EXIT_FAILURE );
    }
}

const std::string
infoString( float fps )
{
    float avg_buildup = buildup_ms/runs;
    float avg_write = write_ms/runs;
    buildup_ms = 0.f;
    write_ms = 0.f;
    runs = 0;

    std::stringstream o;
    o << std::setprecision(5) << fps << " fps, "
      << volume_size_x << 'x'
      << volume_size_y << 'x'
      << volume_size_z << " samples, "
      << (int)( ((volume_size_x-1)*(volume_size_y-1)*(volume_size_z-1)*fps)/1e6 )
      << " MVPS, ";
    if( profile ) {
        o << "build=" << avg_buildup << "ms, "
          << "write=" << avg_write << "ms, ";
    }
    switch( mode ) {
    case OPENGL_NON_INDEXED:
        o << "n_vtx=" << gl_n_isurf->vertices() << ", "
          << "n_tri=" << (gl_n_isurf->vertices()/3u) << ", ";
        break;
    case CUDA_NON_INDEXED:
        o << "n_vtx=" << cu_n_isurf->vertices() << ", "
          << "n_tri=" << (cu_n_isurf->vertices()/3u) << ", ";
        break;

    case CUDA_INDEXED:
        o << "n_vtx=" << cu_i_isurf->vertices() << ", "
          << "n_tri=" << cu_i_isurf->triangles() << ", ";
        break;
    }
    o << "iso=" << iso
      << (wireframe?"[wireframe]":"");
    return o.str();
}
