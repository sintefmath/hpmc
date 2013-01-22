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
#include <cuhpmc/IsoSurface.hpp>


using std::cerr;
using std::endl;

int                             volume_size_x       = 8;
int                             volume_size_y       = 8;
int                             volume_size_z       = 8;
float                           iso                 = 0.5f;

cuhpmc::Constants*              constants           = NULL;
unsigned char*                  field_data_dev      = NULL;
cuhpmc::AbstractField*          field               = NULL;
cuhpmc::IsoSurface*             iso_surface         = NULL;

GLuint                          surface_vbo         = 0;
GLsizei                         surface_vbo_n       = 0;
cudaGraphicsResource*           surface_resource    = NULL;

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
    cerr << "HPMC demo application that visualizes 1-16xyz-4x^2-4y^2-4z^2=iso."<<endl<<endl;
    cerr << "Usage: " << appname << " [options] xsize [ysize zsize] "<<endl<<endl;
    cerr << "where: xsize    The number of samples in the x-direction."<<endl;
    cerr << "       ysize    The number of samples in the y-direction."<<endl;
    cerr << "       zsize    The number of samples in the z-direction."<<endl;
    cerr << "Example usage:"<<endl;
    cerr << "    " << appname << " 64"<< endl;
    cerr << "    " << appname << " 64 128 64"<< endl;
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
    cudaSetDevice( device );

    // create field

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

    std::vector<unsigned char> moo( volume_size_x*volume_size_y*volume_size_z );
    cudaMemcpy( moo.data(), field_data_dev, moo.size(), cudaMemcpyDeviceToHost );

    /*
    for(uint k=0; k<volume_size_z; k++ ) {
        for(uint j=0; j<volume_size_z; j++ ) {
            for(uint i=0; i<volume_size_x; i++ ) {
                if( moo[ (k*volume_size_y + j)*volume_size_x + i ] < 0.5 ) {
                    std::cerr << "-";
                }
                else {
                    std::cerr << "*";
                }
            }
            std::cerr << endl;
        }
        std::cerr << endl;
    }
*/

    // Generate OpenGL VBO that we lend to CUDA
    surface_vbo_n = 1000;
    glGenBuffers( 1, &surface_vbo );
    glBindBuffer( GL_ARRAY_BUFFER, surface_vbo );
    glBufferData( GL_ARRAY_BUFFER,
                  3*sizeof(GLfloat)*surface_vbo_n,
                  NULL,
                  GL_DYNAMIC_COPY );
    glBindBuffer( GL_ARRAY_BUFFER, surface_vbo );

    constants = new cuhpmc::Constants();
    field = new cuhpmc::FieldGlobalMemUChar( constants,
                                             field_data_dev,
                                             volume_size_x,
                                             volume_size_y,
                                             volume_size_z );
    iso_surface = new cuhpmc::IsoSurface( field );


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

    iso_surface->build( iso );


    cudaError_t error = cudaGetLastError();
    if( error != cudaSuccess ) {
        std::cerr << "CUDA error: " << cudaGetErrorString( error ) << endl;
        exit( EXIT_FAILURE );
    }
}

const std::string
infoString( float fps )
{
    std::stringstream o;
    o << std::setprecision(5) << fps << " fps, "
      << volume_size_x << 'x'
      << volume_size_y << 'x'
      << volume_size_z << " samples, "
      << (int)( ((volume_size_x-1)*(volume_size_y-1)*(volume_size_z-1)*fps)/1e6 )
      << " MVPS, "
      << iso_surface->vertices()
      << " vertices, iso=" << iso
      << (wireframe?"[wireframe]":"");
    return o.str();
}
