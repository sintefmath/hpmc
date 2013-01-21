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
// evaluates the scalar field
uniform float shape[12];

float
HPMC_fetch( vec3 p )
{
    p -= 0.5;
    p *= 2.2;
    return -( shape[0]*p.x*p.x*p.x*p.x*p.x +
              shape[1]*p.x*p.x*p.x*p.x +
              shape[2]*p.y*p.y*p.y*p.y +
              shape[3]*p.z*p.z*p.z*p.z +
              shape[4]*p.x*p.x*p.y*p.y +
              shape[5]*p.x*p.x*p.z*p.z +
              shape[6]*p.y*p.y*p.z*p.z +
              shape[7]*p.x*p.y*p.z +
              shape[8]*p.x*p.x +
              shape[9]*p.y*p.y +
              shape[10]*p.z*p.z +
              shape[11] );
}

// evaluates the gradient as well as the scalar field
vec4
HPMC_fetchGrad( vec3 p )
{
    p -= 0.5;
    p *= 2.2;
    return -vec4( 5.0*shape[0]*p.x*p.x*p.x*p.x +
                  4.0*shape[1]*p.x*p.x*p.x +
                  2.0*shape[4]*p.x*p.y*p.y +
                  2.0*shape[5]*p.x*p.z*p.z +
                      shape[7]*p.y*p.z +
                  2.0*shape[8]*p.x,

                  4.0*shape[2]*p.y*p.y*p.y +
                  2.0*shape[4]*p.x*p.x*p.y +
                  2.0*shape[6]*p.y*p.z*p.z +
                      shape[7]*p.x*p.z +
                  2.0*shape[9]*p.y,

                  4.0*shape[3]*p.z*p.z*p.z +
                  2.0*shape[5]*p.x*p.x*p.z +
                  2.0*shape[6]*p.y*p.z*p.z +
                      shape[7]*p.x*p.y +
                  2.0*shape[10]*p.z,

                  shape[0]*p.x*p.x*p.x*p.x*p.x +
                  shape[1]*p.x*p.x*p.x*p.x +
                  shape[2]*p.y*p.y*p.y*p.y +
                  shape[3]*p.z*p.z*p.z*p.z +
                  shape[4]*p.x*p.x*p.y*p.y +
                  shape[5]*p.x*p.x*p.z*p.z +
                  shape[6]*p.y*p.y*p.z*p.z +
                  shape[7]*p.x*p.y*p.z +
                  shape[8]*p.x*p.x +
                  shape[9]*p.y*p.y +
                  shape[10]*p.z*p.z +
                  shape[11] );
};
