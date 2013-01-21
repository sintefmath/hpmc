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
float
HPMC_fetch( vec3 p )
{
    p *= 2.0;
    p -= 1.0;
    return 1.0 - 16.0*p.x*p.y*p.z - 4.0*p.x*p.x - 4.0*p.y*p.y - 4.0*p.z*p.z;
}

// evaluates the gradient as well as the scalar field
vec4
HPMC_fetchGrad( vec3 p )
{
    p *= 2.0;
    p -= 1.0;
    return vec4( -16.0*p.y*p.z - 8.0*p.x,
                 -16.0*p.x*p.z - 8.0*p.y,
                 -16.0*p.x*p.y - 8.0*p.z,
                 1.0 - 16.0*p.x*p.y*p.z - 4.0*p.x*p.x - 4.0*p.y*p.y - 4.0*p.z*p.z );
}
