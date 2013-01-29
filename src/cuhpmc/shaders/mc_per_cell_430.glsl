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

layout(binding=3)   uniform usamplerBuffer  cases_tex;

void
mc_per_cell( out uvec3 i0, out uint mc_case, in uint hp_pos )
{
    // Calc 3D grid pos from linear input stream pos.
    uint c_lix = hp_pos / 800;
    uint t_lix = hp_pos % 800;
    uvec3 ci = uvec3( 31u*( c_lix % CUHPMC_CHUNKS_X ),
                       5u*( (c_lix/CUHPMC_CHUNKS_X) % CUHPMC_CHUNKS_Y ),
                       5u*( (c_lix/CUHPMC_CHUNKS_X) / CUHPMC_CHUNKS_Y ) );
    i0 = uvec3( ci.x + ((t_lix / 5u)%32u),
                ci.y + ((t_lix / 5u)/32u),
                ci.z + ( t_lix%5u ) );

    mc_case = texelFetch( cases_tex, int(hp_pos) ).r;

}
