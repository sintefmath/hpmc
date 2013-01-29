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

out VG {
    uvec3 i0;
    uint mc_case;
    uint remainder;
} out_v;

void
hp5_downtraverse( out uint pos, out uint key_remainder, in uint key );

void
mc_per_cell( out uvec3 i0, out uint mc_case, in uint hp_pos );

void
main()
{
    uint t_pos;
    uint t_remainder;
    hp5_downtraverse( t_pos, t_remainder, gl_VertexID );
    out_v.remainder = 3*t_remainder;

    uvec3 t_i0;
    uint t_mc_case;
    mc_per_cell( t_i0, t_mc_case, t_pos );
    out_v.i0 = t_i0;
    out_v.mc_case = t_mc_case;
}
