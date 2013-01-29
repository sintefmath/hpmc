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
uniform sampler2D  HPMC_histopyramid;
uniform int        HPMC_src_level;
void
main()
{
    ivec2 tp = 2*ivec2( gl_FragCoord.xy );
    vec4 sums = vec4(
        dot( vec4(1.0),  floor( texelFetch( HPMC_histopyramid, tp + ivec2(0,0), HPMC_src_level ) ) ),
        dot( vec4(1.0),  floor( texelFetch( HPMC_histopyramid, tp + ivec2(1,0), HPMC_src_level ) ) ),
        dot( vec4(1.0),  floor( texelFetch( HPMC_histopyramid, tp + ivec2(0,1), HPMC_src_level ) ) ),
        dot( vec4(1.0),  floor( texelFetch( HPMC_histopyramid, tp + ivec2(1,1), HPMC_src_level ) ) )
    );
    gl_FragColor = sums;
}
