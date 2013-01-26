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
uniform vec2       HPMC_delta;
void
main()
{
    vec4 sums = vec4(
        dot( vec4(1.0), texture2D( HPMC_histopyramid, gl_TexCoord[0].xy+HPMC_delta.xx ) ),
        dot( vec4(1.0), texture2D( HPMC_histopyramid, gl_TexCoord[0].xy+HPMC_delta.yx ) ),
        dot( vec4(1.0), texture2D( HPMC_histopyramid, gl_TexCoord[0].xy+HPMC_delta.xy ) ),
        dot( vec4(1.0), texture2D( HPMC_histopyramid, gl_TexCoord[0].xy+HPMC_delta.yy ) )
    );
    gl_FragColor = sums;
}
