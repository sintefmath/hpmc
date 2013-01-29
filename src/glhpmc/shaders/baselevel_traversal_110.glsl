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
//          Scale tp from tile parameterization to scalar field parameterization
//          Now we have found the MC cell, next find which edge that this vertex lies on
// Output sample positions, if needed.

void
HPMC_baseLevelExtract( out vec3 a,
                       out vec3 b,
                       out vec3 p,
                       out vec3 n,
                       in  float key )
{
    vec2 base_texcoord;
    float key_ix;
    float val;
    HPMC_traverseDown( base_texcoord, val, key_ix, key );
    vec2 foo = vec2( HPMC_TILES_X_F,HPMC_TILES_Y_F)*base_texcoord;
    vec2 tp = vec2( (2.0*HPMC_TILE_SIZE_X_F)/HPMC_FUNC_X_F,
                    (2.0*HPMC_TILE_SIZE_Y_F)/HPMC_FUNC_Y_F ) * fract(foo);
    float slice = dot( vec2(1.0,HPMC_TILES_X_F), floor(foo));
    vec4 edge = texture2D( HPMC_edge_table, vec2((1.0/16.0)*(key_ix+0.5), val ) );
#ifdef FIELD_BINARY
    n = 2.0*fract(edge.xyz)-vec3(1.0);
    edge = floor(edge);
#endif // FIELD_BINARY
    vec3 shift = edge.xyz;
    vec3 axis = vec3( equal(vec3(0.0, 1.0, 2.0), vec3(edge.w)) );
    //          Calculate sample positions of the two end-points of the edge.
    vec3 pa = vec3(tp, slice)
            + vec3(1.0/HPMC_FUNC_X_F, 1.0/HPMC_FUNC_Y_F, 1.0)*shift;
    vec3 pb = pa
            + vec3(1.0/HPMC_FUNC_X_F, 1.0/HPMC_FUNC_Y_F, 1.0)*axis;
    a = vec3(pa.x, pa.y, (pa.z+0.5)*(1.0/float(HPMC_FUNC_Z)) );
    b = vec3(pb.x, pb.y, (pb.z+0.5)*(1.0/float(HPMC_FUNC_Z)) );
#ifdef FIELD_BINARY
    p = 0.5*(pa+pb);
#else
    vec3 pa_ = pa;
    pa_.z = (pa.z+0.5)*(1.0/HPMC_FUNC_Z_F);
    vec3 pb_ = pb;
    pb_.z = (pb.z+0.5)*(1.0/HPMC_FUNC_Z_F);
#if FIELD_HAS_GRADIENT
    //          If we have gradient info, sample pa and pb.
    vec4 fa = HPMC_fetchGrad( pa_ );
    vec3 na = fa.xyz;
    float va = fa.w;
    vec4 fb = HPMC_fetchGrad( pb_ );
    vec3 nb = fb.xyz;
    float vb = fb.w;
    //          Solve linear equation to approximate point that edge pierces iso-surface.
    float t = (va-HPMC_threshold)/(va-vb);
#else
    //          If we don't have gradient info, we approximate the gradient using forward
    //          differences. The sample at pb is one of the forward samples at pa, so we
    //          save one texture lookup.
    float va = HPMC_fetch( pa_ );
    vec3 na = vec3( HPMC_fetch( pa_ + vec3( 1.0/HPMC_FUNC_X_F, 0.0, 0.0 ) ),
                    HPMC_fetch( pa_ + vec3( 0.0, 1.0/HPMC_FUNC_Y_F, 0.0 ) ),
                    HPMC_fetch( pa_ + vec3( 0.0, 0.0, 1.0/HPMC_FUNC_Z_F ) ) );
    vec3 nb = vec3( HPMC_fetch( pb_ + vec3( 1.0/HPMC_FUNC_X_F, 0.0, 0.0 ) ),
                    HPMC_fetch( pb_ + vec3( 0.0, 1.0/HPMC_FUNC_Y_F, 0.0 ) ),
                    HPMC_fetch( pb_ + vec3( 0.0, 0.0, 1.0/HPMC_FUNC_Z_F ) ) );
    //          Solve linear equation to approximate point that edge pierces iso-surface.
    float t = (va-HPMC_threshold)/(va-dot(na,axis));
#endif // FIELD_HAS_GRADIENT
    p = mix(pa, pb, t );
    n = vec3(HPMC_threshold)-mix(na, nb,t);
#endif // FIELD_BINARY
    //          p.xy is in normalized texture coordinates, but z is an integer slice number.
    //          First, remove texel center offset
    p.xy -= vec2(0.5/HPMC_FUNC_X_F, 0.5/HPMC_FUNC_Y_F );
    //          And rescale such that domain fits extent.
    p *= vec3( HPMC_GRID_EXT_X_F * HPMC_FUNC_X_F/(HPMC_CELLS_X_F-0.0),
               HPMC_GRID_EXT_Y_F * HPMC_FUNC_Y_F/(HPMC_CELLS_Y_F-0.0),
               HPMC_GRID_EXT_Z_F * 1.0/(HPMC_CELLS_Z_F) );
    n *= vec3( HPMC_GRID_EXT_X_F/HPMC_CELLS_X_F,
               HPMC_GRID_EXT_Y_F/HPMC_CELLS_Y_F,
               HPMC_GRID_EXT_Z_F/HPMC_CELLS_Z_F );
}
