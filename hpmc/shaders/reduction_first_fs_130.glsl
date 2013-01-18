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
