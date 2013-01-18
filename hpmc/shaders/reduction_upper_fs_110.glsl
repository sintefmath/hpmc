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
