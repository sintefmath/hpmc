
SET ( txt2cpp_source 
"#include <string>\n"
"#include <vector>\n"
"#include <iostream>\n"
"#include <fstream>\n"
"#include <iomanip>\n"
"\n"
"\n"
"using std::endl\;\n"
"using std::string\;\n"
"using std::vector\;\n"
"using std::ifstream\;\n"
"using std::ofstream\;\n"
"\n"
"int\n"
"main( int argc, char** argv)\n"
"{\n"
"    if( argc  != 4 ) {\n"
"        std::cerr << \"usage: \" << argv[0] << \" infile symbolname outfile\" << std::endl\;\n"
"        return 1\;\n"
"    }\n"
"\n"
"    string infilename = argv[1]\;\n"
"    string symbolname = argv[2]\;\n"
"    string outfilename = argv[3]\;\n"
"\n"
"    size_t first=0\;\n"
"    size_t curr\;\n"
"    vector<string> symbolpath\;\n"
"    do {\n"
"        curr = symbolname.find( \"::\", first )\;\n"
"        if( curr != string::npos ) {\n"
"            symbolpath.push_back( symbolname.substr( first, curr-first ) )\;\n"
"        }\n"
"        else {\n"
"            symbolpath.push_back( symbolname.substr( first, curr ) )\;\n"
"        }\n"
"        first = curr+2\;\n"
"    } while( curr != string::npos )\;\n"
"\n"
"\n"
"\n"
"    ifstream infile( infilename.c_str(), std::ios::in )\;\n"
"    string src = string( std::istreambuf_iterator<char>(infile),\n"
"                         std::istreambuf_iterator<char>() )\;\n"
"\n"
"\n"
"    ofstream outfile( outfilename.c_str(), std::ios::out | std::ios::trunc )\;\n"
"\n"
"    outfile << \"#include <string>\" << endl\;\n"
"    for(size_t i=0\; i<symbolpath.size()-1u\; i++) {\n"
"        outfile << \"namespace \" << symbolpath[i] << \" {\" << endl\;\n"
"    }\n"
"    outfile << \"std::string \" << symbolpath.back() << \" = \" << endl\;\n"
"    outfile.setf( std::ios::oct, std::ios::basefield )\;\n"
"    outfile.setf( std::ios::showbase )\;\n"
"    if( src.length() == 0 ) {\n"
"        outfile << \"\\\"\\\"\;\" << endl\;\n"
"    }\n"
"    else {\n"
"        for(size_t i=0\; i<src.length()\; i++ ) {\n"
"            if( i==0) {\n"
"                outfile << \"\\\"\"\;\n"
"            }\n"
"            if( src[i] == '\\r' ) {\n"
"            }\n"
"            else if( src[i] == '\\n') {\n"
"                outfile << \"\\\\n\\\"\" << endl << \"\\\"\"\;\n"
"            }\n"
"            else if( src[i] == '\"' ) {\n"
"                outfile << \"\\\\\\\"\"\;\n"
"            }\n"
"            else {\n"
"                outfile << src[i]\;\n"
"            }\n"
"        }\n"
"        outfile << \"\\\"\;\" << endl\;\n"
"    }\n"
"    for(size_t i=0\; i<symbolpath.size()-1u\; i++) {\n"
"        outfile << \"} // of namespace \" << symbolpath[i] << endl\;\n"
"    }\n"
"\n"
"    return 0\;\n"
"}\n"
""\; )

FILE(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/buildtools" )
FILE(WRITE "${CMAKE_BINARY_DIR}/buildtools/txt2cpp.cpp" ${txt2cpp_source})

set_source_files_properties( "${CMAKE_BINARY_DIR}/buildtools/txt2cpp.cpp" PROPERTIES GENERATED 1 )
ADD_EXECUTABLE(app_txt2cpp "${CMAKE_BINARY_DIR}/buildtools/txt2cpp.cpp")
SET_TARGET_PROPERTIES(app_txt2cpp PROPERTIES OUTPUT_NAME "txt2cpp")

MACRO( ADD_TEXT_FILE targetName symbol input )
    GET_FILENAME_COMPONENT(infile_we ${input} NAME_WE)
    GET_FILENAME_COMPONENT(input_filename ${input} ABSOLUTE)
    SET( result_filename "${CMAKE_CURRENT_BINARY_DIR}/${infile_we}.cpp")
    GET_TARGET_PROPERTY(TXT2cppP_EXE app_txt2cpp LOCATION)
    ADD_CUSTOM_COMMAND(
      OUTPUT ${result_filename}
      COMMAND ${TXT2cppP_EXE} ${input_filename} ${symbol} ${result_filename}
	  DEPENDS app_txt2cpp ${input_filename} 
	  )
	STRING(REGEX REPLACE "::" "__" target_name ${symbol}  )
#	message("target ${targetName}")
	add_library(${targetName} ${result_filename})
	add_dependencies(${targetName}  app_txt2cpp )
#	ADD_CUSTOM_TARGET(${targetName} DEPENDS ${result_filename})
ENDMACRO( ADD_TEXT_FILE )