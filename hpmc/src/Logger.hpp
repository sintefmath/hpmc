#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <string>
class HPMCConstants;

class Logger
{
public:
    Logger( const HPMCConstants* constants, const std::string& where, bool force_check = false );

    ~Logger();

    bool
    doDebug() const { return true; }

    void
    debugMessage( const std::string& msg );

    void
    warningMessage( const std::string& msg );

    void
    errorMessage( const std::string& msg );

    void
    setObjectLabel( GLenum identifier, GLuint name, const std::string label );

    const std::string&
    where() const { return m_where; }

protected:
    const HPMCConstants*    m_constants;
    const std::string       m_where;
    const bool              m_force_check;

    const std::string
    glErrorString( GLenum error ) const;

};

#endif // LOGGER_HPP
