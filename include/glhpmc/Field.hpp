#pragma once
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
#include <string>
#include <iostream>
#include <glhpmc/glhpmc.hpp>

namespace glhpmc {


/** Abstract base class for field.
 *
 * This class defines the interface used to tell HPMC how it should fetch
 * samples from a scalar field.
 *
 * HPMC provides some concrete classes, or one the developer may derive from
 * this class.
 *
 */
class Field
{
public:
    /** Abstract base class for state required to use this field with a shader.
     *
     * For each instance that this field will be used with a shader program,
     * this class may provide a concrete class inheriting from this class. It
     * can be used to store per-program specific data, like uniform locations.
     */
    struct ProgramContext
    {
        virtual
        ~ProgramContext() {}
    };

    /** Destructor. */
    virtual
    ~Field() {}

    /** Create a program context for a new shader program.
     *
     * The program is bound before this function is invoked, hence it is safe
     * to directly invoke glUniform etc.
     *
     * \param program  The OpenGL shader program which is used.
     * \return A derivative of ProgramContext, or NULL if no context is needed.
     *
     * \note Not required to be implemented by concrete implementation.
     */
    virtual
    ProgramContext*
    createContext( GLuint program ) const { return NULL; }

    /** Returns true if the field provides gradient vectors.
     *
     * If the field provides gradient vectors, HPMC will use these to determine
     * normal vectors for its output. Otherwise, the gradients will be
     * approximated by forward differences, which involves 6 extra scalar field
     * fetches per output point, and thus directly providing the gradient vector
     * might give a faster implementation.
     *
     * \returns True if field has gradient vectors, false otherwise.
     *
     * \note Must be implemented by concrete implementation.
     */
    virtual
    bool
    gradients() const = 0;

    /** Returns shader source to fetch a scalar field sample.
     *
     * The shader source should provide an implementation of a function with the
     * following signature:
     * \code
     * float
     * HPMC_fetch( vec3 p );
     * \endcode
     * where p is a sample position in [0,1]^3 and it returns the scalar field
     * at that position.
     *
     * \note Must be implement by concrete implementations.
     * \note This source will be used in shaders with a version string matching
     * the HPMC target version, and must adhere to the specification of that
     * version.
     */
    virtual
    const std::string
    fetcherFieldSource( ) const = 0;

    /** Returns shader source to fetch a scalar field and gradient sample.
     *
     * The shader source should provide an implementation of a function with the
     * following signature
     * \code
     * vec4
     * HPMC_fetchGrad( vec3 p );
     * \endcode
     * where p is a sample position in [0,1]^3 and the return value is a vec4
     * where the RGB-channels contains the gradient vector and the A-channel
     * contains the scalar field at that position.
     *
     * \note If gradients() returns true, must be implemented by concrete
     * implementation. Otherwise, it will never be used and the base class
     * implementation suffices.
     * \note This source will be used in shaders with a version string matching
     * the HPMC target version, and must adhere to the specification of that
     * version.
     */
    virtual
    const std::string
    fetcherFieldAndGradientSource( ) const { return ""; }

    /** Invoked each time just before a shader program that uses this field is about to be used.
     *
     * Typically sets uniform values and binds textures.
     *
     * \param program_context  The object provided by createContext, or NULL if
     *                         createContext wasn't overridden.
     * \returns True on success.
     *
     * \note Not required to be implemented by concrete implementation.
     */
    virtual
    bool
    bind( ProgramContext* program_context ) const { return true; }

    /** Invoked each time just after a shader program that uses this field has been used.
     *
     * Should clean up GL state set by bind, typically bind the texture samplers
     * to zero.
     *
     * \param program_context  The object provided by createContext, or NULL if
     *                         createContext wasn't overridden.
     * \returns True on success.
     *
     * \note Not required to be implemented by concrete implementation.
     */
    virtual
    bool
    unbind( ProgramContext* program_context ) const { return true; }

    /** Return the number of logical samples along the X-axis.
     *
     * The field will be sampled with positions (i+0.5)/samplesX() for integer i
     * in [0,samplesX()-1].
     *
     */
    unsigned int
    samplesX() const { return m_samples_x; }

    /** Return the number of logical samples along the X-axis.
     *
     * The field will be sampled with positions (i+0.5)/samplesY() for integer i
     * in [0,samplesY()-1].
     *
     */
    unsigned int
    samplesY() const { return m_samples_y; }

    /** Return the number of logical samples along the X-axis.
     *
     * The field will be sampled with positions (i+0.5)/samplesZ() for integer i
     * in [0,samplesZ()-1].
     *
     */
    unsigned int
    samplesZ() const { return m_samples_z; }


protected:
    /** Protected constructor for abstract class.
     *
     * Concrete implementations should provide public constructors that invoke
     * this constructor.
     */
    Field( HPMCConstants* constants,
           unsigned int samples_x,
           unsigned int samples_y,
           unsigned int samples_z )
        : m_constants( constants ),
          m_samples_x( samples_x ),
          m_samples_y( samples_y ),
          m_samples_z( samples_z )
    {
    }

    /** Provides the constants used by HPMC (which contain GLSL version). */
    HPMCConstants*
    constants() { return m_constants; }

    /** Provides the constants used by HPMC (which contain GLSL version). */
    const HPMCConstants*
    constants() const { return m_constants; }

private:
    HPMCConstants*  m_constants;
    unsigned int    m_samples_x;
    unsigned int    m_samples_y;
    unsigned int    m_samples_z;

};




} // of namespace glhpmc
