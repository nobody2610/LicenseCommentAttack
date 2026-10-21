// SPDX-License-Identifier: LGPL-2.1-or-later
/****************************************************************************
 *                                                                          *
 *   This file is part of FreeCAD.                                          *
 *                                                                          *
 *   FreeCAD is free software: you can redistribute it and/or modify it     *
 *   under the terms of the GNU Lesser General Public License as            *
 *   published by the Free Software Foundation, either version 2.1 of the   *
 *   License, or (at your option) any later version.                        *
 *                                                                          *
 *   FreeCAD is distributed in the hope that it will be useful, but         *
 *   WITHOUT ANY WARRANTY; without even the implied warranty of             *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU       *
 *   Lesser General Public License for more details.                        *
 *                                                                          *
 *   You should have received a copy of the GNU Lesser General Public       *
 *   License along with FreeCAD. If not, see                                *
 *   <https://www.gnu.org/licenses/>.                                       *
 *                                                                          *
 ***************************************************************************/

#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <FCGlobal.h>

#include "Exception.h"

namespace Base
{

constexpr auto unitSymbols = std::to_array<std::string_view>(
    {"mm", "kg", "s", "A", "K", "mol", "cd", "deg"}
);

constexpr auto unitNumExponents {unitSymbols.size()};
using UnitExponents = std::array<int8_t, unitNumExponents>;

constexpr auto unitExponentLimit {8};

class BaseExport Unit final
{
public:
    Unit() = default;

    explicit constexpr Unit(const UnitExponents exps, const std::string_view name = "")
        : _exps {exps}
        , _name {name}
    {
        checkRange();
    }

    /// helper constructor to ease Unit construction from Python
    explicit Unit(
        const int length,
        const int mass = 0,
        const int time = 0,
        const int electricCurrent = 0,
        const int thermodynamicTemperature = 0,
        const int amountOfSubstance = 0,
        const int luminousIntensity = 0,
        const int angle = 0
    );

    bool operator==(const Unit&) const;
    bool operator!=(const Unit& that) const;
    Unit& operator*=(const Unit& that);
    Unit& operator/=(const Unit& that);
    Unit operator*(const Unit&) const;
    Unit operator/(const Unit&) const;

    [[nodiscard]] Unit pow(const double exp) const;
    [[nodiscard]] Unit root(const uint8_t num) const;

    [[nodiscard]] UnitExponents exponents() const;
    [[nodiscard]] int length() const;

    [[nodiscard]] std::string getString() const;       // E.g. kg, mm^2, mm*kg/s^2
    [[nodiscard]] std::string getTypeString() const;   // E.g. "Area", "Length", "Pressure"
    [[nodiscard]] std::string representation() const;  // E.g. "Unit: mm (1,0,0,0,0,0,0,0) [Length]"

    Unit sqrt() const
    {
        return root(2);
    }
    Unit cbrt() const
    {
        return root(3);
    }

private:
    UnitExponents _exps {};
    std::string_view _name;

    constexpr void checkRange()
    {
        for (const auto exp : _exps) {
            if (exp >= unitExponentLimit) {
                throw OverflowError("Unit exponent overflow");
            }
            if (exp < -unitExponentLimit) {
                throw UnderflowError("Unit exponent underflow");
            }
        }
    }

    /** Returns posIndexes, negIndexes*/
    std::pair<std::vector<size_t>, std::vector<size_t>> nonZeroValsIndexes() const;

public:
    static const Unit Acceleration;
    static const Unit AmountOfSubstance