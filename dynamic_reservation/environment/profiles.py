#     Copyright (C) 2024 Lisa Maile
#
#     lisa.maile@fau.de
#
#     This file is part of the DYnamic Reliable rEal-time Communication in Tsn (DYRECTsn) framework.
#
#     DYRECTsn is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     DYRECTsn is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with DYRECTsn.  If not, see <http://www.gnu.org/licenses/>.
#
from _decimal import Decimal

# Data, CMI, Deadline, Max. Packetsize
profile1 = (Decimal('1024'), Decimal('0.000250'), Decimal('0.000250'), Decimal('1024'))
profile2 = (Decimal('2048'), Decimal('0.0005'), Decimal('0.0005'), Decimal('2048'))
profile3 = (Decimal('4096'), Decimal('0.001'), Decimal('0.001'), Decimal('4096'))
profile4 = (Decimal('8192'), Decimal('0.002'), Decimal('0.002'), Decimal('8192'))
profile5 = (Decimal('12176'), Decimal('0.004'), Decimal('0.004'), Decimal('12176'))

profiles = [profile1, profile2, profile3, profile4, profile5]
profile_names = ['profile1', 'profile2', 'profile3', 'profile4', 'profile5']
