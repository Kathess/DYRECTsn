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
from decimal import Decimal


class Flow:
    def __init__(self, flowID, source, sinks, data_per_interval, sending_interval, deadline, max_frame_size,
                 priority=None, frames_per_interval=1, redundancy=False):
        self.flowID = flowID  # Can be anything, int, string, etc. just for output.
        self.source = source
        self.sinks = sinks
        self.data_per_interval = Decimal(str(data_per_interval))
        self.sending_interval = Decimal(str(sending_interval))
        self.deadline = Decimal(str(deadline))
        self.frames_per_interval = frames_per_interval
        self.max_frame_size = Decimal(str(max_frame_size))
        self.priority = priority
        self.redundancy = redundancy

        # After reservation, these are filled.
        self.paths = []  # can optionally also be filled by user
        self.path_delays = []
        self.budget_path_delays = []

        # if the flow requires redundancy, these are the redundancy configuration parameters for the
        # stream reservation function (SRF)
        self.srf_config = {}

    def __repr__(self):
        return (
            f"Flow(id={self.flowID}, source={self.source}, sink={self.sinks}, data_per_interval={self.data_per_interval}, "
            f"sending_interval={self.sending_interval}, deadline={self.deadline}, "
            f"max_frame_size={self.max_frame_size}, priority={self.priority}, redundancy={self.redundancy})")

    def showReservation(self):
        return (f"Flow(paths={self.paths}, path_delays={self.path_delays}, "
                f"budget_path_delays={self.budget_path_delays})")

    def printRedundancyConfig(self):
        return (f"Flow(srf_config={self.srf_config})")

    def resetReservationInfo(self):
        self.paths = []
        self.path_delays = []
        self.budget_path_delays = []
        self.srf_config = {}


if __name__ == '__main__':
    # Example usage:
    flow_example = Flow(1, 22, 44, 1856, 2500e-6, 1e-3, 1856, 1, True)
    print(flow_example)
