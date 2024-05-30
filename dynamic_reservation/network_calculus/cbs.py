#     Copyright (C) 2024 Lisa Maile
#
#     lisa.maile@fau.de
#
#     This file is part of the DYnamic Reliable rEal-time Communication in Tsn (DYRECTsn) framework.
#
#     DYRECTsn is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Lesser General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     DYRECTsn is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public License
#     along with DYRECTsn.  If not, see <http://www.gnu.org/licenses/>.
#
from typing import Union
from _decimal import Decimal
from dynamic_reservation.network_calculus.arithmetic_helper_functions import *


class CBS_system:
    """
    class which can be used on edge to represent the NC information and perform calculation for the
    CBS queue of this edge
    """

    def __init__(self, queue_id: str, outputqueues: int, linkrate: Decimal, be_slope: Decimal, queuesize: Decimal,
                 maxpacket_be: Decimal) -> None:
        """
        Create a CBS object for this edge in the topology.
        This contains all information given by the network.
        The queue is initialized later by setidleslopes(...) etc.

        :param queue_id: Name of the node to identify it for debugging e.g. (1,2,0).
        :param outputqueues: How many queues at each outgoing link are for CBS? E.g. if, we have two priorities for
         CBS, this value is 2. Currently, we only support CBS queues if they are the highest priority in the network,
         meaning that priority 0 needs to be CBS.
        :param linkrate: Outgoing linkrate_graph in bits/s.
        :param be_slope: The rate that needs to be reserved for lower priority queues in bits/s (the sum of CBS
         idleSlopes needs to be smaller than linkrate_graph-be_slope).
        :param queuesize: Maximum size of the queue, before packets are lost in bits.
        :param maxpacket_be: Maximum BE packet size in bits that can occur at this output port.
        """

        # ------------------------------------------  Check arguments  --------------------------------------------
        assert 0 < outputqueues <= 8, "nrqueues should be 1-8, queue_id: {}".format(queue_id)
        assert linkrate > 0, "linkrate_graph should be >0, queue_id: {}".format(queue_id)
        assert maxpacket_be > 0, "max. packet size should be >0, queue_id: {}".format(queue_id)
        assert be_slope >= 0, "be_slope should be >=0, queue_id: {}".format(queue_id)
        assert queuesize > 0, "queuesize should be >0, queue_id: {}".format(queue_id)

        # -------------------------------------------  Set arguments  ---------------------------------------------
        self.id = queue_id
        # topology information
        self.outputqueues = outputqueues
        self.linkrate = linkrate
        self.queuesize = queuesize
        self.maxpacket_be = maxpacket_be  # max packetsize
        self.be_slope = be_slope

        # CBS information
        self.maxpacket = [Decimal('0')] * outputqueues
        self.slopes = [Decimal('0')] * outputqueues
        self.vmax = [Decimal('0')] * outputqueues  # max. possible credit of queue
        self.vmin = [Decimal('0')] * outputqueues  # min. possible credit of queue
        self.shaperburst = [Decimal('inf')] * outputqueues  # the max. allowed burst by the shaper curve

        # arrival information (we have per-edge and per-link arrivals which are shaped separately to form
        # the complete arrival curve for this queue)
        # the arrival curves are represented as lists of tuples with all kinks of the arrival curve
        # each tuple has the form (x-value, y-value, rate from this point), e.g. [(0,0,C),(x,y,r),...]
        # remember: edge (u,v,p), link(u,v)
        self.edgearrivals = {}  # dict[str, List[Tuple[Decimal]]] key: edge, list: arrival from this edge
        self.edge_shaped_arrivals = {}  # dict[str, List[Tuple[Decimal]]] key: edge, list: shaped arrival from this edge
        self.linkarrival = {}  # dict[str, List[Tuple[Decimal]]] key: link, list: sum of edge arrivals shaped to  link
        self.arrivalpoints = []  # List[Tuple[Decimal]] sum of link arrivals, actual arrival for this queue

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------  Idle Slope Functions  ---------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    def setidleslopes(self, slopes: List[Decimal]) -> None:
        """
        sets the idle slopes for all queues at this output port (we need the knowledge of all to calculate
        the right backlog, delay, etc.)

        :param slopes: new idle slopes (in bits/s) in the form of a list [slope for prio 0, slope for prio 1, ...]
        """
        assert all(slope >= 0 for slope in slopes), "idleslopes should not be negative, queue_id:{}".format(self.id)

        if sum(self.maxpacket) == 0:
            print("WARNING: We have not provided the per-queue sizes properly before calculating the service curve.")
            self.maxpacket = [self.maxpacket_be] * self.outputqueues

        self.slopes = slopes

        # recalculate the credits that each queue can reach
        self.updatevmax()
        self.updatevmin()

        # recalculate the shaper curve for each queue
        self.updateshaper()

    def setidleslope(self, queue: int, slope: Decimal) -> None:
        """
        sets the new idle slope for one queue

        :param queue: queue for which the idle slope should be updated
        :param slope: new slope in bit/s
        """
        newslopes = self.getidleslopes()
        if queue < len(newslopes):
            newslopes[queue] = slope
            self.setidleslopes(newslopes)
        else:
            print("Idle slope updated for a queue which does not exist -> nothing was changed; queue_id: {}".format(
                self.id))

    def getidleslopes(self) -> List[Decimal]:
        """
        :return: returns the list of all idle slopes at this output port
        """
        return self.slopes.copy()

    def setpacketsizes(self, sizes: List[Decimal]) -> None:
        assert all(size >= 0 for size in sizes), "packet sizes should not be negative, queue_id:{}".format(self.id)
        self.maxpacket = sizes

    def setpacketsize(self, queue: int, size: Decimal) -> None:
        newsizes = self.maxpacket.copy()
        if queue < len(newsizes):
            if newsizes[queue] < size:
                newsizes[queue] = size
                self.setpacketsizes(newsizes)
        else:
            print("Packet size updated for a queue which does not exist -> nothing was changed; queue_id: {}".format(
                self.id))

    def slope_for_delay(self, arrival: List[Tuple[Decimal, Decimal, Decimal]], delay: Decimal, queue: int,
                        maxpackets=None, slopes=None) -> Union[Decimal, int]:
        """
        calculates the min. idleslope that is needed to guarantee the *delay* at *queue* with assuming the *arrival*

        This is possible, as we know the initial latency of the service curve T
        (as this depends only on the higher slopes and the other packet sizes)

        we look at every kink in the arrival curve:
        -- with given T, how should the idleslope be that we end up with the maximum delay
        -- then return the maximum idleslope of all kinks

        :param arrival: arrival for which delay is guaranteed
        :param delay: delay which needs to be guaranteed
        :param queue: queue for which this guarantee should be calculated
        :param maxpackets:  optional -> if we want to use other packetsizes for higher priority queues than the ones
         that are currently defined in this element (we set them later)
        :param slopes: optional -> if we want to use other slopes for higher priority queues than the ones
         that are currently defined in this element
        :return: max(results) of new_idleslope or None, if the delay cannot be guaranteed
        """
        if not arrival:
            arrival = self.arrivalpoints.copy()

        if not slopes:
            slopes = self.slopes

        if not maxpackets:
            maxpackets = self.maxpacket

        # T = (self.vMax / self.idleSlope[self.prio]))
        # calculate T without dependency on own idle slope (meaning c_max):
        i_sum = Decimal('0')
        s_sum = Decimal('0')
        for j in range(0, queue):
            i_sum += slopes[j]
            s_sum += self.sumHelper(j, slopes, maxpackets)

        rem_slope = i_sum - self.linkrate
        if rem_slope != 0:
            latency_t = (s_sum - self.maxpacket_be) / rem_slope
        else:
            return -1

        if latency_t > delay:
            return -1

        maxslope = Decimal('0')
        # we need R
        for point in arrival:
            # (x+D-T) * I = y (solve for I)
            new_idleslope = point[1] / (point[0] + delay - latency_t)
            if new_idleslope > maxslope:
                maxslope = new_idleslope

        # idleslope must be greater or equal to long term arrival rate
        if 0 < maxslope < arrival[-1][2]:
            return arrival[-1][2]

        return maxslope

    # ----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------  Arrival Curve Functions  --------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    def addarrivalcurve(self, arrival: List[Tuple[Decimal, Decimal, Decimal]], flow_packetsize=Decimal('0'),
                        priority=-1, virtual=False, prevlinkrate=Decimal('0'), port=-1, shaperrate=Decimal('-1'),
                        shaperburst=Decimal('-1')) \
            -> Union[List[Tuple[Decimal, Decimal, Decimal]], None]:
        """
        Updates the arriving traffic at a queue.

        1. add new arrival to the input edge (e.g. key (1,2,0)) "input_edge"
        1a. shape this according to its shaper curve "input_edge_shaped" (not for fD)
        2. sum up the two input queue arrivals from this incoming link (e.g. the keys (1,2,0) and (1,2,1))
        "sum_input_queue"
        3. shape this sum according to link rate "sum_input_queue"
        4. the arrival for this edge is the sum of all "sum_input_queues" from all incoming links (e.g. (1,2,x) and
        (3,2,x)) "new_arrival"

        :param flow_packetsize: since a new flow is added, we update the packetsize
        :param priority: priority of the flow
        :param arrival: new arrival curve: (x1,y1,r1), (x2,y2,r2), ... ] TB: [(0,b,r)], Ct: [0,0,C]
        :param virtual: if virtual, then we only return the new sum of the arrival curve, but do not update the
        arrival curve of this object
        :param prevlinkrate: for link shaping from this edge, -1 if no prev. link
        :param port: as we keep separate arrival curves for all incoming ports, I say -1 if there is no prev. edge
        :param shaperrate: shaping rate from "port", -1 if no prev. cbs queue who shaped (only 2. hop will cbs shape)
        :param shaperburst: shaping burst from "port"
        :return:
        """
        maxpacket = max(self.maxpacket[priority], flow_packetsize)
        if not virtual:
            if self.maxpacket[priority] < flow_packetsize:
                self.maxpacket[priority] = flow_packetsize

        # 1. add to input edge (input port is input edge)
        input_edge = self.addtoinputedge(port, arrival)
        # for fD we leave it unshaped (we will shape the sum with a maximum shaper rate)
        input_edge_shaped = input_edge

        # if shaperrate == -1 then this is the first hop after sending and we cannot shape
        if shaperrate > -1:
            # we sum them up and then shape them
            # 2. sum of incoming link without shaping
            sum_input_queue = add(self.sum_of_shaped_input(port), input_edge)
            # then shape, but to max. shaperrate (both WC shaperrates should be equal)
            sum_input_queue = shape_to_shaper(sum_input_queue.copy(), shaperrate, shaperburst)
        else:
            # 2. sum of incoming link
            sum_input_queue = add(self.sum_of_shaped_input(port), input_edge)

        # if prevlinkrate == 0 then this is the sending device and we will not shape
        if prevlinkrate > 0:
            # 3. shape
            sum_input_queue = shape_to_link(sum_input_queue, prevlinkrate, maxpacket)

        # 4. sum up the other input links and then the current
        new_arrival = add(self.sumoflinks(port), sum_input_queue)

        # if virtual, we simply return the overall sum for arrival curve for this edge
        if virtual:
            return new_arrival

        # if real, we also update the arrival curve variables
        else:
            self.arrivalpoints = new_arrival
            if port == -1:
                inputlink = -1
            else:
                inputlink = (port[0], port[1])
            self.linkarrival[inputlink] = sum_input_queue
            self.edge_shaped_arrivals[port] = input_edge_shaped
            self.edgearrivals[port] = input_edge

    def addredundantflows(self, virtual, priority, arrival_1, prevlinkrate_1, arrival_2, prevlinkrate_2,
                          path_delay_delta, flow_packetsize, arrival_before_replication):
        maxpacket = max(self.maxpacket[priority], flow_packetsize)
        if not virtual:
            if self.maxpacket[priority] < flow_packetsize:
                self.maxpacket[priority] = flow_packetsize

        shaped_arrival_1 = shape_to_link(arrival_1, prevlinkrate_1, maxpacket)
        shaped_arrival_2 = shape_to_link(arrival_2, prevlinkrate_2, maxpacket)
        sum_arrival = add(shaped_arrival_1, shaped_arrival_2)
        shaped_sum_arrival = shape_to_8021cb(sum_arrival, arrival_before_replication, path_delay_delta)
        new_arrival = add(self.arrivalpoints, shaped_sum_arrival)

        if virtual:
            return new_arrival

        # if real, we also update the arrival curve variables
        else:
            self.arrivalpoints = new_arrival

    # ----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------  Guarantee Calculation  ----------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    def getdelay(self, queue: int) -> Decimal:
        """
        calculate max. delay for each kink (x,y) in arrival curve of *queue* and return largest
        D = T + y/R -x

        :param queue: queue for which the delay is wanted
        :return: max. delay in s
        """
        delays = []
        if self.slopes[queue] == 0:
            return Decimal('inf')

        T = self.vmax[queue] / self.slopes[queue]
        R = self.slopes[queue]
        for arrival_tuple in self.arrivalpoints:
            delays.append(T + arrival_tuple[1] / R - arrival_tuple[0])

        return max(delays)

    def get_talker_delay(self, arrival=None):
        """
        We simply use the constant bitrate curve of the linkrate at look at the max. delay at the talker.
        """
        if not arrival:
            arrival = self.arrivalpoints.copy()
        b = arrival[0][1]
        if not arrival[0][0] == 0 and b > 0:
            print("Somethings wrong", arrival)
        delay = b / self.linkrate
        return delay

    def getbacklog(self, queue: int, arrival=None, slopes=None) -> Decimal:
        """
        calculate max. backlog for each kink (x,y) in arrival curve of *queue* and at T and return largest

        :param queue: queue for which the backlog is wanted
        :param arrival: optional -> arrival curve for which we want the backlog
        :param slopes: optional -> idle slopes for which we want the backlog
        :return: max. backlog in bits
        """
        if arrival is None:
            arrival = self.arrivalpoints.copy()
        if not arrival:
            return Decimal('0')
        if slopes:
            T = self.updatevmax(newslopes=slopes, queue=queue) / slopes[queue]
            slope = slopes[queue]
        else:
            T = self.vmax[queue] / self.slopes[queue]
            slope = self.slopes[queue]

        maxbacklog = Decimal('0')

        # get point in curve that is located directly to the right of T
        index = -1
        i = 0
        for point in arrival:
            # for each point in the arrival curve which has a bigger value than T
            # get the distance to the service curve
            if point[0] <= T:
                i += 1
                continue
            # Backlog at this point is:
            # B = y - (x - T) * I
            newbacklog = point[1] - (point[0] - T) * slope
            if newbacklog > maxbacklog:
                maxbacklog = newbacklog
            index = i
            i += 1

        # get the distance from T to the arrival curve
        # use point in curve that is located directly to the right of T
        # this returns the first index that is greater than T
        # old: index = next(iter([x for x, val in enumerate(arrival) if val[0] > T]), -1)
        if index == -1:
            lefttuple = arrival[-1]
        else:
            # point to the left:
            lefttuple = arrival[index - 1]
        # the backlog is B = (T-x) * r + y
        newbacklog = (T - lefttuple[0]) * lefttuple[2] + lefttuple[1]
        if newbacklog > maxbacklog:
            maxbacklog = newbacklog
        # return the maximum backlog
        return maxbacklog

    def arrivaloutput(self, queue: int, delay=-1, arrival=None, burstinessincrease=True) \
            -> Tuple[List[Tuple[Decimal, Decimal, Decimal]], Decimal, Decimal]:
        """
        newalpha = alpha x beta is alpha shifted to the left by T

        Of course the output will be shaped by the outgoing link rate, but this is done in the next queue.
        Instead, here we want to calculate the output of the current arrival, which has been shaped by its
        ingoing link and previous shaper curve.
        Thus, the output is newalpha = min(alpha,link rate)(t + T).

        :param burstinessincrease: only increase output arrival burstiness if the flow's interval < deadline
        :param queue: queue through which the arrival traverses
        :param arrival: arrival curve (e.g. of one flow) for which we want the output. If None, then we calculate
         the aggregated output for the whole queue arrivals.
        :param delay: only for research (assuming that the arrival curve experiences *delay*, used instead of T then)
        :return: maximum output arrival curve as List of Tuples
        """
        assert 0 <= queue < 8, "The queue to check the burstiness increase must be between 0 and 7"
        assert delay >= 0, "For the output flow calculation, we need a delay >= 0"

        if not arrival:
            arrivalcurve = self.arrivalpoints.copy()
        else:
            arrivalcurve = arrival

        if burstinessincrease:
            arrivalcurve = shift(arrivalcurve, delay)

        # for the next hop, we can tell him what is the shaperrate
        # and burst of this queue to properly shape this output there
        shaperrate = self.slopes[queue]
        shaperburst = self.shaperburst[queue]

        return arrivalcurve, shaperrate, shaperburst

    # ----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------  Shaper Functions  -----------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    def maxShaper(self):
        """
        This gets the total max. shaperrate and shaperburst for this link.
        :return: worst case shaping if we shape sum of ports
        """
        shaperrate = self.linkrate - self.be_slope
        # max. burst is calculated if the idle slope for the queue is the max. allowed rate
        # and the others are 0
        # for prio 0, the burst is constant, so we can say that the slope is e.g. 0
        # for example:
        # prio: 1 -> [0, 0]
        # prio: 2 -> [0, linkrate_graph-be_slope]
        shaperbursts = Decimal('0')
        # calc. the max. shaperburts for all queues
        lenslopes = len(self.slopes)
        for q in range(lenslopes):
            max_slopes = [Decimal('0')] * lenslopes
            if q > 0:
                max_slopes[q] = self.linkrate - self.be_slope
            vmax = self.updatevmax(newslopes=max_slopes, maxpackets=[self.maxpacket_be] * self.outputqueues, queue=q)
            vmin = self.updatevmin(newslopes=max_slopes, maxpackets=[self.maxpacket_be] * self.outputqueues)
            shaperbursts += vmax - vmin[q]

        # packetizing effect
        shaperbursts += self.maxpacket_be

        return shaperrate, shaperbursts

    def updateshaper(self) -> None:
        """
        maximum output burst that a queue can get due to the shaping of CBS; updates self.shaperburst[queue];
        always updated when new slopes are set; includes packetizing effect
        (note for myself: it is the real burst, b = I * (Vmax - Vmin) / I)
        """
        for queue in range(self.outputqueues):
            self.shaperburst[queue] = self.vmax[queue] - self.vmin[queue] + self.maxpacket[queue]

    # ----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------  Helper Functions  -----------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    def updatevmax(self, newslopes=None, maxpackets=None, queue=-1) -> Union[List[Decimal], Decimal]:
        """
        maximum credit that a queue can reach; updates self.vmax[queue]; formula of Mohammadpour but this is the
        same as Zhao's
        :param maxpackets: new packetsizes
        :param queue: for which queue? if -1 then for all queues
        :param newslopes: for fD, we want the vmax that would be present with our new virtual idleslopes (we do
        not update self.vmax then)
        :return: for fD the max credit
        """

        if not newslopes:
            slopes = self.slopes
        else:
            slopes = newslopes

        vmax = []
        if queue == -1:
            for queue in range(self.outputqueues):
                i_sum = Decimal('0')
                s_sum = Decimal('0')
                for j in range(0, queue):
                    i_sum += slopes[j]
                    s_sum += self.sumHelper(j, slopes, maxpackets)
                if (i_sum - self.linkrate) == Decimal('0'):
                    vmax.append(Decimal('Infinity'))
                else:
                    vmax.append(((s_sum - self.maxpacket_be) / (i_sum - self.linkrate) * slopes[queue]))
            if not newslopes:
                self.vmax = vmax

            return vmax
        else:
            i_sum = Decimal('0')
            s_sum = Decimal('0')
            for j in range(0, queue):
                i_sum += slopes[j]
                s_sum += self.sumHelper(j, slopes, maxpackets)

            if (i_sum - self.linkrate) == Decimal('0'):
                return Decimal('Infinity')
            else:
                return (s_sum - self.maxpacket_be) / (i_sum - self.linkrate) * slopes[queue]

    def updatevmin(self, newslopes=None, maxpackets=None) -> List[Decimal]:
        """
        minimum credit that a queue can reach (is negative); updates self.vmin[queue];
        :param maxpackets: new packetsizes
        :param newslopes: for fD, we want the vmin that would be present with our new virtual idleslopes (we do
        not update self.vmin then)
        :return: for fD the min credit
        """
        if not newslopes:
            slopes = self.slopes
        else:
            slopes = newslopes

        vmin = []
        for queue in range(self.outputqueues):
            vmin.append(self.sumHelper(queue, slopes, maxpackets))

        if not newslopes:
            self.vmin = vmin

        return vmin

    def sumHelper(self, j: int, newslopes=None, maxpackets=None):
        """
        return the sum of min credits
        :param maxpackets: optional -> if wanted, for new packet sizes
        :param j: the queue queue_id
        :param newslopes: slopes
        :return: result of the sum that is required for the credit calculation
        """
        if not newslopes:
            slopes = self.slopes
        else:
            slopes = newslopes

        if not maxpackets:
            maxpackets = self.maxpacket
        # SendSlope = IdleSlope - Linkrate
        return (maxpackets[j] / self.linkrate) * (slopes[j] - self.linkrate)

    def addtoinputedge(self, edge: Tuple[int], arrival: List[Tuple[Decimal, Decimal, Decimal]]) \
            -> List[Tuple[Decimal, Decimal, Decimal]]:
        """
        add new arrival to the edge for this element (e.g. key (1,2,0))
        """

        # get current input port curve
        if edge in self.edgearrivals:
            input_queue = self.edgearrivals[edge].copy()
        else:
            input_queue = []

        input_queue = add(input_queue, arrival)

        return input_queue

    def addtolink(self, link: Tuple[int], arrival: List[Tuple[Decimal, Decimal, Decimal]]) \
            -> List[Tuple[Decimal, Decimal, Decimal]]:
        """
        add new arrival to the link queue for this element (e.g. key (1,2) "link")
        """

        # get current input port curve
        if link in self.linkarrival:
            link_queue = self.linkarrival[link].copy()
        else:
            link_queue = []

        link_queue = add(link_queue, arrival)

        return link_queue

    def sumoflinks(self, notlink: Tuple[int]) -> List[Tuple[Decimal, Decimal, Decimal]]:
        """
        sum up the input arrivals of all links, except the one of this port (e.g. the keys (1,2,0) and (1,2,1))
        :param notlink: we don't sum because we use this function when a new arrival for notport has been calculated
        :return: sum of links
        """
        if notlink == -1:
            return []

        currentsum = []
        for link in self.linkarrival:
            # check if it is a different input link
            if link != -1:
                if link[0] != notlink[0]:
                    currentsum = self.addtolink(link, currentsum)

        return currentsum

    def sum_of_shaped_input(self, notedge: Tuple[int]) -> List[Tuple[Decimal, Decimal, Decimal]]:
        """
        sum up the input edge arrivals from the same incoming link, except the one of this edge "notedge" (e.g. the
        keys (1,2,0) and (1,2,1))
        :param notedge: this is not summed up as we have a new value for it usually (hint: if we want to sum up all,
        we could write notport = (link,link,-1)
        :return: sum of all edges
        """
        if notedge == -1:
            return []

        currentsum = []
        for port in self.edge_shaped_arrivals:
            # check if it is the same input link, but not the same priority
            if port != -1:
                if port[0] == notedge[0] and port[1] == notedge[1] and port[2] != notedge[2]:
                    currentsum = self.addtoinputedge(port, currentsum)

        return currentsum

    @staticmethod
    def getProcDelay():
        """
        just in case we want to define a maximum processing delay

        :return: delay in s
        """
        processingDelay = Decimal('0.000008')
        return processingDelay

    @staticmethod
    def getProgDelay():
        """
        just in case we want to define a maximum propagation delay

        Hint: The worst-case propagation delay is 100.0 / 200000000.0 for a Ethernet cable of max. length (100m)
        :return: delay in s
        """
        propagationDelay = Decimal('0.00000005')  # 100.0 / 200000000.0
        return propagationDelay
