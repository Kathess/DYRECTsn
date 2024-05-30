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

import sys
from _decimal import Decimal
from networkx import grid_graph
import random
import networkx as nx
import matplotlib.pyplot as plt
from dynamic_reservation.environment.network import Network


def example_topology(linkrate=15000000.0, queuesize=Decimal(sys.float_info.max), output=False, nrCBSqueues=2):
    """
    Hint: We do not model any central controller, otherwise they might be used for routing.

    Parameters:
        linkrate: in bits/second;
        queuesize: maximum buffer size - flows are rejected if the max. queue/buffer size is surpassed;
        nrCBSqueues: how many queues should be available for reserved traffic? Reserved traffic means traffic
        with delay guarantees. There can be more queues with a lower priority that do not need to be modeled. The
        lower priority queues' traffic is considered via the max. packet size;
        output: whether the plot of the topology should be drawn;
    """

    # add the bridges
    network = Network()
    bridges = [1, 2, 3]
    # Default parameter of create_link: bidirectional=True
    network.create_link(nodeA=1, nodeB=2, priorities=nrCBSqueues, queue_in=queuesize, rate_out=linkrate)
    network.create_link(2, 3, nrCBSqueues, queuesize, linkrate)
    network.create_link(1, 3, nrCBSqueues, queuesize, linkrate)

    # add TSN devices
    tsn_devices = [11, 22, 33, 44]
    network.create_link(11, 1, nrCBSqueues, queuesize, linkrate)
    network.create_link(22, 2, nrCBSqueues, queuesize, linkrate)
    network.create_link(44, 2, nrCBSqueues, queuesize, linkrate)
    network.create_link(33, 3, nrCBSqueues, queuesize, linkrate)

    # add cross devices
    cross_devices = [111, 222]
    network.create_link(111, 1, nrCBSqueues, queuesize, linkrate)
    network.create_link(222, 3, nrCBSqueues, queuesize, linkrate)

    if output:
        nx.draw_planar(network.graph, with_labels=True)
        plt.show()

    return network.graph, bridges, tsn_devices, cross_devices


def line_topology(rate_out=15000000.0, queue=512000.0, output=False, nrCBSqueues=2, length=4, nodes_per_bridge=1):
    network = Network()
    curr_id = 1
    bridges = [1]
    for i in range(length - 1):
        network.create_link(i + 1, i + 2, nrCBSqueues, queue, rate_out)
        bridges.append(i + 2)
        curr_id += 1

    curr_id += 1
    end_nodes = []
    for bridge in bridges:
        for i in range(nodes_per_bridge):
            network.create_link(curr_id, bridge, nrCBSqueues, queue, rate_out)
            end_nodes.append(curr_id)
            curr_id += 1

    if output:
        nx.draw_planar(network.graph, with_labels=True)
        plt.show()

    # curr_id is the last node
    return network.graph, end_nodes, curr_id - 1


def star_topology(rate_out=15000000.0, queue=512000.0, output=False, nrCBSqueues=2, nodes_per_level=4):
    network = Network()
    bridges = []
    end_nodes = []

    for n in range(nodes_per_level):
        network.create_link(1, 2 + n, nrCBSqueues, queue, rate_out)
        bridges.append(2 + n)

    i = bridges[-1] + 1
    for bridge in bridges:
        for n in range(nodes_per_level):
            network.create_link(bridge, i, nrCBSqueues, queue, rate_out)
            end_nodes.append(i)
            i += 1

    if output:
        nx.draw_planar(network.graph, with_labels=True)
        plt.show()

    return network.graph, end_nodes


def ring(ringsize=3, nrEdgeNodes=3, nrEndNodes=3, rate_out=15000000.0, queue=512000.0, output=False, nrCBSqueues=2,
         randomES=True, startID=1):
    network = Network()
    list_endnodes = []

    # create ring
    ringIDs = [startID]
    firstID = startID - 1
    for id in range(ringsize - 1):
        if rate_out is None:
            rate_out = random.choice([100000000, 1000000000])
        network.create_link(firstID + id + 1, firstID + id + 2, nrCBSqueues, queue, rate_out)
        ringIDs.append(firstID + id + 2)
    # close ring
    if rate_out is None:
        rate_out = random.choice([100000000, 1000000000])
    network.create_link(firstID + ringsize, startID, nrCBSqueues, queue, rate_out)

    curr_id = firstID + ringsize + 1
    if randomES:
        for endID in range(nrEdgeNodes):
            ringswitch = random.randrange(len(ringIDs)) + startID
            # we add an extra hop at each edge device
            if rate_out is None:
                rate_out = random.choice([100000000, 1000000000])
            network.create_link(ringswitch, curr_id, nrCBSqueues, queue, rate_out)
            endnodes = random.randint(1, nrEndNodes)
            for i in range(endnodes):
                if rate_out is None:
                    rate_out = random.choice([100000000, 1000000000])
                network.create_link(curr_id, curr_id + 1 + i, nrCBSqueues, queue, rate_out)
                list_endnodes.append(curr_id + 1 + i)
            curr_id += endnodes + 1
    else:
        edgenode_perbridge = max(1, int(nrEdgeNodes / len(ringIDs)))
        for ringswitch in ringIDs:
            for edgenode in range(edgenode_perbridge):
                edgenodeID = curr_id
                network.create_link(ringswitch, curr_id, nrCBSqueues, queue, rate_out)
                curr_id += 1
                for endnode in range(nrEndNodes):
                    network.create_link(edgenodeID, curr_id, nrCBSqueues, queue, rate_out)
                    list_endnodes.append(curr_id)
                    curr_id += 1
    if output:
        nx.draw(network.graph, with_labels=True)
        plt.show()

    return network.graph, list_endnodes, ringIDs


def tree_network(branchlength=3, nrbranches=5, rate_out=15000000.0, queue=512000.0, output=False, nrCBSqueues=2):
    network = Network()
    list_endnodes = []
    plc = [1]
    centralSwitch = 2

    network.create_link(1, centralSwitch, nrCBSqueues, queue, rate_out)
    currentID = centralSwitch
    for branch in range(nrbranches):
        network.create_link(centralSwitch, currentID + 1, nrCBSqueues, queue, rate_out)
        currentID += 1
        for leave in range(branchlength):
            network.create_link(currentID, currentID + 1, nrCBSqueues, queue, rate_out)
            currentID += 1
            if leave == branchlength - 1:
                list_endnodes.append(currentID)

    if output:
        nx.draw(network.graph, with_labels=True)
        plt.show()

    return network.graph, list_endnodes, plc


def complete_graph(ringsize=3, linkrate=15000000.0, queuesize=Decimal(sys.float_info.max), output=False, nrCBSqueues=2):
    """
    Connect all nodes with all other nodes.
    """
    network = Network()
    list_endnodes = []
    list_ringnodes = []
    for i in range(ringsize):
        for j in range(i + 1, ringsize):
            network.create_link(i + 1, j + 1, nrCBSqueues, queuesize, linkrate)
        list_ringnodes.append(i + 1)

    id_ES = len(list_ringnodes) + 1
    for i in list_ringnodes:
        network.create_link(i, id_ES, nrCBSqueues, queuesize, linkrate)
        list_endnodes.append(id_ES)
        id_ES += 1

    if output:
        nx.draw(network.graph, with_labels=True)
        plt.show()

    return network.graph, list_endnodes[1:], list_endnodes[0]  # endnodes , PLC


def meshed_graph(levels=3, linkrate=15000000.0, queuesize=Decimal(sys.float_info.max), output=True, nrCBSqueues=2):
    network = Network()
    G = grid_graph(dim=(1, levels, levels))
    sender = 0
    receiver = int(''.join(map(str, list(G.nodes())[-1])))
    for edge in G.edges():
        start = int(''.join(map(str, edge[0])))
        end = int(''.join(map(str, edge[1])))
        network.create_link(start, end, priorities=nrCBSqueues, rate_out=linkrate, queue_in=queuesize,
                            bidirectional=False)

    if output:
        nx.draw(network.graph, with_labels=True)
        plt.show()

    return network.graph, sender, receiver  # endnodes , PLC
