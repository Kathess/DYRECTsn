# Copyright (C) 2024 Timo Salomon for HAW Hamburg, Germany
# timo.salomon@haw-hamburg.de
import sys
from _decimal import Decimal
from networkx import grid_graph
import random
import networkx as nx
import matplotlib.pyplot as plt
from dynamic_reservation.environment.network import Network


def getTopology(inputLinks, stages, queuesize=Decimal(sys.float_info.max), output=False, nrCBSqueues=1):
    # create network
    net = Network()
    linkrate = Decimal(1.0e8)  # 100 Mbit/s
    # create switches
    bridges = ["sw_aggregate"]
    for i in range(0, inputLinks):
        for j in range(0, stages):
            bridges.append("sw_input_" + str(i) + "_stage_" + str(j))
    print("num bridges " + str(len(bridges)))
    links = 0
    # add switches to network
    for i in range(0, inputLinks):
        for j in range(0, stages):
            if j == stages - 1:
                net.create_link(
                    nodeA="sw_input_" + str(i) + "_stage_" + str(j),
                    nodeB="sw_aggregate",
                    priorities=nrCBSqueues,
                    queue_in=queuesize,
                    rate_out=linkrate,
                )
                links += 1
            else:
                net.create_link(
                    nodeA="sw_input_" + str(i) + "_stage_" + str(j),
                    nodeB="sw_input_" + str(i) + "_stage_" + str(j + 1),
                    priorities=nrCBSqueues,
                    queue_in=queuesize,
                    rate_out=linkrate,
                )
                links += 1

    # add hosts
    tsn_devices = ["listener"]
    for i in range(0, inputLinks):
        tsn_devices.append("talker_" + str(i))
        for j in range(0, stages):
            tsn_devices.append("node_input_" + str(i) + "_stage_" + str(j))
    tsn_devices.append("node_aggregate")
    print("num tsn devices " + str(len(tsn_devices)))
    # add hosts to network
    net.create_link(
        nodeA="listener", nodeB="sw_aggregate", priorities=nrCBSqueues, queue_in=queuesize, rate_out=linkrate
    )
    links += 1
    net.create_link(
        nodeA="node_aggregate", nodeB="sw_aggregate", priorities=nrCBSqueues, queue_in=queuesize, rate_out=linkrate
    )
    links += 1
    for i in range(0, inputLinks):
        net.create_link(
            nodeA="talker_" + str(i),
            nodeB="sw_input_" + str(i) + "_stage_0",
            priorities=nrCBSqueues,
            queue_in=queuesize,
            rate_out=linkrate,
        )
        links += 1
        for j in range(0, stages):
            net.create_link(
                nodeA="node_input_" + str(i) + "_stage_" + str(j),
                nodeB="sw_input_" + str(i) + "_stage_" + str(j),
                priorities=nrCBSqueues,
                queue_in=queuesize,
                rate_out=linkrate,
            )
            links += 1

    print("num links " + str(links))
    if output:
        nx.draw_planar(net.graph, with_labels=True)
        plt.show()

    return net.graph, bridges, tsn_devices
