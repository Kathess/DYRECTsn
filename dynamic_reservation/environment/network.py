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
import matplotlib.pyplot as plt
import networkx as nx


class Network:
    """
    class which contains all network calculus information for one hop (arrival and service curves per queue)

    contains functions to calculate the delay and burstiness increase

    """

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.nodes = []

    def create_link(self, nodeA, nodeB, priorities, queue_in, rate_out, bidirectional=True, graph=None):
        if graph is None:
            graph = self.graph
        if not nodeA in self.nodes:
            graph.add_node(nodeA)
            self.nodes.append(nodeA)
        if not nodeB in self.nodes:
            graph.add_node(nodeB)
            self.nodes.append(nodeB)

        for i in range(priorities):
            graph.add_edge(nodeA, nodeB)
            graph[nodeA][nodeB][i]['priority'] = i
            graph[nodeA][nodeB][i]['queue_in'] = queue_in
            graph[nodeA][nodeB][i]['rate_out'] = rate_out
            graph[nodeA][nodeB][i]['reserved'] = 0
            graph[nodeA][nodeB][i]['perc_res'] = 1
            if bidirectional:
                graph.add_edge(nodeB, nodeA)
                graph[nodeB][nodeA][i]['priority'] = i
                graph[nodeB][nodeA][i]['queue_in'] = queue_in
                graph[nodeB][nodeA][i]['rate_out'] = rate_out
                graph[nodeB][nodeA][i]['reserved'] = 0
                graph[nodeB][nodeA][i]['perc_res'] = 1


if __name__ == '__main__':
    newGraph = Network()
    newGraph.create_link(1, 2, 2, 100, 1500)

    nx.draw_planar(newGraph.graph, with_labels=True)
    plt.show()
