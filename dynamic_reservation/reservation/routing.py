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
import heapq
import networkx as nx


def dijkstra_with_delay(graph, simple_graph, start, end, max_delay, weighted=True):
    if not nx.has_path(simple_graph, start, end):
        raise nx.NetworkXNoPath(f"No path between {start} and {end}")

    # Initialize distances and predecessors
    sink = end
    distances = {vertex: float('infinity') for vertex in simple_graph.nodes}
    delays = {vertex: float('-infinity') for vertex in simple_graph.nodes}
    predecessors = {vertex: None for vertex in simple_graph.nodes}
    distances[start] = 0
    delays[start] = 0

    # Use a priority queue to store vertices to visit
    priority_queue = [(0, 0, start)]

    while priority_queue:
        current_distance, current_delay, current_vertex = heapq.heappop(priority_queue)

        # Skip processing if current distance exceeds the maximum allowed delay
        if current_delay > max_delay:
            continue

        # If the current vertex is the target, break out of the loop
        if current_vertex == end:
            break

        # Iterate through the neighbors of the current vertex
        for neighbor in simple_graph.neighbors(current_vertex):
            if weighted:
                weight = simple_graph[current_vertex][neighbor]['perc_res']
            else:
                weight = 1
            distance = current_distance + weight
            delay = max((data['delay'] for u, v, data in graph.edges(data=True) if
                         u == current_vertex and v == neighbor and 'delay' in data), default=None)
            # graph[current_vertex][neighbor]['delay']
            d_delay = current_delay + delay

            # If a shorter path is found and within the delay constraint
            if distance < distances[neighbor] and d_delay <= max_delay:
                distances[neighbor] = distance
                delays[neighbor] = d_delay
                predecessors[neighbor] = current_vertex
                heapq.heappush(priority_queue, (distance, d_delay, neighbor))

    # Reconstruct the shortest path
    path = []
    while end is not None:
        path.append(end)
        end = predecessors[end]
    path.reverse()

    return path if path[-1] == sink else None


def calculate_total_delay_and_weight(multidigraph, digraph, path):
    """
    Calculate the sum of 'delay' attributes for a given path in a MultiDiGraph.

    Parameters:
    - multidigraph: An instance of a networkx.MultiDiGraph.
    - path: A list of tuples representing the path where each tuple is (u, v, key).

    Returns:
    - The sum of the 'delay' attributes on the path.
    """
    total_delay = 0
    weight = 0
    for u, v, key in path:
        edge_data = multidigraph.get_edge_data(u, v, key)
        link_data = digraph.get_edge_data(u, v)
        total_delay += edge_data.get('CBS').getProgDelay()
        # no processing at first hop
        if u != path[0][0]:
            total_delay += edge_data.get('CBS').getProcDelay()
        total_delay += edge_data.get('delay', 0)  # Assumes a default delay of 0 if not present
        weight += link_data.get('perc_res', 1)
    return total_delay, weight
