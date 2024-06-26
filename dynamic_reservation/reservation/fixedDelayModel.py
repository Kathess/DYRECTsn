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
import copy
import heapq
import itertools
from itertools import groupby
from typing import List, Tuple
import networkx as nx
import numpy as np
from _decimal import Decimal
from networkx import MultiDiGraph
from dynamic_reservation.environment.flow import Flow
from dynamic_reservation.network_calculus import cbs
from dynamic_reservation.network_calculus.arithmetic_helper_functions import get_tsn_arrival, \
    find_first_matching_index, find_identical_elements, merge_arrays_with_indexes, modify_arrays_and_get_indexes
from dynamic_reservation.reservation.routing import dijkstra_with_delay, calculate_total_delay_and_weight

out = False


def init_topology(graph: MultiDiGraph,
                  besteffort_perc=Decimal('0'),
                  delays=None,
                  delay_per_queue=None,
                  max_packetsize=Decimal('12336'),
                  constant_priority=False,
                  output=False):
    # global variable for output
    global out
    out = output

    delay_per_queue = delay_per_queue if delay_per_queue is not None else {}  # default value for delays
    delays = delays if delays is not None else []  # default value for delays

    simple_graph = nx.DiGraph(graph)

    if out:
        print("---------------------- Start Initialization ------------------------")

    # -----------------------------  Assign everything we know to graph  ----------------------------------

    for (u, v, i) in graph.edges:
        curr_edge = graph[u][v][i]
        curr_edge['flows'] = []  # which flows are reserved at this edge?
        # 1. Create CBS object (arrival and service/idleSlope will be updated every time a flow is reserved)
        curr_edge['CBS'] = cbs.CBS_system(queue_id='({},{},{})'.format(u, v, i),
                                          outputqueues=graph.number_of_edges(u, v),
                                          linkrate=curr_edge['rate_out'],
                                          queuesize=curr_edge['queue_in'],
                                          be_slope=besteffort_perc * curr_edge['rate_out'],
                                          maxpacket_be=max_packetsize)

        # 2. The following is used, if the routig algorithm uses the utilization as cost function
        curr_simple_link = simple_graph[u][v]
        curr_simple_link['reserved'] = Decimal('0')
        if (Decimal('1') - besteffort_perc) == Decimal('0'):
            curr_simple_link['perc_res'] = Decimal('inf')
        else:
            curr_simple_link['perc_res'] = Decimal('1')

        # 3. Assign delay budgest to each edge
        if not delay_per_queue:
            curr_edge['delay'] = delays[i]
        else:
            if delay_per_queue.get((u, v)) is None:
                curr_edge['delay'] = Decimal('0')
            else:
                curr_edge['delay'] = delay_per_queue.get((u, v))[i]

        if out:
            print("Edge ({},{},{}): Delay: {}".format(u, v, i, curr_edge['delay']))

    if out:
        print("---------------------- End Initialization ------------------------")

    graph_state = {'graph': graph, 'simple_graph': simple_graph, 'max_priority': find_max_priority(graph),
                   'constant_prio': constant_priority, 'reserved_flows': []}
    return graph_state


# Function to find the maximum priority value
def find_max_priority(G):
    max_priority = None
    for u, v, attrs in G.edges(data=True):
        if 'priority' in attrs:
            if max_priority is None or attrs['priority'] > max_priority:
                max_priority = attrs['priority']
    return max_priority + 1


def add_flow(flow,
             graph_state,
             routing=2):
    """
    This method tries to reserve all flows of 'flow_list' in the network 'graph' using the given delay budgets values
    (either 'delays' or 'delay_per_queue').

    All frame sizes are bits and all times are seconds.

    @param flow: Flow to reserve. Some flows might have a priority, some not.
    @param graph_state: dictionary as defined for the current reservations
    @param routing: 1 means the shortest path (cost = path length)
                    and 2 means balanced network (cost = remaining bandwidth)
    @return:     results with
                    results['success']
                    results['graph_state']
                            graph_state['graph']: MultiDiGraph
                            graph_state['simple_graph']: DiGraph
                            graph_state['reserved_flows']: List with Flow IDs
                            graph_state['constant_prio']: Bool -> Flows change their prio?
                            graph_state['max_priority']: int
                    results['statistics']
                            all as Dictionary with (from, to): [value_prio_1, value_prio_2,...]
                            statistics['used_rate']
                            statistics['used_idleSlopes']
                            statistics['used_backlogs']
    """
    global out

    assert routing in [1, 2], "Routing method must be 1 or 2."

    # ---------------------------------------  Initialize  ---------------------------------------------

    assert flow.flowID not in graph_state[
        'reserved_flows'], "Flows must be unique in the network (error for flowID {})".format(
        flow.flowID)

    success = True

    if out:
        print("------------------- Find Path(s) ----------------------")
        print(flow)

    # 1. Find paths for this flow (with delay budgets and enough resources)
    paths, slopes, arrivals = find_path(graph_state, flow, routing)

    if paths is None:
        paths = []
        success = False
        if out:
            print("Path not found: {}".format(paths))

    if out:
        print("Path found: {}".format(paths))
        print("------------------- Reserve Resources ----------------------")

    # 2. If successful, reserve resources for flow on paths
    for i in range(len(paths)):
        reserve_flow(graph_state, flow, paths[i], slopes[i], arrivals[i])

    if flow.redundancy and len(paths) > 1:
        # add nodes after eliminating device again
        path1, path2 = merge_lists(paths[0], paths[1], flow)
        paths = [path1, path2]

    flow.paths = paths

    # ----------------------------------  Calculate statistics  -------------------------------------
    results = {'success': success}
    if success:
        graph_state['reserved_flows'].append(flow.flowID)
    statistics = get_statistics(graph_state['graph'], flow, success)
    results['graph_state'] = graph_state
    results['statistics'] = statistics

    return results


def merge_lists(list1, list2, flow):
    # which list has the complete path?
    if list1[-1][1] == flow.sinks[0]:
        complete_list = list1
        other_list = list2
    else:
        complete_list = list2
        other_list = list1
    # when to merge?
    merge_node = other_list[-1][1]
    for i in range(len(complete_list)):
        if complete_list[i][1] == merge_node:
            other_list.extend(complete_list[(i+1):])
            break

    return complete_list, other_list

def clear_del_flow(graph_state, path):
    for edge in path:
        e0 = edge[0]
        e1 = edge[1]
        e2 = edge[2]
        curr_edge = graph_state['graph'][e0][e1][e2]
        if 'del_flow' in curr_edge['flows']:
            curr_edge['flows'].remove('del_flow')


def remove_flow(flow,
                graph_state):
    global out

    # ---------------------------------------  Initialize  ---------------------------------------------

    assert flow.flowID in graph_state[
        'reserved_flows'], "Flow must be reserved in the network before removal (error for flowID {})". \
        format(flow.flowID)

    success = True

    flow.data_per_interval *= -1
    flow.max_frame_size *= -1
    tmp_flowID = flow.flowID
    flow.flowID = 'del_flow'

    # Get the curves for the flow on its path
    slopes, arrivals = [], []
    for path in flow.paths:
        reservation, slopes_path, arrivalcurves_paths = check_path(graph_state, path, flow)
        slopes.append(slopes_path)
        arrivals.append(arrivalcurves_paths)

    if out:
        print("------------------- Remove Resources ----------------------")

    # 2. remove flow arrival from path
    for i in range(len(flow.paths)):
        reserve_flow(graph_state, flow, flow.paths[i], slopes[i], arrivals[i], removeFlowID=tmp_flowID)

    # 3. clear paths
    for path in flow.paths:
        clear_del_flow(graph_state, path)

    # ----------------------------------  Calculate statistics  -------------------------------------

    statistics = get_statistics(graph_state['graph'], flow, success)

    graph_state['reserved_flows'].remove(tmp_flowID)

    results = {'success': success, 'graph_state': graph_state, 'statistics': statistics}

    flow.data_per_interval *= -1
    flow.flowID = tmp_flowID
    flow.resetReservationInfo()

    return results


def find_path(graph_state, flow: Flow, routing=2):
    paths = []
    path_slopes = []
    arrival_curves = []

    graph = graph_state['graph']
    simple_graph = graph_state['simple_graph']

    assert not (flow.redundancy and len(flow.sinks) > 1), \
        "A flow cannot be multicast and use redundancy (flowID: {}).".format(flow.flowID)

    # --------------------------------- Paths already given ----------------------------------
    if len(flow.paths) > 0:
        # if path is already given, just check reservation:
        success = True

        for priority_path in flow.paths:
            reservation, slopes, arrivalcurves = check_path(graph_state, priority_path, flow)
            path_slopes.append(slopes)
            arrival_curves.append(arrivalcurves)
            if not reservation:
                success = False
                if out:
                    print("Given path does not allow for reservation: {}".format(priority_path))
        if success:
            return flow.paths, path_slopes, arrival_curves
        else:
            return None, None, None

    # ---------------------------------- Paths need to be found --------------------------------
    else:
        # otherwise find paths
        all_sinks = flow.sinks.copy()

        K = 2  # for k-shortest-path algorithm
        N = 2  # of redundant paths checked

        # get the first path with delay constraints
        P = []  # List of shortest paths
        B = []  # Heap of paths

        # Insert the initial path into the heap
        heapq.heappush(B, (0, 0, [flow.source]))  # (weight, delay, path)

        while B:
            C, D, path = heapq.heappop(B)
            u = path[-1]

            if u in all_sinks:
                P.append((C, D, path))
                # check first path
                priority_path, slopes, arrivals = check_reservation(graph_state, path, flow)
                # if no disjoint path is needed, we are done once we find the first of K successfully
                if priority_path:
                    paths.append(priority_path)
                    path_slopes.append(slopes)
                    arrival_curves.append(arrivals)
                    if not flow.redundancy:
                        all_sinks.remove(u)
                        if len(all_sinks) == 0:
                            break
                        else:
                            continue
                    else:
                        # if we have a redundant flow, we have to search for a second path once we find the first
                        # using the edge disjoint shortest path algorithm

                        # maybe, we cannot find fully disjoint paths
                        # then we try to change the start and end of the disjoint path
                        # Iterate over each node in the path but keep the order
                        pairs_with_distance = []
                        # Generate pairs along with the distance between them
                        for i in range(len(path)):
                            for j in range(i + 1, len(path)):
                                distance = j - i  # Calculate the distance
                                pairs_with_distance.append((path[i], path[j], distance))

                        # Sort the pairs based on distance, in descending order
                        pairs_with_distance.sort(key=lambda x: x[2], reverse=True)

                        # Extract just the pairs, without the distance, for final output
                        sorted_pairs = [(pair[0], pair[1]) for pair in pairs_with_distance]

                        # Print sorted pairs
                        for pair in sorted_pairs:
                            start = pair[0]
                            end = pair[1]
                            # Create a copy of the graph
                            G_copy = simple_graph.copy()
                            # path to be removed to get disjoint paths:
                            remove_path = path.copy()
                            for i in range(N):  # N defines the max. number of paths that we try out
                                # Remove the edges of the first path from the graph copy
                                for j in range(len(remove_path) - 1):
                                    G_copy.remove_edge(remove_path[j], remove_path[j + 1])

                                if not nx.has_path(G_copy, start, end):
                                    # no redundant path possible, so try next pair
                                    break

                                path_disjoint = dijkstra_with_delay(graph, G_copy, start, end, flow.deadline,
                                                                    routing)
                                # create full path again
                                new_path = []
                                merge = True
                                for hop in path:
                                    if hop == path_disjoint[-1]:
                                        merge = True
                                    if merge:
                                        new_path.append(hop)
                                    if hop == path_disjoint[0]:
                                        merge = False
                                        sliced_path = path_disjoint[1:-1]
                                        for hop_new in sliced_path:
                                            new_path.append(hop_new)

                                priority_path, slopes, arrivals = check_reservation(graph_state, new_path,
                                                                                    flow,
                                                                                    redundant_flow=[paths[0],
                                                                                                    arrival_curves[0]])
                                remove_path = path_disjoint
                                # if we found a second path, stop
                                if priority_path:
                                    # after the merging, we only use the second path parameters (for arrival,
                                    # slopes, path) to make sure that we only reserve once, so we need to adapt the
                                    # first reservation values
                                    identical_indexes = find_identical_elements(paths[0], priority_path)
                                    # remove the identical_indexes before merging:
                                    if identical_indexes[-1] != len(paths[0])-1:
                                        identical_indexes = []
                                    else:
                                        last_element = identical_indexes[-1]
                                        identical_indexes = [num for num in identical_indexes if
                                                              num == last_element - 1 or num == last_element]

                                    for index in sorted(identical_indexes, reverse=True):
                                        del paths[0][index]
                                        del path_slopes[0][index]
                                        del arrival_curves[0][index]
                                    paths.insert(0, priority_path)
                                    path_slopes.insert(0, slopes)
                                    arrival_curves.insert(0, arrivals)
                                    flow.srf_config['start_node'] = path_disjoint[0]
                                    flow.srf_config['end_node'] = path_disjoint[-1]
                                    return paths, path_slopes, arrival_curves
                        return None, None, None
                # if not successful, find next path, but if we search K paths, then return unsuccessfull
                elif not priority_path and len(P) < K:
                    continue
                else:
                    return None, None, None

            for v in simple_graph.neighbors(u):
                # Check if edge exists and calculate new cost
                if v not in path:  # simple_graph.has_edge(u, v):
                    if routing == 1:
                        new_cost = C + 1
                    else:
                        new_cost = C + simple_graph[u][v]['perc_res']
                    # add max. delay on that link:
                    new_delay = D + max((data['delay'] for ux, vx, data in graph.edges(data=True) if
                                         ux == u and vx == v and 'delay' in data), default=None)
                    if new_delay <= flow.deadline:
                        new_path = path + [v]
                        heapq.heappush(B, (new_cost, new_delay, new_path))

        if len(all_sinks) == 0:
            return paths, path_slopes, arrival_curves
        else:
            return None, None, None


def check_reservation(graph_state, path: List[Tuple], flow: Flow, redundant_flow=None):
    # define path priorities

    graph = graph_state['graph']
    simple_graph = graph_state['simple_graph']

    combined_list = []
    path = list(zip(path, path[1:]))
    # if redundant_flow, then the redundant parts need the same priortiy
    given_prios, indexes = None, None
    if redundant_flow:
        given_prios, remaining_prios, indexes = modify_arrays_and_get_indexes(path, redundant_flow[0][1:])
        path = remaining_prios

    if not flow.priority:
        prios = set(range(graph_state['max_priority']))  # set(range(graph.number_of_edges(path[0][0], path[0][1])))
        if not graph_state['constant_prio']:
            combinations = list(itertools.product(prios, repeat=len(path)))
            for p in combinations:
                tmp_path = []
                for i in range(len(p)):
                    tmp_path.append((*path[i], p[i]))
                if redundant_flow:
                    tmp_path = merge_arrays_with_indexes(tmp_path, given_prios, indexes)
                path_exists = does_path_exist(graph, tmp_path)
                if path_exists:
                    delay, weight = calculate_total_delay_and_weight(graph, simple_graph, tmp_path)
                    if delay <= flow.deadline:
                        combined_list.append((tmp_path, delay))
        else:
            for prio in prios:
                tmp_path = []
                for i in range(len(path)):
                    tmp_path.append((*path[i], prio))
                if redundant_flow:
                    tmp_path = merge_arrays_with_indexes(tmp_path, given_prios, indexes)
                path_exists = does_path_exist(graph, tmp_path)
                if path_exists:
                    delay, weight = calculate_total_delay_and_weight(graph, simple_graph, tmp_path)
                    if delay <= flow.deadline:
                        combined_list.append((tmp_path, delay))
    else:
        tmp_path = []
        for edge in path:
            tmp_path.append((edge[0], edge[1], flow.priority))
        if redundant_flow:
            tmp_path = merge_arrays_with_indexes(tmp_path, given_prios, indexes)
        path_exists = does_path_exist(graph, tmp_path)
        if path_exists:
            delay, weight = calculate_total_delay_and_weight(graph, simple_graph, tmp_path)
            if delay <= flow.deadline:
                combined_list.append((tmp_path, delay))

    combined_list = sorted(combined_list, key=lambda x: -x[1])

    for path in combined_list:
        priority_path = path[0]
        # path_exists = does_path_exist(graph, priority_path)
        # reservation = False
        # if path_exists:
        reservation, slopes, arrivals = check_path(graph_state, priority_path, flow, redundant_flow=redundant_flow)
        if reservation:
            return priority_path, slopes, arrivals
    return None, None, None


# Function to check if a path exists with the given attributes
def does_path_exist(G, path):
    for i in range(len(path) - 1):
        start, end, priority = path[i]
        if not G.has_edge(start, end, key=priority):
            return False
    return True


def get_index_or_none(my_list, element):
    try:
        return my_list.index(element)
    except ValueError:
        return -1  # or return None if you prefer


def check_path(graph_state, path: List[Tuple], flow: Flow, redundant_flow=None) \
        -> Tuple[bool, List[Decimal], List[Tuple[Decimal, Decimal, Decimal]]]:
    """
    check if we can set a idle slope to keep the delays with the new arrival curve in all ports
    """
    curr_arrival = []  # keep track of flow's arrival curve
    input_port = -1  # and from which link I'm coming for link shaping
    # we shape according to the Idle Slope for the last queue = linkrate_graph - BE (others 0)
    shaperrate = -1
    shaperburst = -1
    # -1 if the previous link did not shape
    # (this is only the case for the first hop after the sending device, where we only have link shaping)
    path_slopes = []  # the slopes that we calculated at each hop for this path (we reserve them in the next function)
    path_arrivals = []  # the arrival curves at each, so we don't need to re-calculate them
    sink = path[-1][1]

    graph = graph_state['graph']

    for edge in path:
        e0 = edge[0]
        e1 = edge[1]
        e2 = edge[2]
        curr_link = graph[e0][e1]
        curr_edge = curr_link[e2]
        # for first node:
        if e0 == flow.source:
            curr_arrival, flowrate = get_tsn_arrival(C=curr_edge['rate_out'], flow=flow)
            path_arrivals.append(curr_arrival.copy())

            CBS = curr_edge['CBS']

            sum_arrival = CBS.addarrivalcurve(virtual=True,
                                              arrival=curr_arrival,
                                              priority=e2,
                                              flow_packetsize=flow.max_frame_size,
                                              port=-1,
                                              prevlinkrate=-1,
                                              shaperrate=-1,
                                              shaperburst=-1)

            if sum_arrival[-1][2] > curr_edge['rate_out']:
                if out:
                    print(
                        "Link rate at talker {} surpassed: {} > {}.".format(edge, sum_arrival[-1][2],
                                                                            curr_edge['rate_out']))
                return False, [], []

            delay = CBS.get_talker_delay(sum_arrival)
            if delay > curr_edge['delay']:
                if out:
                    print(
                        "Reservation not possible because delay at talker is surpassed: [{}, {}]".format(edge, flow))
                return False, [], []

            new_slopes = [Decimal('0')] * CBS.outputqueues

            # we assume no backlog limit at the talker

        else:
            merge_hop_index = -1 if not redundant_flow else get_index_or_none(redundant_flow[0], edge)
            prev_disjoint_edge = None if merge_hop_index <= 0 else redundant_flow[0][merge_hop_index - 1]
            # bigger zero because it cant be the first element anyway
            if merge_hop_index > 0 and prev_disjoint_edge != path[len(path_arrivals) - 1]:
                # also add the redundant flow
                graph_disjoint_edge = graph[prev_disjoint_edge[0]][prev_disjoint_edge[1]][prev_disjoint_edge[2]]
                replication_node = find_first_matching_index(path, redundant_flow[0])
                arrival_before_replication = redundant_flow[1][replication_node].copy()
                path_delay_delta = get_delay_difference(path, redundant_flow[0], graph, flow.max_frame_size)
                sum_arrival = curr_edge['CBS'].addredundantflows(virtual=True, priority=e2,
                                                                 arrival_1=redundant_flow[1][merge_hop_index].copy(),
                                                                 prevlinkrate_1=graph_disjoint_edge['rate_out'],
                                                                 arrival_2=curr_arrival,
                                                                 prevlinkrate_2=
                                                                 graph[input_port[0]][input_port[1]][input_port[2]]
                                                                 ['rate_out'],
                                                                 path_delay_delta=path_delay_delta,
                                                                 flow_packetsize=flow.max_frame_size,
                                                                 arrival_before_replication=arrival_before_replication)
            else:
                # B1. add arrival curve
                sum_arrival = curr_edge['CBS'].addarrivalcurve(virtual=True,
                                                               arrival=curr_arrival,
                                                               priority=e2,
                                                               flow_packetsize=flow.max_frame_size,
                                                               port=input_port,
                                                               prevlinkrate=
                                                               graph[input_port[0]][input_port[1]][input_port[2]]
                                                               ['rate_out'],
                                                               shaperrate=shaperrate,
                                                               shaperburst=shaperburst)

            # B2. und A2. check that sum of all arrivalrates <= C - BE
            rate_out = curr_edge['rate_out']
            CBS = curr_edge['CBS']
            be_slope = CBS.be_slope

            # for all other nodes
            # B3. set I (I >= rate of arrival)
            tmp_packetsizes = copy.deepcopy(CBS.all_packetsizes)
            if flow.flowID == 'del_flow':
                tmp_packetsizes[e2].remove(flow.max_frame_size * -1)
            new_packetsizes = [max(tmp_packetsize, default=Decimal('0')) for tmp_packetsize in tmp_packetsizes]
            if new_packetsizes[e2] < flow.max_frame_size:
                new_packetsizes[e2] = flow.max_frame_size

            new_slope = CBS.slope_for_delay(arrival=sum_arrival,
                                            maxpackets=new_packetsizes,
                                            delay=curr_edge['delay'], queue=e2)

            allowed_rate = rate_out - be_slope
            if new_slope < 0 or new_slope > allowed_rate:
                if out:
                    print(
                        "Reservation not possible because no Idle Slope can guarantee this delay "
                        "at edge {} for flow {}.".format(edge, flow))
                return False, [], []

            # B4. check backlog
            # we tell the backlog calculation the new slopes
            new_slopes = CBS.getidleslopes()
            new_slopes[e2] = new_slope
            if not e0 == flow.source:
                if flow.data_per_interval > 0:
                    max_backlog = CBS.getbacklog(queue=e2,
                                                 arrival=sum_arrival,
                                                 slopes=new_slopes)
                    if max_backlog > curr_edge['queue_in']:
                        if out:
                            print(
                                "At edge {}, the backlog is too high (for flow {}). Queuesize: {}, Backlog: {}.".format(
                                    edge, flow, curr_edge['queue_in'], max_backlog))
                        return False, [], []

            # B5. for lower priorities:
            for i in curr_link:
                # Hint: we don't check that they are ordered
                # all priorities below e2 will be left as they are and e2 will be set to new_slope
                # the higher slopes are successively calculated and the reversely update the slope values in ALL edges
                if i > e2:
                    # lower_prios.append(curr_link[i])
                    queue = curr_link[i]
                    queueCBS = queue['CBS']
                    prio = queue['priority']
                    # calculate new idleslope for this priority
                    if new_slopes[prio] > 0:  # but only if something has been reserved
                        lower_slope = queueCBS.slope_for_delay(arrival=None,
                                                               delay=queue['delay'],
                                                               queue=prio,
                                                               slopes=new_slopes,
                                                               maxpackets=new_packetsizes)
                        if lower_slope < 0:
                            if out:
                                print("Reservation not possible at lower priority queue {} "
                                      "for flow {}.".format(queue, flow))
                            return False, [], []
                        new_slopes[prio] = lower_slope

                        #   B5.2 check backlog
                        # check backlog as well, since T is now different
                        max_backlog = queueCBS.getbacklog(queue=prio, slopes=new_slopes)
                        if max_backlog > queue['queue_in']:
                            if out:
                                print(
                                    "At lower priority {}, the backlog is too high (for flow {}). Queuesize: {},"
                                    " Backlog: {}.".format(queue, flow, queue['queue_in'], max_backlog))
                            return False, [], []

            # lower_prios.sort(key=lambda t: t['priority'])

            # B6. check that sum(I) <= C- BE
            if sum(new_slopes) > allowed_rate:
                if out:
                    print(
                        "Remaining rate at edge {} not enough (for flow {}). rate: {}, sum(new_slopes): {}".format(
                            edge, flow, allowed_rate, sum(new_slopes)))
                return False, [], []

        if not e1 == sink:
            # not for last node
            # B7. increase arrival curve

            # caluclate the new output curve for the current flow (assuming that they experience the max. delay)
            # only if interval < deadline
            increase = True
            if flow.sending_interval >= flow.deadline:
                increase = False
            curr_arrival, shaperrate, shaperburst = CBS.arrivaloutput(
                queue=e2,
                arrival=curr_arrival.copy(),
                delay=curr_edge['delay'],
                burstinessincrease=increase)
            # since we need the maximum shaper, we need to calculate shaperrate and shaperburst separately
            shaperrate, shaperburst = CBS.maxShaper()
            path_arrivals.append(curr_arrival)

        # A3. & B8. update input port
        input_port = edge
        path_slopes.append(new_slopes)

    return True, path_slopes, path_arrivals


def reserve_flow(graph_state, flow: Flow, path: List, newslopes,
                 arrival: List[Tuple[Decimal, Decimal, Decimal]], removeFlowID=None) -> None:
    """
    With the given path, we are sure that the reservation is possible, so we need to update the
    arrival curve and slope parameters.
    :param newslopes: the slopes that we calculated and that need to be set
    """
    # keep track of flow's arrival curve
    curr_arrival = []
    # and from which link I'm coming
    input_port = -1
    shaperrate = -1
    shaperburst = -1
    hop_nr = 0  # (so we assign the right slope values)
    sink = path[-1][1]

    graph = graph_state['graph']
    simple_graph = graph_state['simple_graph']

    for edge in path:
        e0 = edge[0]
        e1 = edge[1]
        e2 = edge[2]
        curr_link = graph[e0][e1]
        curr_edge = curr_link[e2]
        curr_link_simple = simple_graph[e0][e1]
        # for FRER elimination, we need to keep track when the flows are merged again
        if flow.flowID not in curr_edge['flows'] and removeFlowID is None:
            # if the flow is not in the edge, just reserve it
            curr_edge['flows'].append(flow.flowID)
        else:
            if removeFlowID:
                if removeFlowID in curr_edge['flows']:
                    curr_edge['flows'].remove(removeFlowID)
                else:
                    # already removed from that edge (e.g. due to FRER or multicast)
                    input_port = edge
                    hop_nr += 1
                    continue
            if flow.flowID not in curr_edge['flows']:
                curr_edge['flows'].append(flow.flowID)
            else:
                # if the flow already is in the edge, it is either before duplication / a multicast flow
                # - for this, check if previous hop also contains the flow
                # or it is a eliminating device
                # - if the previous hop doesn't contain the flow
                if input_port == -1 \
                        or (isinstance(input_port, tuple)
                            and flow.flowID
                            in graph[input_port[0]][input_port[1]][input_port[2]]['flows']):
                    # it is multicast / before duplication, so don't add anything
                    input_port = edge
                    hop_nr += 1
                    continue

        if e0 == flow.source:
            # for first node:
            # A1. add arrivalcurve
            # A2. update input port
            curr_arrival = arrival[hop_nr]
            CBS = curr_edge['CBS']

            CBS.addarrivalcurve(port=-1,
                                arrival=curr_arrival,
                                priority=e2,
                                flow_packetsize=flow.max_frame_size,
                                prevlinkrate=-1,
                                shaperrate=-1,
                                shaperburst=-1)

            # B2. set I's
            for i in curr_link:
                if not flow.flowID == "del_flow":
                    curr_link[i]['CBS'].setpacketsize(queue=e2, size=flow.max_frame_size)
                # for the first hop, we consider the slopes to be the flow rates
                curr_link[i]['CBS'].setidleslopes(newslopes[hop_nr])
                curr_link[i]['CBS'].all_packetsizes = CBS.all_packetsizes

            input_port = edge

        if not e0 == flow.source:
            # for other nodes:
            # B1. add arrivalcurve
            # B2. set I's
            # not for last hop
            # B3. increase curve
            # B4. update input port

            CBS = curr_edge['CBS']
            curr_arrival = arrival[hop_nr]
            CBS.addarrivalcurve(port=input_port,
                                arrival=curr_arrival,
                                priority=e2,
                                flow_packetsize=flow.max_frame_size,
                                prevlinkrate=graph[input_port[0]][input_port[1]][input_port[2]]['rate_out'],
                                shaperrate=shaperrate,
                                shaperburst=shaperburst)

            # B2. set I's
            for i in curr_link:
                if not flow.flowID == "del_flow":
                    curr_link[i]['CBS'].setpacketsize(queue=e2, size=flow.max_frame_size)
                curr_link[i]['CBS'].setidleslopes(newslopes[hop_nr])
                curr_link[i]['CBS'].all_packetsizes = CBS.all_packetsizes

            if not e1 == sink:
                # since we need the maximum shaper, we need to calculate shaperrate and shaperburst separately
                shaperrate, shaperburst = CBS.maxShaper()
                input_port = edge

            newslopessum = sum(newslopes[hop_nr])
            curr_link_simple['reserved'] = newslopessum
            tmp_res = newslopessum / (curr_edge['CBS'].linkrate - curr_edge['CBS'].be_slope)
            tmp_rem = 1 - tmp_res
            if tmp_rem <= 0:
                curr_link_simple['perc_res'] = Decimal('inf')
            else:
                curr_link_simple['perc_res'] = Decimal('1') / tmp_rem
        hop_nr += 1


def get_statistics(graph, flow, success):
    used_rate = {}
    used_backlogs = {}
    used_slopes = {}

    # ----------------------  Calculate statistics for the whole network  -------------------------

    # Calculate remaining rate (without BE), used rate, used slopes, and used backlog per hop
    for key, group in groupby(nx.get_edge_attributes(graph, 'CBS'), lambda x: (x[0], x[1])):
        rate_sum = Decimal('0')
        backlog_sum = Decimal('0')
        for edge in group:
            e2 = edge[2]
            curr_edge = graph[edge[0]][edge[1]][e2]
            if curr_edge['CBS'].arrivalpoints:
                rate_sum += curr_edge['CBS'].arrivalpoints[-1][2]
            if curr_edge['CBS'].getidleslopes()[e2] > 0:
                backlog_sum += curr_edge['CBS'].getbacklog(e2)
                used_slopes[(key, e2)] = curr_edge['CBS'].getidleslopes()[e2]
        used_rate[key] = rate_sum
        used_backlogs[key] = backlog_sum

    statistics = {'used_rate': used_rate, 'used_idleSlopes': used_slopes, 'used_backlogs': used_backlogs}

    # -------------------------  Calculate statistics for the flow  ----------------------------
    if success:
        for path in flow.paths:
            # calculate real E2E delay of path (not max. delay at hop)
            path_delay = Decimal('0')
            budget_path_delay = Decimal('0')
            for edge in path:
                e0 = edge[0]
                e1 = edge[1]
                e2 = edge[2]
                curr_edge = graph[e0][e1][e2]

                budget_path_delay += curr_edge['delay']
                budget_path_delay += curr_edge['CBS'].getProgDelay()
                path_delay += curr_edge['CBS'].getProgDelay()
                # no processing at first hop
                if e0 != flow.source:
                    budget_path_delay += curr_edge['CBS'].getProcDelay()
                    path_delay += curr_edge['CBS'].getProcDelay()
                    # transmission and queueing delay for CBS devices
                    path_delay += curr_edge['CBS'].getdelay(e2)
                else:
                    # the first link is only link shaped
                    # transmission and queueing delay for talker devices
                    path_delay += curr_edge['CBS'].get_talker_delay()

            flow.path_delays.append(path_delay)
            flow.budget_path_delays.append(budget_path_delay)

        # --------------------  Calculate configurations for redundantly transmitted flows  --------------------------
        if flow.redundancy:
            path1 = flow.paths[0]
            path2 = flow.paths[1]
            redundancy_start = flow.srf_config['start_node']
            redundancy_end = flow.srf_config['end_node']
            path1 = path1[[x[0] for x in path1].index(redundancy_start):[x[1] for x in path1].index(redundancy_end) + 1]
            path2 = path2[[x[0] for x in path2].index(redundancy_start):[x[1] for x in path2].index(redundancy_end) + 1]
            delta_d = get_delay_difference(path1, path2, graph, flow.max_frame_size)

            # get the configurations:
            if flow.sending_interval > delta_d and flow.frames_per_interval == 1:
                flow.srf_config['match_recovery'] = True
            else:
                flow.srf_config['match_recovery'] = False
                if flow.frames_per_interval > 1:
                    flow.srf_config['history_length'] = flow.frames_per_interval * \
                                                        np.floor(delta_d / flow.sending_interval + 2)
                else:
                    flow.srf_config['history_length'] = delta_d / flow.sending_interval + 1
            flow.srf_config['reset_timer'] = delta_d + flow.sending_interval

    return statistics


def get_delay_difference(path1, path2, graph, max_frame_size):
    min_delay = [0, 0]
    max_delay = [0, 0]
    for edge in path1:
        e0 = edge[0]
        e1 = edge[1]
        e2 = edge[2]
        curr_edge = graph[e0][e1][e2]
        max_delay[0] += curr_edge['delay']
        min_delay[0] += max_frame_size / curr_edge['rate_out']
    for edge in path2:
        e0 = edge[0]
        e1 = edge[1]
        e2 = edge[2]
        curr_edge = graph[e0][e1][e2]
        max_delay[1] += curr_edge['delay']
        min_delay[1] += max_frame_size / curr_edge['rate_out']
    return max(max_delay) - min(min_delay)
