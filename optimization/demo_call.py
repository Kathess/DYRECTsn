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
from collections import defaultdict

import networkx as nx
from _decimal import Decimal
from dynamic_reservation.environment.default_topologies import example_topology
from dynamic_reservation.environment.flow import Flow
from dynamic_reservation.reservation.fixedDelayModel import init_topology, add_flow, remove_flow
from optimization.heuristics.pso import PSO


def print_flow_information(flow, results):
    """
    Pretty print for the flow information and the used slopes after each flow reservation.
    """
    print("Flow Details: Source: ", flow.source, ", Destination: ", flow.sinks, ", Bits: ", flow.data_per_interval,
          ", Interval: ", f'{flow.sending_interval * 1000:.3f}', "ms , Deadline: ", f'{flow.deadline * 1000:.3f}',
          "ms , User-Defined-Priority: ", flow.priority, ", Redundant Transm.:", flow.redundancy)
    if len(flow.paths) == 0:
        print('Flow not reserved!')
    else:
        for i in range(len(flow.paths)):
            print("Path (from, to, prio): ", flow.paths[i])
            # This is the delay in terms of the assigned delay budgets. This means that the flow will
            # never have a higher delay than the delay denoted here, regardless of future reservations:
            print(f'Reservation Independent Flow Delay: {flow.budget_path_delays[i] * 1000:.3f}ms')
            # This is the current NC delay, which is used to validate the delay budgets by (NC delay <= delay budgets):
            print(f'Reservation Dependent Flow Delay: {flow.path_delays[i] * 1000:.3f}ms')

    # Print slopes
    print("Hop (from, to): slope for highest prio, slope for second highest prio, ...")
    slopes = results['statistics']['used_idleSlopes']

    if not slopes:
        print("No Slopes set yet.")
    else:
        # Determine the maximum priority
        # max_priority = max(key[1] for key in slopes.keys())

        # Grouping values by (from, to) pair and initializing with None
        grouped_data = defaultdict(lambda: [None] * nrCBSqueues)
        for key, value in slopes.items():
            from_to = key[0]
            prio = key[1]
            grouped_data[from_to][prio] = value

        # Printing the values
        for from_to, values in grouped_data.items():
            values_str = ', '.join(str(f'{v:.2f}') if v is not None else 'None' for v in values)
            print(f"({from_to[0]}, {from_to[1]}) in bit/s: {values_str}")


if __name__ == "__main__":
    """
    General information:
    Everything is in bits and seconds, only the variable best_delay_bounds
    is in x*10microseconds. This will be explained later.
    
    The frame sizes need to include the header, the preamble and the IPG.
    The priority is inverse, meaning a TSN priority of 7 (=highest priority)
    is denoted as 0 in the following, 6 as 1 and so on.
    To reduce rounding errors, every float has to be converted to Decimal.
    
    Note on routing:
    Routing=1 means that the shortest path is selected.
    Routing=2 means that the utilization of the network is used as weight
    to balance the load within the network.
    
    Note on the delays: 
    The propagation and processing delay can be changed in
        dynamic_reservation/network_calculus/cbs.py
    in the functions
        - getProcDelay()
        - getProgDelay()
    Talkers do not add processing delays.
    Queuing and transmission delays are determined and upper bounded by the framework itself.
    
    Note on the fitness value:
    The fitness reflects 1) the number of flows (given for offline optimization) that can be reserved,
    2) the bandwidth that could remain free for future flows and 3) how close the flow deadlines are met 
    (as unnecessarily low delays for the high priorities affects the best effort traffic without need).
    Thereby, if the deadlines are met within a 30% range, that is considered perfect.
    (e.g., for a deadline of 1 ms, everything from 0.7ms-1ms is fine)
    The max. fitness is 1.
    
    Note on the Talker behavior:
    In the current implementation, the talkers are not considered to implement a credit-based shaper,
    in order to support legacy devices. Therefore, they don't have idleSlopes assigned.
    """

    # First define the topology, here is the example network, with 100 Mbit/s links
    # (100 Mbit/s links result in very limited capability to reserve flows with low delays,
    # so flows might get rejects rather easily)
    nrCBSqueues = 3  # should be between 1 and 6 in TSN
    graph, bridges, tsn_devices, cross_devices = example_topology(linkrate=Decimal(1.0e8), output=True,
                                                                  nrCBSqueues=nrCBSqueues)
    routing = 1  # 1=shortest path, 2=balanced network load

    flow_list = []
    # unicast flow with given priority
    flow1 = Flow(flowID='F1', source=22, sinks=[44], data_per_interval=1156, sending_interval=2500e-6,
                 deadline=1e-3, max_frame_size=1156, priority=1, frames_per_interval=1, redundancy=False)
    # multicast flow with two frames per interval, if priority=None, the framework derives the priority itself
    flow2 = Flow(flowID='F2', source=22, sinks=[33, 44, 222], data_per_interval=608, sending_interval=5000e-6,
                 deadline=1e-3, max_frame_size=608, priority=None, frames_per_interval=2, redundancy=False)
    # redundant flow using FRER with two frames per interval
    flow3 = Flow(flowID='F3', source=11, sinks=[44], data_per_interval=608, sending_interval=125e-6,
                 deadline=5e-3, max_frame_size=1216, priority=None, frames_per_interval=2, redundancy=False)
    # ------------------------------------ notes on preconfigured paths --------------------------------------
    # if you want the framework to define the paths, just remove the following path definitions!
    # definition can be done as [path1,path2,...] where path defines the hops (from, to, prio) in a list, e.g.:
    flow1.paths = [[(22, 2, 1), (2, 44, 1)]]  # unicast
    flow2.paths = [[(22, 2, 0), (2, 44, 0)],
                   [(22, 2, 0), (2, 3, 0), (3, 33, 0)],
                   [(22, 2, 0), (2, 3, 0), (3, 222, 0)]]  # multicast

    # redundant transmission needs more information, first the path:
    flow3.paths = [[(11, 1, 0), (1, 2, 0), (2, 3, 0), (3, 33, 0)], [(11, 1, 0), (1, 3, 0), (3, 33, 0)]]
    # and then at which nodes (node IDs) the frames should be duplicated and eliminated/merged:
    flow3.srf_config['start_node'] = 1 # after node 1, the frames are duplicated
    flow3.srf_config['end_node'] = 3 # at node 3, the frames are merged
    # note: in end-notes, non-preemptive strict priority scheduling is assumed,
    # where all flows have the highest priority
    flow_list.extend([flow2, flow1, flow3])
    max_best_effort_frame_size = Decimal('12240')

    # The paths can also be statically set in this way:
    # flow = Flow('myflow', sink=[44], ...)
    # flow.paths = [[(22, 2, 0), (2, 44, 0)]]
    # where each tuple defines (fromNode, toNode, priority_on_that_hop)

    # Now, the network is pre-configured with upper bounds on the queue delays.
    # (This is called MDM model of the publication.)
    # If you already know the delay bounds, you can skip this step, then set skip = True
    skip = True

    # The delay bounds are the only variable not given in seconds. Instead, they are
    # in 10 microseconds. So a delay bound of 2 is 20 microseconds. This is because
    # the delay bounds are set as discrete values with a step size of 10 microseconds.
    #
    # Delay bounds can be done uniformly (same for each bridge) or individually for each
    # queue in the network (individual_delays=True). The latter takes more time but can
    # potentially allow for more reservations. Define here:
    individual_delays = False

    # Other parameters which might be interesting:
    delay_bound_limits = (1, 500)  # min. and max. possible delay bound per queue (in x10 microseconds)
    converged_after = 20  # stop if solution doesn't improve for x steps
    stop_after = 50  # stop after at most x iterations
    particles = 300  # solutions evaluated in each step (to explore the search space)
    # if no_unused_queues=True, the only solutions are valid which use the highest priority queues
    # for example, in results like [26, 15, 1] the last queue is not required and could be ommitted
    # at network setup. no_unused_queues=False would also consider solutions such as [1, 1, 26] as valid
    no_unused_queues = True
    besteffort_perc = Decimal('0')  # (1-besteffort_perc) defines the maximum sum of idleSlopes at each port
    # do the flows have the same priority on their whole path (constant_priority = True ) or are they allowed
    # to switch at each hop (in TSN this can be done by using the Internal Priority Value):
    constant_priority = False

    print("----- Start of pre-configuration ----")
    # if desired, additional bandwidth can be reserved for potential future traffic by defining the
    # percentage of link rate that should remain free for future traffic of specific profiles;
    # the profiles are defined in:
    #   dynamic_reservation.environment.profiles
    # e.g., in this example, 10% of the link rate from 33 to 3 and 3 to 222 (each)
    # remains free for future profile 3 traffic. If this is desired, set like this:
    # per_link_rates = {(33, 3): [0, 0, 0.1, 0, 0]}
    # per_link_rates[(3, 222)] = [0, 0, 0.1, 0, 0]
    per_link_rates = None
    # only relevant if link capacity is reserved for future flows, provides an estimate for the
    # average path length of future flows:
    avg_path_length_future_flows = 2

    pso = PSO(flow_list, graph, nrCBSqueues, individual_delays, stop_after, converged_after, particles,
              delay_bound_limits, (-1 * delay_bound_limits[1], delay_bound_limits[1]),
              routing_offline=routing,
              max_frame_size=max_best_effort_frame_size,
              per_link_rates=per_link_rates,
              no_unused_queues=no_unused_queues,
              besteffort_perc=besteffort_perc,
              constant_priority=constant_priority,
              pathLength=avg_path_length_future_flows)

    if individual_delays and not skip:
        # individual delay budgets increase the search space significantly, therefore, the search is initialized
        # with the delay budgets of uniform network budgets (= same delay budgets at each port)
        best_delay_bounds, best_fitness = pso.particle_swarm_optimization_two_steps()
    elif not skip:
        best_delay_bounds, best_fitness = pso.particle_swarm_optimization()

    if skip:
        # optionally, you can set your own delay bounds like this
        if not individual_delays:
            best_delay_bounds = [26, 15, 1]
        else:
            best_delay_bounds = [26, 15, 1] * 32

    # Transform the delays to seconds (and in case of individual delays per queue, to dictionaries):
    best_delay_bounds = pso.get_delays_per_queue(best_delay_bounds)

    print("----- Pre-configuration finished ----")

    print("----- Start Online Admission ----")
    # Now we use the configuration results from the previous steps to actually reserve flows
    # (this represents the online admission control phase)

    # Initialize the network with the delay budgets from the offline optimization phase:
    if individual_delays:
        graph_state = init_topology(graph=graph, delay_per_queue=best_delay_bounds,
                                    besteffort_perc=besteffort_perc, max_packetsize=max_best_effort_frame_size,
                                    output=False, constant_priority=constant_priority)
    else:
        graph_state = init_topology(graph=graph, delays=best_delay_bounds,
                                    besteffort_perc=besteffort_perc, max_packetsize=max_best_effort_frame_size,
                                    output=False, constant_priority=constant_priority)

    # ------------------------- add time-triggered gates ---------------------------
    # add gate control list per hop - important: only one gate opening time (trivial calculation applied)
    # edge['gcl'] = (period, open time) in seconds
    # use the following syntax to add gates: dummy values for the output port at node 1 to node 3:
    # period = 500 microsec., gate open time = 50 microsec.
    nx.set_edge_attributes(graph_state['simple_graph'], {(1, 3): {'gcl': (Decimal(0.0005), Decimal(0.00005))}})
    nx.set_edge_attributes(graph_state['simple_graph'], {(3, 33): {'gcl': (Decimal(0.0005), Decimal(0.00005))}})
    # ------------------------------------------------------------------------------

    # Handle new flow requests, in this example, we use the same flows as during offline optimization
    i = 0
    successfully_reserved_flows = 0
    reserved_flows = []
    for flow in flow_list:
        results = add_flow(flow=flow, graph_state=graph_state, routing=routing)
        # Important, otherwise in the next add_flow call the graph state is not updated:
        graph_state = results['graph_state']
        print("----- ", i + 1, ". Flow Registration -----")
        print_flow_information(flow, results)
        reserved_flows = results['graph_state']['reserved_flows']
        i += 1
        if results['success']:
            successfully_reserved_flows += 1
        """
        Further information can be accessed via:
            results['statistics']
            flow.showReservation()
        or for FRER configurations:
            flow.printRedundancyConfig()
        """
    print("--------------------------------")
    print("Successfully reserved flows: ", successfully_reserved_flows, " of ", len(flow_list))
    print("Reserved Flows:", reserved_flows)

    # Optionally, here is how to remove flows:
    flow = flow_list[-1]
    if flow.flowID in results['graph_state']['reserved_flows']:
        results_del = remove_flow(flow=flow, graph_state=graph_state)
        print_flow_information(flow, results_del)
        reserved_flows = results_del['graph_state']['reserved_flows']
    print("Reserved Flows After Removal:", reserved_flows)
    print("------- Finished -------")
