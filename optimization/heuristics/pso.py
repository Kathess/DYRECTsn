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
import itertools
import operator
import random
import string
from time import time

import numpy as np
from _decimal import Decimal

from dynamic_reservation.environment.flow import Flow
from dynamic_reservation.environment.profiles import profiles
from dynamic_reservation.reservation.fixedDelayModel import add_flow, init_topology


class PSO:
    def __init__(self, flow_list, graph, nrPrios, per_hop, num_iterations, converged, num_particles,
                 search_space, vel_space, w=0.5, c1=2, c2=2.4, pathLength=1,
                 addFlowsAfter=5, wReserved=0.9, wAdditional=0.09, wDeadline=0.01,
                 routing_online=2, routing_offline=1, per_link_rates=None, max_frame_size=Decimal('12336'),
                 no_unused_queues=False, besteffort_perc=Decimal('0'), constant_priority=False):
        """
        Best Parameter: w=0.5, c1=2, c2=2.4
        """
        self.graph = graph
        self.flow_list = flow_list
        if per_link_rates:
            print("Bandwidth for additional flows will be considered if the fitness is > 0.99 or"
                  "if the solutions converged for", addFlowsAfter, "steps.")
        self.per_link_rates = per_link_rates
        self.per_hop = per_hop
        self.nrPrios = nrPrios
        self.max_frame_size = max_frame_size
        self.besteffort_perc = besteffort_perc
        self.constant_priority = constant_priority

        # Set the number of particles and iterations (where global best does not change, before stopping)
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.converged = converged
        self.no_unused_queues = no_unused_queues
        # if nothing changes for x steps, then we start evaluating the additional flows (to speed
        # up the early convergence)
        self.addFlowsAfter = addFlowsAfter
        self.pathLength = pathLength

        # Define the search space and velocity space for each solution
        self.conversion_factor = 100000
        self.search_space = search_space
        self.vel_space = vel_space

        # Define the learning parameters (inertia weight, cognitive, and social component)
        self.w = w  # must be smaller than 1
        self.c1 = c1  # typically between [1, 3]
        self.c2 = c2  # typically between [1, 3]

        # weights for reward function
        self.wReserved = wReserved
        self.wAdditional = wAdditional
        self.wDeadline = wDeadline

        # routing algorithm
        self.routing_offline = routing_offline
        self.routing_online = routing_online

    def particle_swarm_optimization_two_steps(self):
        self.per_hop = False
        tmp_num_particles = self.num_particles
        self.num_particles = 100
        tmp_converged = self.converged
        self.converged = 15
        tmp_link_rates = self.per_link_rates
        self.per_link_rates = None
        gbest, gbest_fitness = self.particle_swarm_optimization()
        self.per_hop = True
        self.per_link_rates = tmp_link_rates
        self.num_particles = tmp_num_particles
        self.converged = tmp_converged
        return self.particle_swarm_optimization(initSol=[gbest, gbest_fitness])

    def particle_swarm_optimization(self, initSol=None, outputfile=None, rep=-1):
        # ----------   Initialize position and velocity of each particle -----------------
        # how many elements are optimized?
        sol_size = self.nrPrios
        if self.per_hop:
            sol_size = self.graph.number_of_edges() * 2
        particles = np.random.randint(self.search_space[0], self.search_space[1],
                                      (self.num_particles, sol_size))
        velocities = np.random.randint(self.vel_space[0], self.vel_space[1], (self.num_particles, sol_size))

        # Initialize global best position and local best position for each particle
        if initSol is None:
            gbest = particles[0].copy()
        else:
            gbest = np.tile(initSol[0].copy(), int(self.graph.number_of_edges() / self.nrPrios) * 2)
        lbest = particles.copy()

        # Initialize the fitness value for each particle
        starttime = time()
        fitness = np.array([float(self.evaluate_fitness(particle, 0)) for particle in particles])
        if initSol is None:
            gbest_fitness = fitness[0]
        else:
            gbest_fitness = initSol[1]
        deltatime_start = time()
        lbest_fitness = fitness.copy()
        deltatime = time() - deltatime_start
        converged = 0
        i = 0
        prev_best = -1
        while converged < self.converged and i < self.num_iterations:
            for j in range(self.num_particles):
                # Update the velocity of each particle
                velocities[j] = self.w * velocities[j] + self.c1 * np.random.rand() * (
                        lbest[j] - particles[j]) + self.c2 * np.random.rand() * (gbest - particles[j])
                particles[j] = particles[j] + velocities[j]

                # Ensure that the solution remains within the search space
                particles[j] = np.clip(particles[j], self.search_space[0], self.search_space[1])

                # Evaluate the fitness of the updated particle
                fitness[j] = self.evaluate_fitness(particles[j], converged)

                # Update the local best position for each particle
                if fitness[j] > lbest_fitness[j]:
                    lbest[j] = particles[j].copy()
                    lbest_fitness[j] = fitness[j]

                # Update the global best position
                if fitness[j] > gbest_fitness:
                    gbest = particles[j].copy()
                    gbest_fitness = fitness[j]
            if gbest_fitness > prev_best:
                # reset converged counter
                converged = 0
                prev_best = gbest_fitness
            else:
                converged += 1
            i += 1
            print("Iteration: ", i)
            print("Converged: ", converged)
            print("Best solution: ", gbest)
            print("Best fitness: ", gbest_fitness)
            if outputfile:
                with open(outputfile, "a+") as file:
                    best_sol_string = ""
                    for element in gbest:
                        best_sol_string += str(element) + " "
                    file.write("pso,"
                               + str(len(self.flow_list)) + ","
                               + str(self.nrPrios) + ","
                               + str(self.per_hop) + ","
                               + str(self.search_space[1]) + ","
                               + str(rep) + ","
                               + str(i) + ","
                               + best_sol_string + ","
                               + str(gbest_fitness) + ","
                               + str(time() - starttime - deltatime) + "\n")

        return gbest, gbest_fitness

    def get_deadline_utility(self, flow_delays):
        """
        Utility Fct. optimises for number of flows when a plateau is introduced between the optimum delay and
        30% below it.

        Parameters
        ----------
        flow_delays

        Returns
        -------

        """
        i = 0
        delay_level = 0
        for delays in flow_delays:
            for delay in delays:
                perfect_delay = self.flow_list[i].deadline  # / Decimal('1.3')
                if delay is None or delay == Decimal('0'):
                    pass
                else:
                    # 3. sum up all delay distances
                    if delay <= perfect_delay * Decimal('0.7'):
                        delay_level += delay / perfect_delay  # e.g. 8ms to 10 ms = 8/10= 0.8
                    else:
                        # at from 70% to 100% of the delay value, the delay is considered as optimal
                        delay_level += 1

            i += 1
        # how close are we to the perfect delays?
        perfect_delay_level = len(self.flow_list)  # e.g. 2 for 2 flows and delay_level could then be 0.8+1 = 1.8
        utility = Decimal(delay_level) / perfect_delay_level
        return utility

    def accumulate(self, l):
        it = itertools.groupby(l, operator.itemgetter(0))
        for key, subiter in it:
            yield key, [item[1] for item in subiter]

    def test_with_solution(self, best_solution, flow_list=None, output=False, routing=None):
        if not routing:
            routing = self.routing_offline
        if not flow_list:
            flowlist = self.flow_list.copy()
        else:
            flowlist = flow_list.copy()

        delays = self.get_delays_per_queue(best_solution)
        if self.per_hop:
            results = self.run_flow_reservation(flow_list=flowlist, graph=self.graph, delay_per_queue=delays,
                                           besteffort_perc=self.besteffort_perc,
                                           routing=routing, max_packetsize=self.max_frame_size)
        else:
            results = self.run_flow_reservation(flow_list=flowlist, graph=self.graph, delay=delays,
                                           besteffort_perc=self.besteffort_perc,
                                           routing=routing, max_packetsize=self.max_frame_size)

        flow_delays = [f.budget_path_delays for f in flowlist]
        nr_reserved = len(results['graph_state']['reserved_flows'])
        flow_paths = [f.paths for f in flowlist]
        graph = results['graph_state']['graph']
        simple_graph = results['graph_state']['simple_graph']
        graph_state = results['graph_state']
        slopes = results['statistics']['used_idleSlopes']
        # whether a queue is used
        used_queues = list(self.accumulate(slopes.keys()))

        if output:
            print("Flows: ", results['graph_state']['reserved_flows'])
            print("Slopes: ", slopes)

        return flow_delays, used_queues, nr_reserved, flow_paths, slopes, graph, simple_graph, graph_state

    def evaluate_fitness(self, solution, converged=-1):
        """
        Each objective sums up to max. 1. Then, we add weights (percentages, so that the total max. reward remains 1).
        """
        # We reserve all flows (without deadline requirement) and then evaluate whether they met their deadline
        flow_delays, used_queues, reserved, paths, slopes, graph, simplegraph, graph_state = self.test_with_solution(solution,
                                                                                                        flow_list=
                                                                                                        self.flow_list)

        # how many offine flows have been reserved?
        fitnessReserved = reserved / len(self.flow_list)

        # how good is their deadline?
        fitnessDeadline = self.get_deadline_utility(flow_delays)

        # check that no high priority queue is unused, otherwise, the solution is not valid
        valid = True
        if self.no_unused_queues:
            valid = self.check_validity(used_queues)

        # if we want to keep space for additional flows
        # for the maximum additional flow reservation, the algorithm can get 1 reward in total
        if not self.per_link_rates:
            fitnessAdditional = 1
        else:
            fitnessAdditional = 0
        if self.per_link_rates and (
                fitnessReserved > 0.99 or converged >= self.addFlowsAfter or converged == -1):
            fitnessAdditional, used_queues = self.reserve_additional(graph_state=graph_state, used_queues=used_queues,
                                                                     solution=solution)

            if not valid:
                # check if the solution would be valid NOW
                valid = self.check_validity(used_queues)

        if not valid:
            return -1
        else:

            fitness = self.wReserved * float(fitnessReserved) + \
                      self.wAdditional * float(fitnessAdditional) + \
                      self.wDeadline * float(fitnessDeadline)

            return fitness

    def check_validity(self, used_queues):
        # we only want solutions where higher queues are actually used
        flat_used_queues = np.unique([item for sublist in used_queues for item in sublist[1]])  # make list flat
        # high prios used?
        high_used = all(
            flat_used_queues[i] == i for i in range(min(self.nrPrios, max(flat_used_queues, default=-1)) + 1))
        if high_used:
            return True
        else:
            return False

    def reserve_additional(self, graph_state, used_queues, solution):
        # Alternative implementation exists as optimization problem, but due to non-linearity suffers
        # from performance. Therefore, this implementation iterative reserves flows until no longer possible.
        # track fitness
        required_flows = 0
        reserved_flows = 0
        new_used_queues = used_queues.copy()
        used_links = [x[0] for x in used_queues]
        new_graph_state = copy.deepcopy(graph_state)
        graph = new_graph_state['graph']
        for port in self.per_link_rates.keys():

            # ----- get all information -----
            # For each port, get all links connected to the same node
            connected_links = set(graph.edges(port[0]))
            # Check if #Links > 1 (if not, it is an ES and we do nothing)
            if not len(connected_links) > 1:
                continue
            # remove own edge
            connected_links.remove(port)

            # 4. Iterate over all profiles
            i = 0
            cycle_links = itertools.cycle(connected_links)
            for profile in profiles:
                (data, cmi, deadline, packetsize) = profile
                # 5. How many flows from this profile are expected?
                profile_rate = data / cmi
                link_rate = graph[port[0]][port[1]][0]['rate_out']
                required_rate = Decimal(self.per_link_rates.get((port[0], port[1]), [0, 0, 0, 0, 0])[i]) * link_rate
                nr_flows = round(required_rate / profile_rate)
                required_flows += nr_flows
                i += 1
                # 6. Add flows successively from the different input links,
                # by defining the path as input link - output link
                sink = port[1]
                successes = 0
                # how often did it fail (if it fails for all links, stop)
                failed = []
                while len(failed) < len(connected_links) and successes < nr_flows:
                    # get next link in a loop
                    current_input = next(cycle_links)
                    while current_input in failed:
                        current_input = next(cycle_links)
                    source = current_input[1]
                    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=15))
                    flow = Flow(flowID=random_string, source=source, sinks=[sink], max_frame_size=packetsize,
                                sending_interval=cmi, data_per_interval=data, deadline=deadline / self.pathLength)
                    result = add_flow(flow=flow, graph_state=new_graph_state, routing=1)
                    if result['success']:
                        successes += 1
                        reserved_flows += 1
                        # ---- update the used queues ----
                        used_queue = flow.paths[0][-1]
                        try:
                            index_queue = used_links.index(port)
                        except ValueError:
                            index_queue = -1
                        if index_queue >= 0:
                            if not used_queue[2] in new_used_queues[index_queue][1]:
                                new_used_queues[index_queue][1].append(used_queue[2])
                        else:
                            new_used_queues.append((port, [used_queue[2]]))
                        # -------------------------------
                    else:
                        failed.append(current_input)
            fitness = reserved_flows / required_flows
            return fitness, new_used_queues

    def run_flow_reservation(self, flow_list, graph, besteffort_perc, routing, max_packetsize, delay_per_queue=None,
                             delay=None):
        if delay:
            graph_state = init_topology(graph=graph, delays=delay,
                                        besteffort_perc=besteffort_perc, max_packetsize=max_packetsize,
                                        output=False, constant_priority=self.constant_priority)
        else:
            graph_state = init_topology(graph=graph, delay_per_queue=delay_per_queue,
                                        besteffort_perc=besteffort_perc, max_packetsize=max_packetsize,
                                        output=False, constant_priority=self.constant_priority)

        results = None
        flowlist = copy.deepcopy(flow_list)
        for flow in flowlist:
            results = add_flow(flow=flow, graph_state=graph_state, routing=routing)
            graph_state = results['graph_state']

        return results

    def get_delays_per_queue(self, delays):
        continous_solution = [(x / Decimal(self.conversion_factor)) for x in delays]
        if self.per_hop:
            edges = self.graph.edges()
            # dictionary with key: edge and value: individual delay
            delay_per_queue = {}
            for edge in edges:
                if edge not in delay_per_queue:
                    per_hop_val = []
                    for sol_length in range(self.nrPrios):
                        per_hop_val.append(continous_solution.pop(0))
                    delay_per_queue[edge] = per_hop_val
            return delay_per_queue
        return continous_solution

