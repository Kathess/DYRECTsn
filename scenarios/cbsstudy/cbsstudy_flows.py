# Copyright (C) 2024 Timo Salomon for HAW Hamburg, Germany
# timo.salomon@haw-hamburg.de
from dynamic_reservation.environment.flow import Flow
from math import floor


def getFlows(inputLinks, stages, totalTalkerBandwidth, talkerInterval, ctInterval, talkerDeadline, withPrioCt):
    ctFlows = []
    talkerFlows = []
    IFG = 96
    singleTalkerBandwidth = totalTalkerBandwidth / (inputLinks * 1.0)
    talkerData = floor((singleTalkerBandwidth / 8.0) * talkerInterval) * 8
    ctData = floor(((totalTalkerBandwidth - singleTalkerBandwidth) / 8.0) * talkerInterval) * 8

    for i in range(0, inputLinks):
        talkerFlows.append(
            Flow(
                flowID="t" + str(i),
                source="talker_" + str(i),
                sinks=["listener"],
                data_per_interval=talkerData,
                sending_interval=talkerInterval,
                deadline=talkerDeadline,
                max_frame_size=talkerData,
                priority=0,
            )
        )
        if withPrioCt:
            for j in range(0, stages):
                node = "node_input_" + str(i) + "_stage_" + str(j)
                if j == stages - 1:
                    if i == inputLinks - 1:
                        destination = "node_input_" + str(0) + "_stage_" + str(0)
                    else:
                        destination = "node_input_" + str(i + 1) + "_stage_" + str(0)
                else:
                    destination = "node_input_" + str(i) + "_stage_" + str(j + 1)
                ctFlows.append(
                    Flow(
                        flowID="ct_" + str(i) + "_" + str(j),
                        source=node,
                        sinks=[destination],
                        data_per_interval=ctData,
                        sending_interval=ctInterval,
                        deadline=10.0,
                        max_frame_size=ctData,
                        priority=0,
                    )
                )

    return talkerFlows, ctFlows
