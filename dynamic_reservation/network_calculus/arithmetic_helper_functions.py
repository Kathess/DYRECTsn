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
from typing import List, Tuple
from dynamic_reservation.environment.flow import Flow


def get_tsn_arrival(C: Decimal, flow: Flow) -> Tuple[List[Tuple[Decimal, Decimal, Decimal]], Decimal]:
    """
    calculates the arrival curve as token bucket given the link rate and the flow with CMI and m
    (relevant at talker)
    :param C: outgoing link rate
    :param flow: flow [source, destination, CMI, m, deadline]
    :return:
    """
    m = flow.data_per_interval
    CMI = flow.sending_interval
    # rate for TSN is: r = m/CMI
    r = m / CMI
    # packetized burst for TSN: b = m - r * (m-l)/C
    b = m - r * (m - flow.max_frame_size) / C
    # previously: b = m * (Decimal('1') - (r / C))
    return [(Decimal('0'), b, r)], r


def createnewpoint(smaller: Tuple[Decimal, Decimal, Decimal], ll: Tuple[Decimal, Decimal, Decimal]) \
        -> Tuple[Decimal, Decimal, Decimal]:
    """
    gets new point and rate for ll (x is the same but what is y and r?)

    :param smaller: arrival point that is smaller than the current point
    :param ll: arrival point that need to be updated
    :return: new values for ll
    """
    ll0 = ll[0]
    s2 = smaller[2]
    newy = s2 * (ll0 - smaller[0]) + smaller[1] + ll[1]  # (float(ll[0]) - smaller[0]) + smaller[1] + float(ll[1])
    newrate = ll[2] + s2
    newtuple = (ll0, newy, newrate)

    return newtuple


def add(oldarrival: List[Tuple[Decimal, Decimal, Decimal]],
        newarrival: List[Tuple[Decimal, Decimal, Decimal]]) -> List[Tuple[Decimal, Decimal, Decimal]]:
    """
    general adding of curves
    :param oldarrival: first arrival curve to sum up
    :param newarrival: second arrival curve to sum up
    :return:
    """
    # add oldarrival to newarrival
    res = []
    i, j = 0, 0
    smaller = (Decimal('0'), Decimal('0'), Decimal('0'))

    if oldarrival is None:
        oldarrival = []
    if newarrival is None:
        newarrival = []

    lenold = len(oldarrival)
    lennew = len(newarrival)

    assert lenold == 0 or oldarrival[0][0] == 0, "first element in arrival curve must always have x=0 (oldarrival)"
    assert lennew == 0 or newarrival[0][0] == 0, "first element in arrival curve must always have x=0 (newarrival)"

    # sort the elements of the arrivals and calculate their new results
    while i < lenold and j < lennew:
        # in case the x values of two elements in both arrivals are the same
        oldi = oldarrival[i]
        newj = newarrival[j]
        if oldi[0] == newj[0]:
            newtuple = (oldi[0], oldi[1] + newj[1], oldi[2] + newj[2])
            res.append(newtuple)
            if not j + 1 < lennew:
                smaller = newj
            elif not i + 1 < lenold:
                smaller = oldi
            elif oldarrival[i + 1][0] <= newarrival[j + 1][0]:
                smaller = newj
            elif oldarrival[i + 1][0] > newarrival[j + 1][0]:
                smaller = oldi
            i += 1
            j += 1
        elif oldi[0] < newj[0]:
            newtuple = createnewpoint(smaller, oldi)
            res.append(newtuple)
            if not i + 1 < lenold:
                smaller = oldi
            elif oldarrival[i + 1][0] <= newj[0]:
                smaller = newj
            elif oldarrival[i + 1][0] > newj[0]:
                smaller = oldi
            i += 1
        else:
            newtuple = createnewpoint(smaller, newj)
            res.append(newtuple)
            if not j + 1 < lennew:
                smaller = newj
            elif oldi[0] <= newarrival[j + 1][0]:
                smaller = newj
            elif oldi[0] > newarrival[j + 1][0]:
                smaller = oldi
            j += 1

    # add last items
    if oldarrival[i:]:
        for restelements in oldarrival[i:]:
            newtuple = createnewpoint(smaller, restelements)
            res.append(newtuple)
    elif newarrival[j:]:
        for restelements in newarrival[j:]:
            newtuple = createnewpoint(smaller, restelements)
            res.append(newtuple)

    return res


def nextIntersection(pointbefore: Tuple[Decimal, Decimal, Decimal], shaperrate: Decimal, shaperburst: Decimal) \
        -> Tuple[Decimal, Decimal]:
    """
    we know that the intersection is after pointbefore (works for pointbefore is bigger and smaller than shaper)
    :param pointbefore: point before the intersection
    :param shaperrate: for the shaper that we want to intersect
    :param shaperburst: for the shaper that we want to intersect
    :return: intersection of shaper and curve
    """
    # y + (t-x) * r = sb + sr * t
    # thus
    # t = ( y - xr - sb ) / (sr - r)
    intersection_x = (pointbefore[1] - pointbefore[0] * pointbefore[2] - shaperburst) / (
            shaperrate - pointbefore[2])
    intersection_y = shaperrate * intersection_x + shaperburst
    return intersection_x, intersection_y


def linkrateIntersection(firstcurve: List[Tuple[Decimal, Decimal, Decimal]], rate: Decimal, l_max: Decimal) \
        -> Tuple[Decimal, Decimal, Decimal]:
    """
    between (link)rate + l_max and a curve
    :param firstcurve: curve to be shaped
    :param rate: link rate
    :param l_max: max packet size of the link shaper curve
    :return: intersection of link and curve
    """

    # find the last point in arrival curve that is bigger than curve derived from rate + l
    lastupperpoint = -1
    for i, arrival_tuple in enumerate(firstcurve):
        if arrival_tuple[1] >= rate * arrival_tuple[0] + l_max:
            # i is too big, so we need the intersection of the shaper and arrival curve between i and i+1
            lastupperpoint = i

    # if there is a point bigger than the link rate, we need the intersection, otherwise, we do not need
    # to change anything in the curve
    if lastupperpoint > -1:
        # int_x = (y-rx-l_max) / (C-r)
        int_sec_point = firstcurve[lastupperpoint]
        intersectionx = (int_sec_point[1] - int_sec_point[0] * int_sec_point[2] - l_max) / (rate - int_sec_point[2])
        # int_y = int_x * C
        intersectiony = intersectionx * rate + l_max
        intersectionrate = int_sec_point[2]
        return intersectionx, intersectiony, intersectionrate
    else:
        return Decimal('0'), Decimal('0'), Decimal('0')


def shift(arrivalcurve, delay):
    # more than one point at zero?
    lastpoint = arrivalcurve[0]
    belowzero = 0
    for i, arrival_tuple in enumerate(arrivalcurve):
        if arrival_tuple[0] - delay < Decimal('0'):
            arrivalcurve[i] = (arrival_tuple[0] - delay, arrival_tuple[1], arrival_tuple[2])
            belowzero += 1
            lastpoint = (arrival_tuple[0] - delay, arrival_tuple[1], arrival_tuple[2])
        else:
            arrivalcurve[i] = (arrival_tuple[0] - delay, arrival_tuple[1], arrival_tuple[2])
    # remove belowzero elements from the list, so we have only the highest point at zero
    arrivalcurve = arrivalcurve[belowzero:]

    # add beginning of arrival curve
    arrivalcurve.insert(0, (Decimal('0'), -1 * lastpoint[0] * lastpoint[2] + lastpoint[1], lastpoint[2]))

    return arrivalcurve


def find_first_matching_index(array1, array2):
    for i, (item1, _, _) in enumerate(array1):
        for item2, _, _ in array2:
            if item1 == item2:
                return i
    return -1  # Return -1 if no match is found


def modify_arrays_and_get_indexes(array1, array2):
    updated_array1 = []
    tmp_x = []
    indexes = []

    for i, item1 in enumerate(array1):
        match_found = False
        for item2 in array2:
            if item1[0] == item2[0] and item1[1] == item2[1]:
                updated_array1.append(item1 + (item2[2],))
                indexes.append(i)
                match_found = True
                break
        if not match_found:
            tmp_x.append(item1)

    return updated_array1, tmp_x, indexes


def merge_arrays_with_indexes(updated_array1, tmp_x, indexes):
    # Merge the updated_array1 and tmp_x
    merged_array = updated_array1.copy()
    for index, item in zip(indexes, tmp_x):
        # Insert each item from updated_array1 into the merged_array at the corresponding index
        if index < len(merged_array):
            merged_array.insert(index, item)
        else:
            # Append the item to the end if the index is beyond the current length of merged_array
            merged_array.append(item)
    return merged_array


def find_identical_elements(array1, array2):
    identical_elements = []
    for i, item1 in enumerate(array1):
        for j, item2 in enumerate(array2):
            if item1 == item2:
                identical_elements.append(i)
    return identical_elements


def shape_to_8021cb(sum_arrival, arrival_before_replication, path_delay_delta):
    shaping_curve = shift(arrival_before_replication, path_delay_delta)
    shaped_arrival = shape_to_link(sum_arrival, shaping_curve[0][2], shaping_curve[0][1])
    return shaped_arrival


def shape_to_link(portcurve: List[Tuple[Decimal, Decimal, Decimal]], linkrate: Decimal,
                  max_link_packet: Decimal) -> List[Tuple[Decimal, Decimal, Decimal]]:
    """
    shape this sum according to link rate

    2. find intersection of portcurve and linkrate_graph, add this point and remove points with lower x value
    :param max_link_packet: link packet size
    :param portcurve: curve to shape
    :param linkrate: rate of the link (care: we packetize here automatically in the linkrateIntersection)
    :return:
    """

    # 2. shape to linkrate_graph + l_max
    x_intersect, y_intersect, r_intersect = linkrateIntersection(portcurve, linkrate, max_link_packet)

    # if no intersection was found, leave the curve as it is:
    if x_intersect == 0:  # das war davor, ist aber vermutlich falsch: and y_intersect == 0:
        return portcurve
    else:
        # remove lower x points:
        new_portcurve = portcurve.copy()
        for element in portcurve:
            if element[0] < x_intersect:
                new_portcurve.remove(element)
            else:
                break

        # add point first linkrate_graph point and add intersection point
        new_portcurve.insert(0, (x_intersect, y_intersect, r_intersect))
        new_portcurve.insert(0, (Decimal('0'), Decimal(str(max_link_packet)), linkrate))

    return new_portcurve


def shape_to_shaper(portcurve: List[Tuple[Decimal, Decimal, Decimal]], shaperrate: Decimal,
                    shaperburst: Decimal) -> List[Tuple[Decimal, Decimal, Decimal]]:
    """

    1. make sure that arrival curve is nowhere higher than the shaper curve

    :param portcurve: curve to be shaped
    :param shaperrate: max. rate
    :param shaperburst: max. burst, no packetizing needed for this
    :return:
    """

    # 1. shape to shaper (we can only do this, if there has been a CBS scheduling before)
    if shaperrate > 0:
        first_bigger = -1
        last_bigger = -1
        # queue_id's of first_bigger and last_bigger:
        first = 0
        last = 0
        i = 0
        for element in portcurve:
            if element[1] > shaperburst + shaperrate * element[0]:
                if first_bigger == -1:
                    first_bigger = element
                    first = i
                last_bigger = element
                last = i
            i += 1

        # if it is bigger somewhere, do the shaping
        # as we now that the arrival curve is bigger than the shaper curve between f and l,
        # we will delete these points
        if not first_bigger == -1:
            # remove all elements between first and last bigger point (including these)
            # and calculate new intersection points

            arrival_bigger = False
            # will curve be smaller than shaper at any point again?
            if last == len(portcurve) - 1:
                # last_bigger is still above shaper:
                # then we need to check whether the long-term arrival rate is smaller than the shaperrate
                if last_bigger[2] < shaperrate:
                    # a) get intersection when arrival becomes smaller again (t):
                    x, y = nextIntersection(last_bigger, shaperrate, shaperburst)
                    portcurve.insert(last + 1, (x, y, last_bigger[2]))
                    del portcurve[first:(last + 1)]
                elif last_bigger[2] == shaperrate:
                    del portcurve[first:]
                else:
                    # print("WARNING: We would have a buffer overflow since shaper is smaller than arrival."
                    #      "Shaper: {}, portcurve:{}.".format((shaperrate, shaperburst), portcurve))
                    arrival_bigger = True
                    del portcurve[first:]
            else:
                # the arrival get smaller than the shapercurve between two of its points
                # a) get intersection when arrival becomes smaller again (t):
                x, y = nextIntersection(last_bigger, shaperrate, shaperburst)
                portcurve.insert(last + 1, (x, y, last_bigger[2]))
                del portcurve[first:(last + 1)]

            if arrival_bigger:
                rate = last_bigger[2]
            else:
                rate = shaperrate
            if first == 0:
                # if the arrival curve is directly bigger than the shaper, we just add the shapercurve points at
                # the beginning
                portcurve.insert(0, (Decimal('0'), shaperburst, rate))
            else:
                # else we caluclate the intersection with the point before f
                element_before = portcurve[first - 1]
                x, y = nextIntersection(element_before, shaperrate, shaperburst)
                portcurve.insert(first, (x, y, rate))

    return portcurve
