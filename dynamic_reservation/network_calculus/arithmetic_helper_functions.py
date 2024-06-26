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
from bisect import bisect_left
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
    max_frame = flow.max_frame_size
    # to remove:
    if m < 0:
        C *= -1
    # packetized burst for TSN: b = m - r * (m-l)/C
    b = m - r * (m - max_frame) / C
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


def linkrateIntersection(portcurve: List[Tuple[Decimal, Decimal, Decimal]], linkrate: Decimal, max_link_packet: Decimal) \
        -> Tuple[Decimal, Decimal, Decimal]:
    """
    between (link)rate + l_max and a curve
    :param firstcurve: curve to be shaped
    :param rate: link rate
    :param l_max: max packet size of the link shaper curve
    :return: intersection of link and curve
    """
    intersections = []
    for x, y, r in portcurve:
        # The equation of the curve segment is y = r*(x - x0) + y0 where (x0, y0) is the starting point
        # We need to solve r*(x - x0) + y0 = linkrate * x + max_link_packet
        a = r - linkrate
        b = y - r * x - max_link_packet

        if a == 0:
            if b == 0:
                # They are the same line, add the entire line or segment
                intersections.append((x, y, r))
            # No intersection as the lines are parallel
        else:
            x_intersect = -b / a
            y_intersect = linkrate * x_intersect + max_link_packet

            # Check if the intersection is within the valid segment of the portcurve
            if r != 0:
                t = (x_intersect - x) / r
                if int(t) >= 0:
                    intersections.append((x_intersect, y_intersect, min(r, linkrate)))

    if not intersections:
        return Decimal('0'), Decimal('0'), Decimal('0')
    else:
        # Currently only returns the intersection with the smallest x value
        intersections.sort()
        return intersections[0]


def shift(arrivalcurve, delay):
    shifted_curve = []
    interpolated = False

    for x, y, r in arrivalcurve:
        new_x = x - delay
        new_y = y

        # If the new x is less than 0 and we haven't interpolated yet
        if new_x < 0 and not interpolated:
            # Find the point where x = 0
            interpolated_y = y + r * (-new_x)
            shifted_curve.append((0, interpolated_y, r))
            interpolated = True
        elif new_x >= 0:
            shifted_curve.append((new_x, new_y, r))

    return shifted_curve

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

    new_curve = []
    # if no intersection was found, leave the curve as it is:
    if x_intersect <= 0:  # das war davor, ist aber vermutlich falsch: and y_intersect == 0:
        return portcurve

    for x,y,r in portcurve:
        # Calculate the y value of the linkrate line at x
        y_linkrate = linkrate * x + max_link_packet
        if y_linkrate < y:
            new_curve.append((x,y_linkrate,linkrate))
        else:
            new_curve.append((x,y,r))

    def insert_in_order(tuples_list, new_tuple):
        # Extract the x values from the tuples for binary search
        x_values = [t[0] for t in tuples_list]
        # Find the insertion point
        insert_index = bisect_left(x_values, new_tuple[0])
        # Insert the new tuple at the correct index
        tuples_list.insert(insert_index, new_tuple)
        return tuples_list

    # add intersection point
    new_curve = insert_in_order(new_curve, (x_intersect, y_intersect, r_intersect))
    tmp_rate = -1
    for element in new_curve:
        if tmp_rate == element[0]:
            new_curve.remove(element)
        tmp_rate = element[2]

    return new_curve



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
