'''
Python version of cython functions for space curve analysis.
'''

import numpy as n
from numba import njit, jit

csqrt = n.sqrt
floor = n.floor


@njit
def cross_product(px, py, qx, qy):
    '''Simple cython cross product for 2D vectors.'''
    return px * qy - py * qx

@njit
def mag_difference(a, b):
    '''The magnitude of the vector joining a and b'''
    return csqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

@njit
def sign(a):
    return (1. if a > 0. else (-1. if a < 0. else 0.))
                                      
@njit
def do_vectors_intersect(px, py, dpx, dpy,
                                qx, qy, dqx, dqy):
    """Takes four vectors p, dp and q, dq, then tests whether they cross in
    the dp/dq region. Returns this boolean, and the (fractional) point where
    the crossing actually occurs.
    """
    if abs(cross_product(dpx, dpy, dqx, dqy)) < 0.00001:
        return (0, 0., 0.)

    t = cross_product(qx - px, qy - py, dqx, dqy) / cross_product(dpx, dpy,
                                                                  dqx, dqy)
    if t < 1.0 and t > 0.0:
        u = cross_product(qx - px, qy - py, dpx, dpy) / cross_product(dpx, dpy,
                                                                      dqx, dqy)
        if u < 1.0 and u > 0.0:
            return (1, t, u)
    return (0, -1., -1.)
        

@jit('(float64[:,:])(float64[:], float64[:], float64[:,:], float64[:], int64, int64, int64, int64)',
     nopython=True)
def find_crossings(v, dv,
                   points,
                   segment_lengths,
                   current_index,
                   comparison_index,
                   max_segment_length,
                   jump_mode
                   ):
    '''
    Searches for crossings between the given vector and any other
    vector in the
    list of points, returning all of them as a list.
                     
    Parameters
    ----------
    v0 : ndarray
        The current point, a 1D vector.
    dv : ndarray
        The vector connecting the current point to the next one
    points : ndarray
        The array or (x, y) values of all the other points
    segment_lengths : ndarray
        The length of each segment joining a point to the
        next one.
    current_index : long
        The index of the point currently being tested.
    comparison_index : long
        The index of the first comparison point
    jump_mode : int
        1 to check every jump distance, 2 to jump based on
        the maximum one, 3 to never jump and check the length
        of every step.
    '''

    crossings = n.zeros((len(points), 4))
    vx = v[0]
    vy = v[1]
    vz = v[2]
    dvx = dv[0]
    dvy = dv[1]
    dvz = dv[2]

    twice_max_segment_length = 2*max_segment_length

    i = 0
    already_jumped = 0

    while i < len(points) - 1:
        point = points[i]
        distance = csqrt(pow(vx - point[0], 2) + pow(vy - point[1], 2))
        if distance < twice_max_segment_length or already_jumped:
            already_jumped = 0
            next_point = points[i+1]
            jump_x = next_point[0] - point[0]
            jump_y = next_point[1] - point[1]
            jump_z = next_point[2] - point[2]


            intersect, intersect_i, intersect_j = do_vectors_intersect(
                vx, vy, dvx, dvy, point[0], point[1],
                jump_x, jump_y)

            if intersect:
                pz = point[2]
                dpz = jump_z

                crossing_sign = sign((vz + intersect_i * dvz) -
                                     (pz + intersect_j * dpz))

                crossing_direction = sign(cross_product(
                    dvx, dvy, jump_x, jump_y))

                crossings[i] += n.array(
                    [current_index + intersect_i,
                     (comparison_index + intersect_j + i),
                     crossing_sign,
                     crossing_sign * crossing_direction])
                crossings[i] += n.array(
                    [(comparison_index + intersect_j + i),
                     current_index + intersect_i,
                     -1. * crossing_sign,
                     crossing_sign * crossing_direction])
            i += 1

        elif jump_mode == 3:
            i += 1  # naive mode - check everything
            already_jumped = 1
        elif jump_mode == 2:
            num_jumps = int((floor(distance / max_segment_length)) - 1)
            if num_jumps < 1:
                num_jumps = 1
            i += num_jumps
            already_jumped = 1
        else:  # Catch all other jump modes
            distance_travelled = 0.
            jumps = 0
            while (distance_travelled < (distance - max_segment_length) and
                   i < len(points)):
                jumps += 1
                distance_travelled += segment_lengths[i]
                i += 1
            if jumps > 1:
                i -= 2
            already_jumped = 1
            # This keeps jumping until we might be close enough to intersect,
            # without doing vector arithmetic at every step
                                  
    return crossings


@jit('Tuple((float64[:,:], float64[:,:]))(float64[:,:], float64[:], int64, int64, float64[:], int64)',
     nopython=True)
def raw_cross_loop(lines, segment_lengths, max_segment_length, jump_mode, cumulative_lengths, include_closures):
    for line_index, line in enumerate(lines):
        for other_index, other_line in enumerate(lines[line_index+1:]):
            other_index += line_index + 1
            #by this point these are just two array-likes
            points = line
            comparison_points = other_line
            if include_closures:
                comparison_points = n.vstack((comparison_points,
                                              comparison_points[:1]))
            other_seg_lengths = segment_lengths[other_index]
            #other_seg_lengths is already corrected to include
            #closures if necessary
            first_line_range = range(len(points))
            if not include_closures:
                first_line_range = first_line_range[:-1]
            for i in first_line_range:
                v0 = points[i]
                dv = points[(i + 1) % len(points)] - v0
                
                vnum = i
                compnum = 0  # start at beginning of other line
                
                new_crossings = find_crossings(
                    v0, dv, comparison_points, other_seg_lengths,
                    vnum, compnum,
                    max_segment_length,
                    jump_mode)

                if not len(new_crossings):
                        continue
                first_crossings = n.array(new_crossings[::2])
                first_crossings[:, 0] += cumulative_lengths[line_index]
                first_crossings[:, 1] += cumulative_lengths[other_index]
                sec_crossings = n.array(new_crossings[1::2])
                sec_crossings[:, 0] += cumulative_lengths[other_index]
                sec_crossings[:, 1] += cumulative_lengths[line_index]

    return first_crossings, sec_crossings
