import numpy as n


def get_rotation_angles(number):
    '''
    Returns a list of theta, phi values, approximately evenly
    distributed on the sphere.

    Uses the generalised spiral points algorithm explained in
    E B Saff and
    A B J Kuijlaars, Distributing many points on a sphere, The
    Mathematical Intelligencer 19(1) 1997.
    '''

    angles = n.zeros((number, 2))
    angles[0] = n.array([n.arccos(-1), 0])

    for k in range(2, number+1):
        h_k = -1. + (2. * (k - 1)) / (number - 1)
        theta = n.arccos(h_k)
        phi = (angles[k-2, 1] + 3.6/n.sqrt(number) *
               1. / n.sqrt(1 - h_k**2)) % (2*n.pi)
        angles[k-1, 0] = theta
        angles[k-1, 1] = phi
    angles[-1, 1] = 0.  # Last phi will be inf otherwise

    return angles


def rotate_to_top(theta, phi):
    '''
    Returns a rotation matrix that will rotate a sphere such that
    the given positions are at the top.

    Parameters
    ----------
    theta : float
        The latitudinal variable.
    phi : float
        The longitudinal variable.
    '''

    chi = -1 * phi
    alpha = theta

    cc = n.cos(chi)
    sc = n.sin(chi)
    ca = n.cos(alpha)
    sa = n.sin(alpha)
    first_rotation = n.array([[cc, -sc, 0],
                              [sc, cc, 0],
                              [0, 0, 1]])
    second_rotation = n.array([[ca, 0, -sa],
                               [0, 1, 0],
                               [sa, 0, ca]])

    return second_rotation.dot(first_rotation)
