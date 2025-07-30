import numpy as np


def angular_distance(vecs_1, vecs_2):

    dot = np.einsum("ij,ij->i", vecs_1, vecs_2) / (
        np.linalg.norm(vecs_1, axis=1) * np.linalg.norm(vecs_2, axis=1)
    )
    dot = np.clip(dot, -1, 1)

    return np.abs(np.rad2deg(np.arccos(dot)))


def gaze_offset_direction(vecs1, vecs2, reference_directions, l_min=0.3):
    """
    Calculate the angular offset direction between two sets of gaze vectors.

    Args:
        vecs1 (np.ndarray): First set of gaze vectors.
        vecs2 (np.ndarray): Second set of gaze vectors. This should come after vecs1
        reference_directions (np.ndarray): Reference directions for normalization.

    Returns:
        np.ndarray: Angular offsets in degrees.
    """
    vec_diff = (vecs2 - vecs1) * l_min
    delta = np.linalg.norm(reference_directions, axis=1) ** 2 + 2 * np.einsum(
        "ij,ij->i", reference_directions, vec_diff
    )

    return delta


def single_angular_distance(vec_1, vec_2):
    dot = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
    dot = np.clip(dot, -1, 1)

    return np.abs(np.rad2deg(np.arccos(dot)))


def cos_distance(vecs_1, vecs_2):
    dot = np.einsum("ij,ij->i", vecs_1, vecs_2) / (
        np.linalg.norm(vecs_1, axis=1) * np.linalg.norm(vecs_2, axis=1)
    )
    dot = np.clip(dot, -1, 1)
    return dot


def project_gaze_to_fix_plane(direction, origin, z_value=-0.2):
    # Compute the scalar multiple of direction needed to reach the plane
    if direction[2] == 0:  # Ray is parallel to the plane
        return None  # No intersection

    # Calculate how far we need to go along the ray to hit the plane
    t = (z_value - origin[2]) / direction[2]

    # If t is negative, the plane is behind our origin
    if t < 0:
        return None

    # Calculate the intersection point
    x = origin[0] + t * direction[0]
    y = origin[1] + t * direction[1]

    return x, y


def project_gazes_to_fix_plane(directions, origins, z_value=-0.2):
    projected_gazes = []
    flag = False
    for direction, origin in zip(directions, origins):
        # print(direction, origin)
        projected_gaze = project_gaze_to_fix_plane(direction, origin, z_value)
        if projected_gaze is None:
            flag = True
            break
        projected_gazes.append(projected_gaze)
    # print("projected_gazes: ", projected_gazes)
    if flag:
        print("False plane encountered")
        # for now, just return a faked projected gaze, all at (100, 100)
        projected_gazes = [[2, 2] for _ in range(len(directions))]
    return projected_gazes
