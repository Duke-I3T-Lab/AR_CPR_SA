def merge_periods(all_extracted_periods, interval_steps_tolerance):
    i = 0
    while i < len(all_extracted_periods) - 1:
        if (
            all_extracted_periods[i + 1][0] - all_extracted_periods[i][1]
            < interval_steps_tolerance
        ):
            # replace the i th and i+1 th element with a merged element
            all_extracted_periods[i] = (
                all_extracted_periods[i][0],
                all_extracted_periods[i + 1][1],
            )
            all_extracted_periods.pop(i + 1)
        else:
            i += 1
    return all_extracted_periods


def discard_short_periods(all_extracted_periods, minimum_fixation_steps):
    return [
        period
        for period in all_extracted_periods
        if period[1] - period[0] >= minimum_fixation_steps
    ]


def project_gaze_to_fix_plane(direction, origin, z_value=0.15):
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


def project_gazes_to_fix_plane(directions, origins, z_value=0.15):
    projected_gazes = []
    for direction, origin in zip(directions, origins):
        projected_gaze = project_gaze_to_fix_plane(direction, origin, z_value)
        projected_gazes.append(projected_gaze)
    return projected_gazes
