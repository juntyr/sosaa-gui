import itertools


def generate_time_level_windows():
    # -0.5h, -1.5h, -3h, -6h, -12h, -24h, -48h
    # 0, -2, -5, -11, -23, -47, -95
    time_windows = [
        (0, 0),
        (-2, -1),
        (-5, -3),
        (-11, -6),
        (-23, -12),
        (-47, -24),
        (-95, -48),
    ]

    # +1l, +2l, +4l, +8l, +16l, +32l, +64
    top_windows = [(1, 1), (1, 2), (1, 4), (2, 8), (2, 16), (3, 32), (3, 64)]
    mid_windows = [(0, 0), (0, 0), (0, 0), (-1, 1), (-1, 1), (-2, 2), (-2, 2)]
    bot_windows = [
        (-1, -1),
        (-2, -1),
        (-4, -1),
        (-8, -2),
        (-16, -2),
        (-32, -3),
        (-64, -3),
    ]

    return list(
        itertools.chain(
            zip(time_windows, top_windows),
            zip(time_windows, mid_windows),
            zip(time_windows, bot_windows),
        )
    )


def generate_windowed_feature_names(columns):
    time_windows = ["-0.5h", "-1.5h", "-3h", "-6h", "-12h", "-24h", "-48h"]

    top_windows = ["+1l", "+2l", "+4l", "+8l", "+16l", "+32l", "+64l"]
    mid_windows = ["+0l", "+0l", "+0l", "±1l", "±1l", "±2l", "±2l"]
    bot_windows = ["-1l", "-2l", "-4l", "-8l", "-16l", "-32l", "-64l"]

    names = []

    for t, l in itertools.chain(
        zip(time_windows, top_windows),
        zip(time_windows, mid_windows),
        zip(time_windows, bot_windows),
    ):
        for c in columns:
            names.append(f"{c}{t}{l}")

    return names


def time_level_window_mean(input, t_range, l_range, progress=None):
    import numpy as np

    output = np.zeros(shape=input.shape)

    for t in range(input.shape[0]):
        mint = min(max(0, t + t_range[0]), input.shape[0])
        maxt = max(0, min(t + 1 + t_range[1], input.shape[0]))

        if mint == maxt:
            continue

        for lev in range(input.shape[1]):
            minl = min(max(0, lev + l_range[0]), input.shape[1])
            maxl = max(0, min(lev + 1 + l_range[1], input.shape[1]))

            if minl == maxl:
                continue

            output[t, lev, :] = np.mean(
                input[mint:maxt, minl:maxl, :], axis=(0, 1)
            )

    if progress is not None:
        progress.update_minor()

    return output
