import hashlib


# https://stackoverflow.com/a/67809235
def df_to_numpy(df):
    try:
        shape = [len(level) for level in df.index.levels]
    except AttributeError:
        shape = [len(df.index)]
    ncol = df.shape[-1]
    if ncol > 1:
        shape.append(ncol)
    return df.to_numpy().reshape(shape)


def hash_for_dt(dt):
    if not (isinstance(dt, tuple) or isinstance(dt, list)):
        dt = [dt]

    dt_str = ".".join(dtt.strftime("%d.%m.%Y-%H:00%z") for dtt in dt)

    h = hashlib.shake_256()
    h.update(dt_str.encode("ascii"))

    return h
