def flatten(l):
    return [item for sublist in l for item in sublist]

def for_each(f, l):
    for x in l:
        f(x)