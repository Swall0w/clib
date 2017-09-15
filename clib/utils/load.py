def load_class(namesfile):
    with open(namesfile, 'r') as f:
        tags = {int(x): item.strip() for x, item in enumerate(f.readlines())}
    return tags
