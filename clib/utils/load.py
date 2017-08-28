def load_class(namesfile):
    with open(namesfile, 'r') as f:
        tags = {item.strip(): int(x) for x, item in enumerate(f.readlines())}
    return tags
