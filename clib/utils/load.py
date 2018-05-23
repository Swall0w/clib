import codecs

def load_class(namesfile):
    with codecs.open(namesfile, 'r', encoding='utf-8') as f:
        tags = {int(x): item.strip() for x, item in enumerate(f.readlines())}
    return tags


def get_index_from_label(tags, value):
    return int(list(tags.keys())[list(tags.values()).index(value)])
