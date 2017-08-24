import xml.etree.ElementTree as ET

def xml_parse(f):
    tree = ET.parse(f)
    xmlroot = tree.getroot()
    size = xmlroot.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    ground_truths = []
    for obj in xmlroot.iter('object'):
        xmlbox = obj.find('bndbox')
        ground_truths.append({
            'xmin': int(xmlbox.find('xmin').text),
            'ymin': int(xmlbox.find('ymin').text),
            'xmax': int(xmlbox.find('xmax').text),
            'ymax': int(xmlbox.find('ymax').text),
            'label': obj.find('name').text
        })
    return (width, height), ground_truths
    
def voc_load(file_path):
    with open(file_path) as f:
        ret = xml_parse(f)
    return ret
