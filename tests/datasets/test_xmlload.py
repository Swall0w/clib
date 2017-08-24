import unittest
from clib.datasets import xml_parse
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import io


def _gen_xml(folder, name, shape, bbox_list):
    x_annotation = ET.Element('annotation')
    x_folder = ET.SubElement(x_annotation, 'folder')
    x_folder.text = folder
    x_name = ET.SubElement(x_annotation, 'filename')
    x_name = name
    x_size = ET.SubElement(x_annotation, 'size')
    x_width = ET.SubElement(x_size, 'width')
    x_width.text = str(shape[0])
    x_height = ET.SubElement(x_size, 'height')
    x_height.text = str(shape[1])
    x_depth = ET.SubElement(x_size, 'depth')
    x_depth.text = str(shape[2])

    for bbox in bbox_list:
        obj = ET.SubElement(x_annotation, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = str(bbox[4])
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(bbox[0]))
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(bbox[1]))
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(bbox[2]))
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(bbox[3]))
    
    string = ET.tostring(x_annotation, 'utf-8')
    pretty_string = minidom.parseString(string).toprettyxml(indent='    ')
    return io.StringIO(pretty_string)


class VOCLOAD(unittest.TestCase):
    def test_voc_load(self):
        self.assertEqual(xml_parse(_gen_xml(folder='001',
                                           name='2007_21.jpg',
                                           shape=(281, 500, 3),
                                           bbox_list=[(104, 78, 375, 183, 'aeroplane')]
        )),((281, 500), [{'xmin': 104, 'ymin': 78, 'xmax': 375, 'ymax': 183, 'label': 'aeroplane'}]))

if __name__ == '__main__':
    unittest.main()
