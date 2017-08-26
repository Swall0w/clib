from clib.datasets import voc_xmlload
from clib.datasets import load_image

from clib.datasets.voc_xmlload import voc_load
from clib.datasets.voc_xmlload import xml_parse
from clib.datasets.load_image import (crop_image_random_transform,
                                      uniform, trans_crop, to_rgb,
                                      convert_2_array, gamma_table)
