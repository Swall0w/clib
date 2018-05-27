from clib.links.model.recognition import alexnet
from clib.links.model.recognition import densenet
from clib.links.model.recognition import squeezenet
from clib.links.model.recognition import res

from clib.links.model.recognition.alexnet import Alex
from clib.links.model.recognition.densenet import (DenseBlock,
                                                   Transition,
                                                   DenseNet
                                                   )
from clib.links.model.recognition.squeezenet import Fire, SqueezeNet
from clib.links.model.recognition.res import (BottleNeckA, BottleNeckB,
                                              ResNet50, ResNet101, ResNet152)
from clib.links.model.recognition.vggnet import VGGNet
