from chainer.links import VGG16Layers
from PIL import Image


def main():
    model = VGG16Layers()
    img = Image.open('0.jpg')
    feature = model.extract([img], layers=['fc7'])['fc7']
    print(feature)
    print(len(feature[0]))


if __name__ == '__main__':
    main()
