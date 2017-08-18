from PIL import Image, ImageDraw


def viz_bbox(image, outputs):
    for output in outputs:
        draw = ImageDraw.Draw(image)
        draw.rectangle(((output['left'], output['top']),
                       (output['right'], output['bottom'])),
                       outline='red')
        text = '{0}({1:.1f}%)'.format(
            output['class'], output['prob']*100)
        draw.text((output['left'], output['top'] - 8),
                  text, fill='black')
    return image
