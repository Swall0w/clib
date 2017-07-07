import cv2
import numpy as np
from chainer import serializers, Variable, cuda
import chainer.functions as F
import argparse
from yolo.model.yolov2 import YOLOv2, YOLOv2Predictor
from yolo.common.format_image_size import resize_to_yolo
from yolo.common.utils import nms, Box

class Predictor:
    def __init__(self,weight_file,data_label,\
                gpu,n_classes,\
                n_boxes,detection_thresh,iou_thresh):

        self.weight_file = weight_file
        self.data_label = data_label
        self.gpu = gpu
        self.n_classes = n_classes
        self.n_boxes = n_boxes
        self.detection_thresh = detection_thresh
        self.iou_thresh = iou_thresh

        with open(self.data_label,'r') as f:
            self.labels = [item for item in f.read().split('\n') if item is not '']

        # load model
        print("loading model...")
        yolov2 = YOLOv2(n_classes=self.n_classes, n_boxes=self.n_boxes)
        model = YOLOv2Predictor(yolov2)
        serializers.load_npz(weight_file, model) # load saved model

        if self.gpu >=0:
            cuda.get_device(self.gpu).use()
            model.to_gpu()
            print('gpu')
        model.predictor.train = False
        model.predictor.finetune = False
        self.model = model

    def __call__(self, orig_img):
        orig_input_height, orig_input_width, _ = orig_img.shape
        img = resize_to_yolo(orig_img)
        input_height, input_width, _ = img.shape
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        # forward
        x_data = img[np.newaxis, :, :, :]
        x = Variable(x_data)
        x, y, w, h, conf, prob = self.model.predict(x)

        # parse results
        _, _, _, grid_h, grid_w = x.shape
        x = F.reshape(x, (self.n_boxes, grid_h, grid_w)).data
        y = F.reshape(y, (self.n_boxes, grid_h, grid_w)).data
        w = F.reshape(w, (self.n_boxes, grid_h, grid_w)).data
        h = F.reshape(h, (self.n_boxes, grid_h, grid_w)).data
        conf = F.reshape(conf, (self.n_boxes, grid_h, grid_w)).data
        prob = F.transpose(F.reshape(prob, (self.n_boxes, self.n_classes, grid_h, grid_w)), (1, 0, 2, 3)).data
        detected_indices = (conf * prob).max(axis=0) > self.detection_thresh

        results = []
        for i in range(detected_indices.sum()):
            results.append({
                "label": self.labels[prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax()],
                "probs": prob.transpose(1, 2, 3, 0)[detected_indices][i],
                "conf" : conf[detected_indices][i],
                "objectness": conf[detected_indices][i] * prob.transpose(1, 2, 3, 0)[detected_indices][i].max(),
                "box"  : Box(
                            x[detected_indices][i]*orig_input_width,
                            y[detected_indices][i]*orig_input_height,
                            w[detected_indices][i]*orig_input_width,
                            h[detected_indices][i]*orig_input_height).crop_region(orig_input_height, orig_input_width)
            })

        nms_results = nms(results, self.iou_thresh)
        return nms_results

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i',type=str,default='image.jpg',help='Input image data')
    parser.add_argument('--output','-o',type=str,default='result.jpg',help='Output image data')
    parser.add_argument('--model','-m',type=str,default='result/yolo.weights',help='model data')
    parser.add_argument('--names','-n',type=str,default='data/voc.names',help='label list')
    parser.add_argument('--gpu','-g',type=int,default=-1,help='gpu device')
    parser.add_argument('--cls','-c',type=int,default=20,help='number of class')
    parser.add_argument('--box','-b',type=int,default=5,help='number of class')
    parser.add_argument('--detection','-d',default=0.3,help='detection thresh')
    parser.add_argument('--iou',default=0.3,help='iou thresh')
    return parser.parse_args()

if __name__ == "__main__":
    args = arg()
    image_file = args.input

    # read image
    print("loading image...")
    orig_img = cv2.imread(image_file)

    predictor = Predictor(weight_file=args.model, data_label=args.names, gpu = args.gpu,n_classes = args.cls, n_boxes= args.box, detection_thresh = args.detection, iou_thresh=args.iou)
    import time
    start_time = time.time()
    nms_results = predictor(orig_img)
    duration = time.time() - start_time
    print('duration :',duration)


    # draw result
    for result in nms_results:
        left, top = result["box"].int_left_top()
        cv2.rectangle(
            orig_img,
            result["box"].int_left_top(), result["box"].int_right_bottom(),
            (255, 0, 255),
            3
        )
        text = '%s(%2d%%)' % (result["label"], result["probs"].max()*result["conf"]*100)
        cv2.putText(orig_img, text, (left, top-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        print(text)
    cv2.imwrite(args.output, orig_img)
