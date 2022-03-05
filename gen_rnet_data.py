import numpy as np
import cv2
import os
import numpy.random as npr
import sys
sys.path.append("../")
from tool import IoU,IoU_boxes
import _pickle as pickle
from mtcnn.mtcnn import MTCNN


image_size=24
detector = MTCNN()
anno_file = r"D:\graduatestudy\DIRECTION\MTCNN\keras-mtcnn-master\keras-mtcnn-master\data\wider_face_train.txt"
im_dir = r"D:\graduatestudy\DIRECTION\MTCNN\dataset\WIDER_train\WIDER_train\images"
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print ("{:d} pics in total".format(num))
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # part
idx = 0 # image
negative = []
positive = []
part=[]


neg_cls_list = []
part_roi_list = []

for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    bbox = list(map(float, annotation[1:]) )
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
    height, width, channel = img.shape
    idx += 1
    if idx % 1000 == 0:
        print (idx, "images done")

    result = detector.detect_faces(img)
    boxes_pnet = result[0]
    n_num=0
    for box_pnet in boxes_pnet:
        nx1 = int(box_pnet[0])
        ny1 = int(box_pnet[1])
        nx2 = int(box_pnet[2])
        ny2 = int(box_pnet[3])
        if nx1<0 or ny1<0 or nx2>width or ny2>height:
            continue
        iou,idx_bb=IoU_boxes(box_pnet,boxes)
        if iou>=0.4:
            box = boxes[idx_bb]
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue
            size = nx2 - nx1
            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            cropped_im = img[ny1: ny2, nx1: nx2, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_AREA)
            # save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
            # cv2.imwrite(save_file, resized_im)
            resized_im = (resized_im - 127.5) * 0.0078125
            roi = [offset_x1, offset_y1, offset_x2, offset_y2]
            if iou >= 0.65:
                label = 1
                positive.append([resized_im, label, roi])
                p_idx += 1
            else:
                label = -1
                part_roi_list.append([resized_im, label, roi])
                d_idx += 1
        elif iou < 0.3:
            if n_num>10:
                continue
            cropped_im = img[ny1: ny2, nx1: nx2, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_AREA)
            resized_im = (resized_im - 127.5) * 0.0078125
            label = 0
            neg_cls_list.append([resized_im, label])
            n_idx += 1
            n_num+=1

if len(part_roi_list)>p_idx:
    part_keep = npr.choice(len(part_roi_list), size=p_idx * 1, replace=False)
    for i in part_keep:
        part.append(part_roi_list[i])
else:
    part.extend(part_roi_list)
if len(neg_cls_list)>=p_idx*3:
    neg_keep = npr.choice(len(neg_cls_list), size=p_idx*3, replace=False)
    for i in neg_keep:
        negative.append(neg_cls_list[i])
else:
    negative.extend(neg_cls_list)


fid = open("imdb/{0:d}/positive_v1.imdb".format(image_size),'wb')
pickle.dump(positive, fid)
fid.close()

fid = open("imdb/{0:d}/part_v1.imdb".format(image_size),'wb')
pickle.dump(part, fid)
fid.close()

fid = open("imdb/{0:d}/negative_v1.imdb".format(image_size),'wb')
pickle.dump(negative, fid)
fid.close()
