import numpy as np
import cv2
import os
import numpy.random as npr
import sys
sys.path.append("../")
from tool import IoU
import _pickle as pickle


image_size=12
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
box_idx = 0
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
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    idx += 1
    if idx % 100 == 0:
        print (idx, "images done")

    height, width, channel = img.shape


    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        if max(w, h) < 20 or x1 < 0 or y1 < 0:
            continue

        # generate positive examples and part faces
        p_idx1 = 0
        d_idx1 = 0
        for i in range(6):
            #{
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(int(-w * 0.2), int(w * 0.2))
            delta_y = npr.randint(int(-h * 0.2), int (h * 0.2))

            nx1 = max( int(x1 + w / 2 + delta_x - size / 2 ), 0)
            ny1 = max( int(y1 + h / 2 + delta_y - size / 2), 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            #}

            crop_box = np.array([nx1, ny1, nx2, ny2])

##############
            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                cropped_im = img[ny1 : ny2, nx1 : nx2, :]
                resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_AREA)
                #save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                #cv2.imwrite(save_file, resized_im)
                resized_im = (resized_im - 127.5)*0.0078125
                label = 1
                roi = [offset_x1, offset_y1, offset_x2, offset_y2]
                positive.append([resized_im, label, roi])
                p_idx += 1
                p_idx1 += 1
            elif IoU(crop_box, box_) >= 0.4:
                cropped_im = img[ny1 : ny2, nx1 : nx2, :]
                resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_AREA)
                #save_file = os.path.join(part_save_dir, "%s.jpg"%d_idx)
                #cv2.imwrite(save_file, resized_im)
                resized_im = (resized_im - 127.5)*0.0078125
                label = -1
                roi = [offset_x1, offset_y1, offset_x2, offset_y2]
                part_roi_list.append([resized_im, label, roi])
                d_idx += 1
                d_idx1 += 1
        if p_idx1 == 0 and d_idx1 != 0:
            for i in range(20):
                size = npr.randint(int(min(w, h) * 0.9), np.ceil(1.1 * max(w, h)))

                # delta here is the offset of box center
                delta_x = npr.randint(int(-w * 0.1), int(w * 0.1))
                delta_y = npr.randint(int(-h * 0.1), int(h * 0.1))

                nx1 = max(int(x1 + w / 2 + delta_x - size / 2), 0)
                ny1 = max(int(y1 + h / 2 + delta_y - size / 2), 0)
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                # }

                crop_box = np.array([nx1, ny1, nx2, ny2])

                ##############
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                box_ = box.reshape(1, -1)
                if IoU(crop_box, box_) >= 0.65:
                    cropped_im = img[ny1: ny2, nx1: nx2, :]
                    resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_AREA)
                    # save_file = os.path.join(pos_save_dir, "%s.jpg"%p_idx)
                    # cv2.imwrite(save_file, resized_im)
                    resized_im = (resized_im - 127.5) * 0.0078125
                    label = 1
                    roi = [offset_x1, offset_y1, offset_x2, offset_y2]
                    positive.append([resized_im, label, roi])
                    p_idx += 1
                    p_idx1 += 1
                if p_idx1>0:
                    break
        box_idx += 1
        print ("{:d} images done, positive: {:d} part: {:d} negative: {:d}".format(idx, p_idx, d_idx, n_idx))



    neg_num = 0
    while neg_num < 10:
        #{
        size = npr.randint(20, int( min(width, height) / 2 ))
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])
        #}
#method="MIN"
        iou = IoU(crop_box, boxes)

        if np.max(iou) < 0.3:   # output class label only
            cropped_im = img[ny : ny + size, nx : nx + size, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_AREA)
            # save_file = os.path.join(neg_save_dir, "%s.jpg"%n_idx)
            f2.write(r"xx\negative\%s"%n_idx + ' 0\n')
            resized_im = (resized_im - 127.5)*0.0078125
            label = 0
            neg_cls_list.append([resized_im, label])
            n_idx += 1
            neg_num += 1

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
