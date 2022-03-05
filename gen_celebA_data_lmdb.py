import numpy as np
import cv2
import os
import numpy.random as npr
import _pickle as pickle

image_size = 24
anno_file = r"D:\graduatestudy\DIRECTION\MTCNN\dataset\CELE\list_bbox_celeba.txt"
landmark_file = r"D:\graduatestudy\DIRECTION\MTCNN\dataset\CELE\list_landmarks_celeba.txt"
im_dir = r"D:\graduatestudy\DIRECTION\MTCNN\dataset\CELE\img_celeba.7z\img_celeba"

with open(anno_file, 'r') as f:
    anno = f.readlines()
with open(landmark_file, 'r') as f2:
    lpl = f2.readlines()
##with open(r'imdb\12\positive.imdb', 'rb') as fid:
##    positive = pickle.load(fid)
annotations=[]
landmark_positions_list=[]
pts_keep = npr.choice(len(anno[2:]), size=64000, replace=False)#size depend on positive size
for i in pts_keep:
    annotations.append(anno[i+2])
    landmark_positions_list.append(lpl[i+2])
print(len(pts_keep), "images total")
img_idx = 0
pts_list=[]

for idx,annotation in enumerate(annotations):
    pts = [0]*10
    annotation = annotation.strip().split()
    im_path = annotation[0]
    bbox = list(map(float, annotation[1:]))
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    landmark_positions = landmark_positions_list[idx]
    landmark_positions = landmark_positions.strip().split()
    pts_list_raw = list(map(float, landmark_positions[1:]))   # need to convert to relative pts positions

    file_basename = os.path.splitext(im_path)[0]
    img = cv2.imread(os.path.join(im_dir, file_basename + '.jpg'))  # the celebA database seems to have used wrong name
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_idx += 1
    if img_idx % 100 == 0:
        print (img_idx, "images done")


    for box in boxes:
        x1, y1, w, h = box
        x2 = w + x1 -1
        y2 = h + y1 -1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        cropped_im = img[int(y1): int(y2), int(x1): int(x2), :]
        resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_AREA)
        im = (resized_im - 127.5)*0.0078125
        pts_xs = pts_list_raw[0::2]
        pts_ys = pts_list_raw[1::2]
        pts_xs_real = [(tmpval - x1 )/w for tmpval in pts_xs]
        pts_ys_real = [(tmpval - y1 )/h for tmpval in pts_ys]
        pts[0::2] = pts_xs_real
        pts[1::2] = pts_ys_real
        pts_list.append([im, pts])

fid = open("imdb/{0:d}/pts_v1.imdb".format(image_size),'wb')
pickle.dump(pts_list, fid)
fid.close()
