import cv2
import glob
import random
import tqdm
import os
import sys

if __name__ == '__main__':
    path_data = './data/mask_no-mask_dataset'

    all_imgs = glob.glob(os.path.join(path_data, '*.jpg'))

    random.seed(10101)
    random.shuffle(all_imgs)

    f_train = open('data_train.txt', 'w')
    f_test = open('data_test.txt', 'w')

    split = 0.1 * len(all_imgs)

    for i, img_path in enumerate(tqdm.tqdm(all_imgs)):
        if i < split:
            f_test.write(f'.{img_path}\n')
        else:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            tail, head = os.path.split(img_path)
            with open(os.path.join(tail, head[:-4] + '.txt')) as f_annot:
                annot = f_annot.read().split('\n')
                annot = [l.split() for l in annot]
            f_train.write(f'.{img_path}')
            for line in annot:
                if len(line) > 0:
                    cls = line[0]
                    x_center, y_center, width, height = map(float, line[1:])
                    x_center *= w
                    y_center *= h
                    width *= w
                    height *= h
                    x_min = x_center - width/2
                    y_min = y_center - height/2
                    f_train.write(f' {round(x_min)},{round(y_min)},{round(x_min+width)},{round(y_min+height)},{cls}')
            f_train.write('\n')

    f_train.close()
    f_test.close()