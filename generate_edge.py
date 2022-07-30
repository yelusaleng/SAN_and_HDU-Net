from skimage import filters
from skimage import io
import numpy as np
import os, cv2

path = '/home/hdd_2T/coco_forgery_3w/generate_mask/'
new = '/home/hdd_2T/coco_forgery_3w/generate_edge/'

for index, i in enumerate(os.listdir(path)):
    print('{}/{}'.format(index, len(os.listdir(path))))
    img = os.path.join(path, i)
    name = i.split('.')[0]
    # name = i.replace('_mask.png', '')
    mat = io.imread(img)
    edge = filters.sobel(mat) * 255
    edge = np.asarray(edge, np.uint8)
    _, edge = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
    if not os.path.exists(new):
        os.makedirs(new)
    io.imsave(os.path.join(new, '{}.png'.format(name)), edge)
