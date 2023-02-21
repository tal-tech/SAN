import os
import argparse
import glob
from tqdm import tqdm
import cv2
import pickle as pkl

parser = argparse.ArgumentParser(description='Spatial channel attention')
parser.add_argument('--images_path', default='/home/yuanye/work/data/CROHME2014/14_off_image_test', type=str, help='测试image路径')
parser.add_argument('--labels_path', default='/home/yuanye/work/data/CROHME2014/test_caption.txt', type=str, help='测试label路径')
parser.add_argument('--train_test', default='/home/yuanye/work/data/CROHME2014/test_caption.txt', type=str, help='测试train/test路径')
parser.add_argument('--image_type', default='png', type=str, help='png/jpg/bmp/jpeg/etc')
args = parser.parse_args()

image_path = args.images_path #'./train/train_images'
image_out = args.train_test + '_image.pkl'
laebl_path = args.labels_path #'./train'
label_out = args.train_test + '_label.pkl'

images = glob.glob(os.path.join(image_path, '*.'+args.image_type))
image_dict = {}

for item in tqdm(images):

    img = cv2.imread(item)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_dict[os.path.basename(item).replace('.'+args.image_type,'')] = img

with open(image_out,'wb') as f:
    pkl.dump(image_dict, f)

labels = glob.glob(os.path.join(laebl_path, '*.txt'))
label_dict = {}

for item in tqdm(labels):
    with open(item,encoding="utf8") as f:
        lines = f.readlines()
    label_dict[os.path.basename(item).replace('.txt','')] = lines

with open(label_out,'wb') as f:
    pkl.dump(label_dict, f)