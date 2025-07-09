from ultralytics import YOLO
import os
import random
import shutil


images_dir = 'yolo/images'
labels_dir = 'yolo/labels'

out_img_train = 'dataset/images/train'
out_img_val = 'dataset/images/val'
out_lbl_train = 'dataset/labels/train'
out_lbl_val = 'dataset/labels/val'

for d in [out_img_train, out_img_val, out_lbl_train, out_lbl_val]:
    os.makedirs(d, exist_ok=True)


images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(images)


split = int(0.8 * len(images))
train_imgs = images[:split]
val_imgs = images[split:]


def move_files(image_list, img_dst, lbl_dst):
    for img in image_list:
        base = os.path.splitext(img)[0]
        lbl = base + '.txt'
        shutil.copy(os.path.join(images_dir, img), os.path.join(img_dst, img))
        shutil.copy(os.path.join(labels_dir, lbl), os.path.join(lbl_dst, lbl))


move_files(train_imgs, out_img_train, out_lbl_train)
move_files(val_imgs, out_img_val, out_lbl_val)

