import numpy as np
import pandas as pd

from glob import glob
import pydicom

import cv2
from skimage.transform import resize
from skimage.io import imread

import os
from tqdm import tqdm
import shutil

from joblib import Parallel, delayed

import argparse
import numpy as np
from rawdata.rle_encode import rle_encode
from rawdata.dicom_reader import *


def argparser():
    parser = argparse.ArgumentParser(description='Prepare png dataset for pneumatorax')
    parser.add_argument('-train_path', default='./rawdata/train', type=str, nargs='?', help='directory with train')
    parser.add_argument('-test_path', default='./rawdata/test', type=str, nargs='?', help='directory with test')
    parser.add_argument('-out_path', default='./rawdata/dataset512_ys', type=str, nargs='?',
                        help='path for saving dataset')
    parser.add_argument('-n_train', default=-1, type=int, nargs='?', help='size of train dataset')
    parser.add_argument('-img_size', default=512, type=int, nargs='?', help='image size')
    parser.add_argument('-n_threads', default=4, type=int, nargs='?', help='number of using threads')
    return parser.parse_args()


def to_binary(img, lower, upper):
    return (lower <= img) & (img <= upper)


def save_train_file(f, out_path, img_size):

    name = f.split('\\')[-1][:-4]
    # img = read_dicom(f, window_widht=400, window_level=40)
    # img = resize(img, (img_size, img_size)) * 255

    sum_im = np.zeros([img_size, img_size, 3])
    for i, wl in enumerate([-128, 0, 128]):
        img1 = read_dicom(f, window_widht=256, window_level=wl)
        img1 = resize(img1, (img_size, img_size)) * 255
        sum_im[:, :, i] = img1[:, :, 0]
    cv2.imwrite('{}/train/{}.png'.format(out_path, name), sum_im)



    # 레이블 이미지
    label_img = imread(args.train_path+ '/Label/' + name + '.png')
    encode = resize(label_img, (img_size, img_size))*255

    color_im = np.zeros([img_size, img_size, 3])
    for i in range(1,4):
        encode_ = to_binary(encode, i*1.0, i*1.0) * 255
        color_im[:, :, i-1] = encode_

    cv2.imwrite('{}/mask/{}.png'.format(out_path, name), encode)
    cv2.imwrite('{}/mask_sum/{}_c.png'.format(out_path, name), color_im)


def save_test_file(f, out_path, img_size):
    name = f.split('\\')[-1][:-4]

    sum_im = np.zeros([img_size, img_size, 3])
    for i, wl in enumerate([-128, 0, 128]):
        img1 = read_dicom(f, window_widht=256, window_level=wl)
        img1 = resize(img1, (img_size, img_size)) * 255
        sum_im[:, :, i] = img1[:, :, 0]

    cv2.imwrite('{}/test/{}.png'.format(out_path, name), sum_im)


def save_train(train_images_names, out_path, img_size=128, n_train=-1, n_threads=1):
    # if os.path.isdir(out_path):
    #     shutil.rmtree(out_path)
    os.makedirs(out_path + '/train', exist_ok=True)
    os.makedirs(out_path + '/mask', exist_ok=True)
    os.makedirs(out_path + '/mask_sum', exist_ok=True)

    if n_train < 0:
        n_train = len(train_images_names)
    try:
        Parallel(n_jobs=n_threads, backend='threading')(delayed(save_train_file)(
            f, out_path, img_size) for f in tqdm(train_images_names[:n_train]))
    except pydicom.errors.InvalidDicomError:
        print('InvalidDicomError')


def save_test(test_images_names, out_path='../dataset128', img_size=128, n_threads=1):
    os.makedirs(out_path + '/test', exist_ok=True)
    try:
        Parallel(n_jobs=n_threads, backend='threading')(delayed(save_test_file)(
            f, out_path, img_size) for f in tqdm(test_images_names))
    except pydicom.errors.InvalidDicomError:
        print('InvalidDicomError')


def main():

    train_fns = sorted(glob('{}/*/*.dcm'.format(args.train_path)))
    test_fns = sorted(glob('{}/*/*.dcm'.format(args.test_path)))
    out_path = args.out_path
    img_size = args.img_size
    n_train = args.n_train
    n_threads = args.n_threads

    save_train(train_fns, out_path, img_size, n_train, n_threads)
    save_test(test_fns, out_path, img_size, n_threads)

if __name__ == '__main__':
    args = argparser()
    main()
