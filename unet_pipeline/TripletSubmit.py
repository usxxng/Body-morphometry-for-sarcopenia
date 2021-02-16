import argparse
import pickle
from tqdm import tqdm
from pathlib import Path
import os

import cv2

import numpy as np
import pandas as pd
from collections import defaultdict

from utils.mask_functions import mask2rle, rle_encode
from utils.helpers import load_yaml

def argparser():
    parser = argparse.ArgumentParser(description='Body Morp pipeline')
    parser.add_argument('cfg', type=str, help='experiment name')
    return parser.parse_args()

def extract_largest(mask, n_objects):
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    areas = [cv2.contourArea(c) for c in contours]
    contours = np.array(contours)[np.argsort(areas)[::-1]]
    background = np.zeros(mask.shape, np.uint8)
    choosen = cv2.drawContours(
        background, contours[:n_objects],
        -1, (255), thickness=cv2.FILLED
    )
    return choosen

def remove_smallest(mask, min_contour_area):
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    background = np.zeros(mask.shape, np.uint8)
    choosen = cv2.drawContours(
        background, contours,
        -1, (255), thickness=cv2.FILLED
    )
    return choosen


def apply_thresholds(mask, n_objects, area_threshold, top_score_threshold,
                     bottom_score_threshold, leak_score_threshold, use_contours, min_contour_area, name_i):
    # if n_objects == 1:
    #     crazy_mask = (mask > top_score_threshold).astype(np.uint8)
    #     if crazy_mask.sum() < area_threshold:
    #         return -1
    #     mask = (mask > bottom_score_threshold).astype(np.uint8)
    # else:
    #
    mask = mask.astype(np.uint8) * 255

    # if min_contour_area > 0:
    #     choosen = remove_smallest(mask, min_contour_area)
    # elif use_contours:
    #     choosen = extract_largest(mask, n_objects)
    # else:
    #     choosen = mask * 255

    if mask.shape[0] == 512:
        reshaped_mask = mask
    else:
        reshaped_mask = cv2.resize(
            mask,
            dsize=(512, 512),
            interpolation=cv2.INTER_LINEAR
        )

    reshaped_mask = (reshaped_mask > 63).astype(int) * 255
    # cv2.imwrite(name_i + '_.png', reshaped_mask)
    return rle_encode(reshaped_mask)

def remove_smallest_multiclass(mask, min_contour_area, name_i):


    mask_uint8 = (mask*255).astype(np.uint8)

    num_class = mask_uint8.shape[0]
    mask_larges = np.zeros(mask.shape, np.uint8)

    min_contours = {}

    # 1st-pass 작은 영역 지워내기
    for i in range(0, num_class):
        mask_i = mask_uint8[i, :, :]

        # contour 찾기
        contours, _ = cv2.findContours(mask_i.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 영역을 크기별로 구분함 - min은 2nd-pass를 위해 저장
        min_contours[i] = [c for c in contours if cv2.contourArea(c) <= min_contour_area]
        max_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

        # if (name_i == 'case209'):
        #     # 영역을 크기별로 구분함 - min은 2nd-pass를 위해 저장
        #     min_contours[i] = [c for c in contours if cv2.contourArea(c) <= 1100]
        #     max_contours = [c for c in contours if cv2.contourArea(c) > 1100]

        # 영역 따로 그리기
        mask_larges[i,:,:] = cv2.drawContours(mask_larges[i,:,:], max_contours,-1, (255), thickness=cv2.FILLED)


    # 2nd-pass - 겹치는 영역이 젤 많은곳으로 컨투어를 보낸다.
    for i in range(0, num_class):
        # 작은 조각 하나씩 둘러보면서 적용하기
        min_contours_ = min_contours[i]

        for contour in min_contours_:
            mask_larges_ = mask_larges.copy()/255

            # 작은 영역은 그린 후 팽창(dilate) 시킨다
            mask_small = cv2.drawContours(np.zeros([mask_uint8.shape[1],mask_uint8.shape[2]], np.uint8), contour,-1, (255), thickness=cv2.FILLED)
            mask_small_dilate = cv2.dilate(mask_small, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=3)

            # 작은영역 boolean mask
            mask_small_dilate = (mask_small_dilate>0)
            # 큰 영역의 확률맵
            prob_large_contour = mask * mask_larges_

            # 확률이 제일 높은 영역의 채널 번호를 가져온다.
            score = []
            for c in range(0, num_class):
                # 탐색하는곳과 원본 class가 같아도, small은 이미 0이므로 그대로 진행
                score.append((prob_large_contour[c, :, :] * mask_small_dilate).sum())
            # 최고 점수가 나온 채널 획득
            idxes = np.argwhere(score == np.amax(score))

            # 높은 영역의 채널로 small contour를 편입시킨다.
            mask_larges[idxes[0], :, :] += mask_small
    # k=1
    # mask_uint8[k, :, :]
    # mask_larges[k,:,:]

    output = mask_larges.astype(np.float64)/255.0

    return output



def build_rle_dict(mask_dict, n_objects_dict,  
                   area_threshold, top_score_threshold,
                   bottom_score_threshold,
                   leak_score_threshold, 
                   use_contours, min_contour_area, sub_img_path):
    rle_dict = {}

    for name, mask in tqdm(mask_dict.items()):

        # 물체 개수를 판단 (채널이 다르므로 늘 1개)
        # TODO: 대회를 위한 후처리 하드코딩
        #  class 개수 4개 설정
        num_class = 4

        if mask.shape[1] != 512:
            # 마스크 리사이즈
            reshaped_mask = np.zeros([4, 512, 512])
            for i in range(0, num_class):
                reshaped_mask[i,:,:] = cv2.resize(mask[i,:,:], dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
            mask = reshaped_mask

        max_mask = (mask.max(axis=0,keepdims=1) == mask) * 1.0
        mask_123 = np.zeros([max_mask.shape[1], max_mask.shape[2]])

        # rgb 형태 마스크 저장 (확률 반영) - 전처리 없음
        max_masked = mask * max_mask
        rgb_image = np.transpose((max_masked[1:4, :, :]/np.max(max_masked[1:4, :, :])) * 255, (1, 2, 0))
        cv2.imwrite(sub_img_path + name + '_rgb.png', rgb_image)

        # 레이블 형태 마스크 저장
        for i in range(1, num_class):
            # 데이터 이름에 _1,_2,_3 붙이기
            name_i = name + f'_{i}'
            n_objects = n_objects_dict.get(name_i, 0)
            mask_123 = mask_123 + max_mask[i,:,:]*i
        # cv2.imwrite(sub_img_path + name + '.png', mask_123)

        # 레이블 영역 thresholding
        if min_contour_area > 0:
            mask_postproc = remove_smallest_multiclass(max_masked, min_contour_area, name)
            # rgb 형태 마스크 저장 (확률 반영) - 전처리 없음
            rgb_image = np.transpose(mask_postproc[1:4, :, :] * 255, (1, 2, 0))
            cv2.imwrite(sub_img_path + name + '_rgb_masked.png', rgb_image)
        else:
            mask_postproc = max_masked

        # 레이블 rle로 저장
        for i in range(1, num_class):
            # 데이터 이름에 _1,_2,_3 붙이기
            name_i = name + f'_{i}'
            mask_i = mask_postproc[i, :, :]

            # 마스크는 0아니면 1
            rle_dict[name_i] = apply_thresholds(
                mask_i, n_objects,
                area_threshold, top_score_threshold,
                bottom_score_threshold,
                leak_score_threshold,
                use_contours, min_contour_area, sub_img_path + name_i
            )

    return rle_dict

def buid_submission(rle_dict, sample_sub):
    sub = pd.DataFrame.from_dict([rle_dict]).T.reset_index()
    sub.columns = sample_sub.columns
    sub.loc[sub.EncodedPixels == '', 'EncodedPixels'] = -1
    return sub

def load_mask_dict(cfg):
    reshape_mode = cfg.get('RESHAPE_MODE', False)
    if 'MASK_DICT' in cfg:
        result_path = Path(cfg['MASK_DICT'])
        with open(result_path, 'rb') as handle:
            mask_dict = pickle.load(handle)
        return mask_dict
    if 'RESULT_WEIGHTS' in cfg:
        result_weights = cfg['RESULT_WEIGHTS']
        mask_dict = defaultdict(int)
        for result_path, weight in result_weights.items():
            print(result_path, weight)
            with open(Path(result_path), 'rb') as handle:
                current_mask_dict = pickle.load(handle)
                for name, mask in current_mask_dict.items():
                    if reshape_mode and mask.shape[1] != 512:
                        reshaped_mask = np.zeros([mask.shape[0],512,512])
                        for c in range(mask.shape[0]):
                            reshaped_mask[c,:,:] = cv2.resize(mask[c,:,:], dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
                        mask = reshaped_mask
                    #crazy_mask = (mask > 0.75).astype(np.uint8)
                    #if crazy_mask.sum() < 1000:
                    #  mask = np.zeros_like(mask)
                    mask_dict[name] = mask_dict[name] + mask * weight
        return mask_dict


def main():
    args = argparser()
    # config_path = 'experiments/albunet_public/05_submit.yaml' #
    config_path = Path(args.cfg.strip("/"))
    sub_config = load_yaml(config_path)
    print(sub_config)
    
    sample_sub = pd.read_csv(sub_config['SAMPLE_SUB'])
    n_objects_dict = sample_sub.ImageId.value_counts().to_dict()
    
    print('start loading mask results....')
    mask_dict = load_mask_dict(sub_config)
    
    use_contours = sub_config['USECONTOURS']
    min_contour_area = sub_config.get('MIN_CONTOUR_AREA', 0)

    area_threshold = sub_config['AREA_THRESHOLD']
    top_score_threshold = sub_config['TOP_SCORE_THRESHOLD']
    bottom_score_threshold = sub_config['BOTTOM_SCORE_THRESHOLD']
    if sub_config['USELEAK']:
        leak_score_threshold = sub_config['LEAK_SCORE_THRESHOLD']
    else:
        leak_score_threshold = bottom_score_threshold

    sub_file = Path(sub_config['SUB_FILE'])
    sub_img_path = '../subs/' + sub_file.parts[-1][:-4] + '/'
    if not os.path.exists(sub_img_path):
        os.mkdir(sub_img_path)

    rle_dict = build_rle_dict(
        mask_dict, n_objects_dict, area_threshold,
        top_score_threshold, bottom_score_threshold,
        leak_score_threshold, use_contours, min_contour_area, sub_img_path
    )
    sub = buid_submission(rle_dict, sample_sub)
    print((sub.EncodedPixels != -1).sum())
    print(sub.head())


    sub.to_csv(sub_file, index=False)

if __name__ == "__main__":
    main()
