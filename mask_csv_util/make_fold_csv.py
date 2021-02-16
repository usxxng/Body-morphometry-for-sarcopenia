# 라이브러리 불러오기
import os
import pandas as pd
import numpy as np
from data.rle_encode import rle_encode
from data.dicom_reader import *
from skimage.io import imread
import math

# 경로 지정 (폴더 위치에 따라 수정이 필요함 *현재는 바탕화면 기준)
path_nrm = "./data/dataset512/train"
path_test = "./data/dataset512/test"
path_test_mask = "./data/dataset512/mask_predicted"

file_list = os.listdir(path_nrm)
file_list_test = os.listdir(path_test)
file_list_test_mask = os.listdir(path_test_mask)

# fname,fold,exist_labels
# 1.2.276.0.7230010.3.1.4.8323329.1000.1517875165.878027.png,0,0
# 1.2.276.0.7230010.3.1.4.8323329.10001.1517875220.930580.png,4,0

# data frame 생성
df1 = pd.DataFrame(file_list, columns=['fname'])

df1['fold'] = 0
df1['exist_labels'] = 1
for i in range(len(df1)):
    df1.loc[i, 'fold'] = i%5
# csv 파일로 저장
df1 = df1.reset_index(drop=True)
df1.to_csv('./train_folds_5.csv', index=False)

# ImageId,EncodedPixels
# case183_1,1 1
# case183_2,1 1
df2 = pd.DataFrame(file_list_test, columns=['ImageId'])
df2_mask = pd.DataFrame(file_list_test_mask, columns=['ImageId'])

temp_f_index = 0
for i in range(len(df2_mask)):
    mask_image = (imread(path_test_mask + '/'+ df2_mask.loc[i, 'ImageId'])/255) * (temp_f_index%3+1)
    mask_rle = rle_encode(mask_image)
    mask_rle_string = [str(int) for int in mask_rle]
    mask_rle_s_list = ' '.join(mask_rle_string)
    df2_mask.loc[i, 'EncodedPixels'] = mask_rle_s_list

    # 여기 고쳐야함
    temp_file_name = (df2.loc[int(math.floor(temp_f_index/3)), 'ImageId'].split('.')[0])
    df2_mask.loc[i, 'ImageId'] = temp_file_name + '_' + str(temp_f_index%3+1)
    temp_f_index +=1


# csv 파일로 저장
df2_mask.to_csv('./sample_submission1.csv', index=False)




# # df1 와 df2 를 합치기
# df = pd.merge(df1, df2, how='outer')
#
# # '-'를 기준으로 왼쪽 텍스트는 Person No. , 오른쪽 텍스트는 Image No. 로 분류
# df['Person No.'] = df.File_Name.str.split('-').str[0]
# df['Image No.'] = df.File_Name.str.split('-').str[1]
