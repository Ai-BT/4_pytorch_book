# %%
import imageio

img_arr = imageio.imread("../Ch04/bobby.jpg")
img_arr.shape # [720, 1280, 3]

# pytorch 모듈은 텐서가 C x H x W 채널 높이 너비 순으로 배치되어야 한다

# %%
# 1. 하나의 이미지 레이아웃 변경
import torch

img = torch.from_numpy(img_arr) # 일반형 -> torch 텐서로 변환
out = img.permute(2, 0, 1) # [3, 720, 1280]

# permute 치환하다
x = torch.rand(16, 32, 3)
y = x.transpose(0, 2) # [3, 32, 16] 0번째 2번째 순서 바꿈
z = x.permute(2, 1, 0) # [3, 32, 16] 해당 인덱스 순서를 바꿈

# %%

# 2. 여러 장의 이미지 레이아웃 변경 (batch)

# 채널 높이 너비 (C H W)
batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

import os

data_dir = '../Ch04/image-cats/'

filenames = [name for name in os.listdir(data_dir)
            if os.path.splitext(name)[-1] == '.png']

# enumerate filename 에 인덱스 할당
for i, filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2,0,1) # C H W
    img_t[:3] # RGB 채널 3개만 유지 (3, 256, 256)
    batch[i] = img_t

# *** 참고 ***
# 파일 확장자 반환 - os.path.splitext(name)
# for name in os.listdir(data_dir):
#     print(os.path.splitext(name))
# ('cat1', '.png')
# ('cat2', '.png')
# ('cat3', '.png')

# emuerate
# 인덱스와 원소를 동시에 접근하면서 루프를 돌릴 수 있는 방법
# for entry in enumerate(['a', 'b', 'c']):
#     print(entry)

# letters = ['a', 'b', 'c']
# for i in range(len(letters)):
#     letter = letters[i]
#     print(i, letter)


# %%
