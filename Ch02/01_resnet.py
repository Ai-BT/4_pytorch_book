# %%
# 1. 사전 훈련된 모델 가져오기
from torchvision import models

print(dir(models))

# %%
# 2. AlexNet 클래스로 인스턴스 생성
alexnet = models.AlexNet()

# %%
# 3. 
# 1000개의 카테고리로 구분
# 120만개의 이미지 데이터셋인 이미지넷으로 훈련시킨 resnet101 가중치 내려 받기

resnet = models.resnet101(pretrained=True)

# renet 구조 확인
print(resnet)

# %%
# 4.

# - 이미지 전처리
# 동일한 숫자 범위 안에 색상 값이 들어올 수 있도록 크기 조정이 필요
# 기본적인 전처리 함수로 빠르게 파이프라인을 만들 수 있도록 torchvision 모듈에서 transforms 제공
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256), # 입력 이미지 256x256 조정
    transforms.CenterCrop(224), # 중심으로부터 224x224 잘라냄
    transforms.ToTensor(), # 텐서로 변환
    transforms.Normalize( # 훈련에 사용한 이미지 형식과 일치 (지정된 평균과 표준편차를 가지도록 RGB 정규화)
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225] 
    )])
# C:\Users\the35\Documents\4_pytorch_master_book\Ch02#
# C:\Users\the35\Documents\4_pytorch_master_book\Ch02
# - 이미지 불러오기
from PIL import Image
img = Image.open("../Ch02/dog.jpg")
img
# img.show()

# - 파이프 라인으로 이미지 통과시키기
img_t = preprocess(img) # torch.Size([3, 224, 224])

# - unsqueeze 
import torch
batch_t = torch.unsqueeze(img_t, 0) # torch.Size([1, 3, 224, 224])

# - 실행
# 딥러닝 사이클에서 훈련된 모델에 새로운 데이터를 넣어 결과를 보는 과정을 추론(inference) 라고 한다.
# 추론을 수행하려면 신경망을 eval 모드로 설정해야한다.
resnet.eval()



# %%
# - 연산
# 4,450만 개에 이르는 파라미터가 관련된 엄청난 연산을 실행되어
# 1000 개의 스코어를 만들어냈고 점수 하나 하나는 이미지넷 클래스에 각각 대응된다.

out = resnet(batch_t)
out

# %%
# - 레이블 찾기
with open('../Ch02/class/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# 이제 점수가 가장 높은 클래스의 레이블만 찾으면 된다.
_, index = torch.max(out, 1) 

# 참고
# 언더스코어(_) 사용은 어떤 특정값을 무시하기 위한 용도 혹은 값이 필요하지 않을때 할당한다
# 위의 코드에서 index 값만 필요

# %%
# - 결과 출력
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()


# %%

# - 다른 결과 출력 확인
_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# %%
