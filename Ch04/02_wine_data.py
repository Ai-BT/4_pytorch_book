# %%
import csv
from debugpy import trace_this_thread
import numpy as np
import torch

# 1. 데이터 로드
wine_path = "../Ch04/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
wineq_numpy

# %%
col_list = next(csv.reader(open(wine_path), delimiter=';'))
wineq_numpy.shape, col_list

# %%
# 2. 넘파이 배열을 파이토치 텐서로 변환
wineq = torch.from_numpy(wineq_numpy)
wineq.shape, wineq.dtype

# %%
# 3. 점수 표현하기
data = wineq[:, :-1] # 모든 행과 마지막을 제외한 모든 열 선택
print(data, data.shape)

target = wineq[:, -1] # 모든 행과 마지막 열 선택 (float32)
print(target, target.shape)

target = wineq[:, -1].long() # 정수로 변환 int64
print(target)

# %%
# 4. 원핫 인코딩

target_onehot = torch.zeros(target.shape[0], 10) # 4895 x 10 의 0값을 가진 tensor 생성
target_onehot.scatter_(1, target.unsqueeze(1), 1.0) 
# 차원 축(1=가로방향(열방향), 0=아래방향(행방향)), 새로 나타낼 인덱스, 새로 저장할 입력 값
# target.unsqueeze(1) 각 와인의 점수
# 해당하는 인덱스를 모드 1.0 으로 바꿔줌

target_unsqueezed = target.unsqueeze(1)

# *** 참고 ***
# squeeze, unsqueeze
# squeeze 차원이 1인 차원을 제거
x = torch.rand(3,1,20,128)
x = x.squeeze() # [3,1,20,128] -> [3,20,128] 

# 반대로 차원이 1인 차원을 추가 (자리를 꼭 지정해줘야 한다.)
y = torch.rand(3,20,128)
y = y.unsqueeze(dim=1) # [3,20,128] -> [3,1,20,128] 

# scatter_ 메소드 역할
# 일단 이름의 마지막에 _ 이 붙어 있다.
# 파이토치에서 이름 끝에 _ 이 붙으면 새로운 텐서를 반환하는 메소드가 아니라, 
# 텐서를 바꿔치기 하는 방법으로 변경하는 메소드임을 알 수 있다.

# scatter_(축, 새로 나타낼 인덱스, 새로 저장할 입력값)

a = torch.rand(2,5)
print("a = ",a)
b = torch.zeros(3,5)

print(b.scatter_(0, torch.LongTensor([[0, 1, 2, 0, 0],[2, 0, 0, 1, 1]]),a))

# %%

# 5. 데이터 가공
data_mean = torch.mean(data, dim=0) # 11개 각 변수의 평균 
data_mean

data_var = torch.var(data, dim=0) # 11개 각 변수의 표준편차
data_var

# 정규화
data_normalized = (data - data_mean) / torch.sqrt(data_var)
data_normalized

# %%

# 6. 임계값으로 찾기
bad_indexes = target <= 3
print(bad_indexes.sum())

bad_data = data[bad_indexes] # bad 와인을 11개 변수로 다시 할당해서 20행 11열로 만듬
bad_data.shape # torch.size[(20,11)]

# 위의 방법으로 좋음, 보통, 나쁨 카테고리로 나눌 수 있다.
# 각 열에서 평균값으로 확인

bad_data = data[target <= 3]
mid_data = data[(target>3) & (target < 7)]
good_data = data[target >= 7]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

# col_list 는 열의 변수명
for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

# 데이터에서 알 수 있는 정보는 나쁜 와인은 이산화황(sulfur dioxide)가 높은 듯 하다.
# 평가 기준을 이산화황 총량을 임계값을 사용 할 수 있을 것 같다.
# 이산화황 총량에서 중앙점보다 낮은 인덱스만 가져와 보자


# %%
# 7. 중앙값보다 낮은 인덱스만 추출

# 실제 데이터 
total_sulfur_threshold = 141.83
total_sulfur_data = data[:, 6] # 모든 행과 6번째 열 만 선택
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
print(predicted_indexes.sum()) # 실제 데이터 총 4898 개의 와인 중에 2727 개가 중간보다 낮다고 할 수 있다.

# 좋은 와인의 인덱스를 뽑기
actual_indexex = target > 5
print(actual_indexex.sum()) # 실제 데이터보다 500개나 더 많이 존재하므로 완벽하지 않다는 증거

 
# 더욱 정확하게 통계적으로 가능하지만,
# 간단한 신경망을 적용하면 더욱 정확하고 한계를 뛰어 넘을 수 있다.


# 점수로 표현된 데이터를 통계적으로 접근하는 코드 이다.


# %%
