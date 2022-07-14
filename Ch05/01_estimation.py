# %%
import torch

# 섭씨 단위 온도 데이터를 기록 하고,
# 하나는 모르는 온도계로 데이터를 기록 한 것

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c) # 섭씨
t_u = torch.tensor(t_u) # 모르는 값
# %%

# 두 측정 값 사이에서 변환하는 역할을 해줄 모델 중 가장 단순한 것을 가정
# 둘 사이에 선형 관계가 있을지 모르니, t_u에 어떤값을 곱하고 상수를 더하면 섭씨 단위의 온도값을 얻을지도 모른다.

# 그럴 듯한 가정을 만든 것이다.
# t_c = w * t_u +b

# 일단 알 수 없는 파라미터(w, b)를 가진 모델이 있고,
# 이 파라미터의 값을 잘 추정해서 측정된 값과 예측값인 출력값 사이의
# 오차가 최대한 작아지게 만든다.

# 오차 측정을 어떻게 할지 구체적으로 정의해야 하는데,
# 손실함수(Loss funtion) 측정 함수를 만들되 
# 오차가 높으면 함수의 출력값도 높아지도록 정의하여 완벽하게 일치할 경우에는 함수의 출력값을 가능한 한 작게 만들면 된다.

# 손실 함수의 값이 최소인 지점에서 w,b를 찾는 과정을 최적화 과정이라 한다.

# %%

# 손실 함수 란?
# 훈련 샘플로부터 기대하는 출력값과 모델이 샘플에 대해 실제 출력한 값 사이의 차이를 계산한다.

# 온도계 예제로 설명하면,
# 모델이 출력한 온도인 t_p 와 계측한 값과의 차이인 t_p - t_c 이다.
# %%

# 1. 모델 정의
def model(t_u, w, b):
    return w * t_u +b

# 2. Loss funtion
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2 # 제곱하여 볼록하게 그래프를 만들어 줌
    return squared_diffs.mean() # 평균 제곱 손실 (모든 오차를 더해서 평균)

w = torch.ones(())
b = torch.zeros(())

t_p = model(t_u, w, b)
print(t_p)

loss = loss_fn(t_p, t_c)
print(loss)


# %%

delta = 0.1

loss_rate_of_chage_w = \
    (loss_fn(model(t_u, w + delta, b), t_c) -
    loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)

# 얼마 만큼 바꿔갈 것인지에 대한 값
learning_rate = 1e-2

w = w -learning_rate * loss_rate_of_chage_w

loss_rate_of_chage_b = \
    (loss_fn(model(t_u, w + delta, b), t_c) -
    loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)

b = b - learning_rate * loss_rate_of_chage_b


# %%
def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)  # 평균의 도 함수로 나눔
    return dsq_diffs

# 모델에 미분 적용하기
def dmodel_dw(t_u, w, b):
    return t_u

def dmodel_db(t_u, w, b):
    return 1.0


# 경사 함수 정의 하기
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()]) 

# 트레이닝 함수
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        w, b = params

        t_p = model(t_u, w, b)  # 순방향 전달
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)  # 역방향 전달
        params = params - learning_rate * grad

        print('Epoch %d, Loss %f' % (epoch, float(loss))) # <3>
            
    return params

# 과적합 발생
# 앞뒤로 진동하면서 조정 값이 점점 커지고, 다음 차레에는 더 심한 과잉 교정으로 이어진다.
# 최적화 과정은 불안정해지고, 수렴하는게 아닌 발산해버려서 과적합이 일어난다.
# training_loop(
#     n_epochs = 100, 
#     # learning_rate = 1e-2, 
#     learning_rate = 1e-4,
#     params = torch.tensor([1.0, 0.0]), 
#     t_u = t_u, 
#     t_c = t_c)

t_un = 0.1 * t_u

# 트레이닝 진행
training_loop(
    n_epochs = 100, 
    learning_rate = 1e-2, 
    params = torch.tensor([1.0, 0.0]), 
    t_u = t_un, # <1>
    t_c = t_c)

# %%


