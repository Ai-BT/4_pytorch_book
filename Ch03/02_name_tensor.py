# %%
import torch

# channels, rows, colums
# 정육면체 형태 3차원
img_t = torch.randn(3, 5, 5)
weights = torch.tensor([0.2126, 0.7152, 0.0722])

print(img_t)
print(weights)

# %%

# 정육면체가 2개 있는 4차원
batch_t = torch.randn(2, 3, 5, 5)
print(batch_t)

# %%

# -3 위치에 배열(행렬)에 대한 평균값
img_grey_naive = img_t.mean(-3)
batch_grey_naive = batch_t.mean(-3)
print(img_grey_naive.shape, batch_grey_naive)

# %%

# unsqueeze
# 마지막 차원에 새로운 차원을 추가 하는 것
unsqueezed_weights_01 = weights.unsqueeze(-1)
unsqueezed_weights_02 = weights.unsqueeze(-1).unsqueeze_(-1)

print(unsqueezed_weights_01)
print(unsqueezed_weights_02)
print(unsqueezed_weights_01.shape)
print(unsqueezed_weights_02.shape)


# unsqueezed_weights_01 = weights.unsqueeze(-1):
# 이 코드는 weights 텐서에 새로운 차원을 추가하여 unsqueezed_weights_01에 저장합니다.
# -1을 사용하여 가장 오른쪽(마지막) 차원에 새로운 차원을 추가합니다.

# unsqueezed_weights_02 = weights.unsqueeze(-1).unsqueeze_(-1):
# 여기서는 unsqueeze(-1)을 두 번 연속으로 적용하여 두 번째 차원까지 추가합니다.
# 첫 번째 unsqueeze(-1)은 마지막 차원에 새로운 차원을 추가하고,
# 두 번째 unsqueeze_(-1)은 이전에 만들어진 텐서의 마지막 차원에 새로운 차원을 추가합니다.

# %%
