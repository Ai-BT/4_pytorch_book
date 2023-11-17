# %%
import torch
import numpy as np

# 1. 데이터 로드
# 각 행은 시간별로 분리된 데이터
bikes_numpy = np.loadtxt(
    "../Ch04/hour-fixed.csv",
    dtype=np.float32,
    delimiter=",",
    skiprows=1,
    converters={1: lambda x: float(x[8:10])},  # 첫번째 열의 일자 문자열을 숫자로 변환
)
bikes = torch.from_numpy(bikes_numpy)
bikes

# %%
# 2. 일별로 매 시간의 데이터 셋을 구하기 위해 동일 텐서를 24시간 배치로 바라보는 뷰가 필요
bikes.shape, bikes.stride()  # 17520 시간에 17개 열

# %%
# 3. 데이터를 일자, 시간, 17개 열의 세 개 축으로 만들기
daily_bikes = bikes.view(-1, 24, bikes.shape[1])
daily_bikes.shape, daily_bikes.stride()

daily_bikes = daily_bikes.transpose(1, 2)
daily_bikes.shape, daily_bikes.stride()


# %%
# 4. 훈련준비
# 날씨 상태를 나타내는 데이터는 순서값 총 4단계 1은 좋고 4는 안좋고
# 이 값을 카테고리로 간주해 각 단계를 레이블로 볼 수 있고 동시에 연속값으로 볼 수 있다.

first_day = bikes[:24].long()
weater_onehot = torch.zeros(first_day.shape[0], 4)
first_day[:, 9]  # 하루 시간대의 날씨 상태을 나열


# %%
# 5. 날씨 수준에 따라 행렬을 원핫 인코딩으로 변한
weater_onehot.scatter_(
    dim=1,
    index=first_day[:, 9].unsqueeze(1).long()
    - 1,  # 날씨 상태는 1 부터 4까지 지만 인덱스는 0부터 시작하므로 -1
    value=1.0,
)

torch.cat((bikes[:24], weater_onehot), 1)[:1]
# %%
daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2])
print(daily_weather_onehot.shape)

daily_weather_onehot.scatter_(1, daily_bikes[:, 9, :].long().unsqueeze(1) - 1, 1.0)
print(daily_weather_onehot.shape)

daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1)

# 레이블 자체에 순서 관계가 있으므로 연속 변수에 해당하는 특수값으로 가장도 가능
# 값을 변환해서 0.0 에서 1.10 으로 바꿀 수 있다
daily_bikes[:, 9, :] = (daily_bikes[:, 9, :] - 1.0) / 3.0


# %%

# 0.0 에서 1.0 으로 매핑
temp = daily_bikes[:, 10, :]
temp_min = torch.min(temp)
temp_max = torch.max(temp)
daily_bikes[:, 10, :] = (daily_bikes[:, 10, :] - temp_min) / (temp_max - temp_min)

# 모든 값에서 평균을 빼고 표준편차로 나누기도 한다.
temp = daily_bikes[:, 10, :]
daily_bikes[:, 10, :] = (daily_bikes[:, 10, :] - torch.mean(temp)) / torch.std(temp)
