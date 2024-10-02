# 배에 물을 채워서 평형을 유지하는데, 바다에 마음대로 방류할 수 없음.
# 미생물 농도를 측정해서 오염도가 낮아야 물을 바다에 버릴 수 있음.
# 각종 균의 온도에 따른 생존률 continuous graph 이 주어진다.
# 1번 균을 죽이기 위한 합리적인 온도 범위를 머신러닝을 통해 예측해서 제시하라.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 가상의 온도 데이터
temperatures = np.linspace(10, 100, 100)

# 가상의 두 클래스: '균 생존'과 '균 사멸'
# 각 클래스에 대한 사전 확률 (사전 지식)
prior_survival = 0.7
prior_death = 0.3

# 클래스별 우도 함수 설정 (정규 분포 가정)
mean_survival = 40
std_survival = 10
mean_death = 65
std_death = 5

# 각 클래스의 우도 계산
likelihood_survival = norm.pdf(temperatures, mean_survival, std_survival)
likelihood_death = norm.pdf(temperatures, mean_death, std_death)

# MAP 기반 사후 확률 계산
posterior_survival = (likelihood_survival * prior_survival) / (likelihood_survival * prior_survival + likelihood_death * prior_death)
posterior_death = (likelihood_death * prior_death) / (likelihood_survival * prior_survival + likelihood_death * prior_death)

# 두 클래스의 사후 확률이 같은 임계값을 찾음
threshold = temperatures[np.argmin(np.abs(posterior_survival - posterior_death))]

# 그래프 시각화
plt.plot(temperatures, posterior_survival, label='Survival Posterior')
plt.plot(temperatures, posterior_death, label='Death Posterior')
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}°C')
plt.title('MAP-based Segmentation for Survival and Death')
plt.xlabel('Temperature (°C)')
plt.ylabel('Posterior Probability')
plt.legend()
plt.show()

print(f"최적 임계값은 {threshold:.2f}°C입니다.")
