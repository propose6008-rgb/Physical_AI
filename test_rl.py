import gymnasium as gym
from stable_baselines3 import PPO
import time

print("===== 학습 완료된 Physical AI 모델 검증 시작 =====")

# 1. 테스트를 위한 환경 생성 (사람이 볼 수 있도록 render_mode="human")
env = gym.make("HalfCheetah-v4", render_mode="human")

# 2. 저장했던 학습 완료 모델(.zip) 불러오기
# 아까 train_rl.py에서 저장한 모델 이름을 그대로 적어줍니다.
model = PPO.load("ppo_halfcheetah_model")
print("💾 성공적으로 학습된 모델을 불러왔습니다.")

# 3. 환경 초기화
obs, info = env.reset()

print("🏃‍♂️ 치타 로봇의 주행을 시작합니다! 창을 확인하세요.")

# 1000스텝(약 10초 이상) 동안 로봇의 주행을 관찰합니다.
for step in range(1000):
    
    # [핵심] 관측 데이터(obs)를 보고 학습된 모델이 최적의 행동(action)을 결정합니다.
    # deterministic=True는 무작위성을 배제하고 가장 똑똑한 행동만 하도록 강제하는 옵션입니다.
    action, _states = model.predict(obs, deterministic=True)
    
    # 결정된 행동을 환경에 입력합니다.
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 화면을 물리 법칙에 맞게 실시간으로 렌더링합니다.
    env.render()
    
    # 우리 눈으로 부드럽게 볼 수 있도록 살짝 타임 슬립을 줍니다.
    time.sleep(0.01)
    
    # 만약 로봇이 망가지거나 목표를 잃으면 루프를 종료합니다.
    if terminated or truncated:
        break

# 4. 환경 종료
env.close()
print("===== 검증 종료 =====")