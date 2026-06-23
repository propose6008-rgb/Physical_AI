import gymnasium as gym
from stable_baselines3 import PPO

print("===== Physical AI 최적화 튜닝 학습 시작 =====")

# 환경 생성 (화면 꺼짐)
env = gym.make("HalfCheetah-v4", render_mode=None)

# 🛠️ 물리 로봇 제어에 최적화된 PPO 하이퍼파라미터 세팅
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0003,
    n_steps=2048,          # 한 번에 환경을 경험할 스텝 수 (충분한 물리 데이터 수집)
    batch_size=64,         # 신경망을 업데이트할 미니배치 크기 (안정적인 경사하강법)
    n_epochs=10,           # 수집한 데이터를 몇 번 반복해서 학습할지
    gae_lambda=0.95,       # 물리적 미래 보상에 대한 가중치 필터 (GAE)
    clip_range=0.2         # 정책이 급격하게 변해서 망가지는 것을 방지
)

# 🌟 중요: 학습량을 50만 스텝으로 대폭 늘립니다. 
# 화면이 꺼진 상태이므로 컴퓨터 사양에 따라 3~10분 내외로 완료됩니다.
total_steps = 500000
print(f"🤖 치타가 최적화된 조건으로 {total_steps}번의 초고속 학습을 시작합니다...")
model.learn(total_timesteps=total_steps)

# 모델 저장
model.save("ppo_halfcheetah_model")
print("💾 튜닝된 모델이 'ppo_halfcheetah_model.zip'로 저장되었습니다.")

env.close()
print("===== 학습 종료 =====")