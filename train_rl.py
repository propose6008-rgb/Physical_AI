import gymnasium as gym
from stable_baselines3 import PPO

print("===== Physical AI 초고속 학습 시작 =====")

# 🌟 중요: render_mode를 None으로 하여 화면 연산 부하를 제로로 만듭니다.
env = gym.make("HalfCheetah-v4", render_mode=None)

# 모델 생성
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# 🌟 중요: 학습량을 20만 스텝으로 늘립니다. 화면이 꺼져서 생각보다 금방 끝납니다!
print("🤖 치타가 대뇌 속에서 초고속으로 20만 번의 시뮬레이션을 돌리며 달리는 법을 배웁니다...")
model.learn(total_timesteps=200000)

# 모델 저장
model.save("ppo_halfcheetah_model")
print("💾 학습 완료! 'ppo_halfcheetah_model.zip'로 저장되었습니다.")

env.close()