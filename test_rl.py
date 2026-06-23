import gymnasium as gym
from stable_baselines3 import PPO
import time

print("===== 카메라 추적 기능이 탑재된 검증 시작 =====")

# 1. 환경 생성
env = gym.make("HalfCheetah-v4", render_mode="human")

# 2. 모델 로드
model = PPO.load("ppo_halfcheetah_model")

obs, info = env.reset()

# 🌟 [핵심] 카메라가 치타의 몸통(torso)을 자동으로 쫓아가도록 설정합니다.
# 이 설정을 켜면 격자 무대가 무한히 확장되는 것처럼 카메라가 이동합니다.
env.unwrapped.mujoco_renderer.viewer.cam.type = 1  # 1번 모드: tracking mode
env.unwrapped.mujoco_renderer.viewer.cam.trackbodyid = 1  # 1번 body: 치타 몸통(torso)

print("🏃‍♂️ 무한 트랙에서 치타 주행 시작!")

for step in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render()
    time.sleep(0.01)
    
    if terminated or truncated:
        break

env.close()
print("===== 검증 종료 =====")