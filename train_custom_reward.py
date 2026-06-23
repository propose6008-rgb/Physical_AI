import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

# 1. 커스텀 보상을 적용하기 위해 Gymnasium의 RewardWrapper를 상속받습니다.
class StraightRunningCheetahWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        # env.unwrapped.data를 통해 MuJoCo 내부의 실시간 물리 상태에 접근합니다.
        # qpos[2]는 높이, qpos[1]은 y축 회전(Pitch, 앞구르기 방향 각도) 등을 나타냅니다.
        qpos = self.env.unwrapped.data.qpos
        qvel = self.env.unwrapped.data.qvel
        
        # 1) 몸통의 기울기 (Pitch angle) 패널티
        # 치타 몸통의 회전각은 qpos[2]에 라디안(rad) 단위로 들어있습니다.
        # 정상적인 서 있는 자세는 0도 부근입니다. 앞/뒤로 너무 기울어지면 패널티를 줍니다.
        body_tilt = float(qpos[2]) 
        tilt_penalty = 0.0
        if abs(body_tilt) > 0.5:  # 약 30도 이상 기울어지기 시작하면
            tilt_penalty = -5.0 * abs(body_tilt)
            
        # 2) 앞구르기 회전 각속도 (Pitch angular velocity) 패널티
        # qvel[2]는 앞구르기 방향의 회전 속도(w)입니다. 뱅글뱅글 돌수록 값이 커집니다.
        angular_velocity = float(qvel[2])
        rotation_penalty = 0.0
        if abs(angular_velocity) > 2.0:  # 너무 빠르게 회전하면 (앞구르기 징후)
            rotation_penalty = -10.0 * abs(angular_velocity)
            
        # 3) 완전히 뒤집어지거나 머리가 땅에 박히는 극단적인 상황 패널티
        extreme_penalty = 0.0
        if abs(body_tilt) > 1.2:  # 70도 이상 뒤집어지면 앞구르기 확정이므로 대량 감점
            extreme_penalty = -50.0

        # 원래 환경이 주는 보상(전진 속도)에 우리가 만든 패널티들을 더합니다.
        custom_reward = reward + tilt_penalty + rotation_penalty + extreme_penalty
        return custom_reward

print("===== 커스텀 보상 함수 기반 Physical AI 학습 시작 =====")

# 2. 순정 환경 생성 (화면 꺼짐)
base_env = gym.make("HalfCheetah-v4", render_mode=None)

# 3. 우리가 만든 커스셜 보상 규칙으로 환경을 감쌉니다(Wrapping)
env = StraightRunningCheetahWrapper(base_env)

# PPO 모델 설정
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gae_lambda=0.95,
    clip_range=0.2
)

# 50만 스텝 동안 새 규칙 아래에서 학습시킵니다.
total_steps = 500000
print(f"🤖 '구르면 벌점' 규칙을 배운 치타가 {total_steps}번의 초고속 학습을 시작합니다...")
model.learn(total_timesteps=total_steps)

# 새 모델 저장
model.save("ppo_halfcheetah_custom_model")
print("💾 얌전하게 달리는 치타 모델이 'ppo_halfcheetah_custom_model.zip'로 저장되었습니다.")

env.close()