import gymnasium as gym
import time

# Ant(개미) 로봇 환경을 사람이 볼 수 있는 모드로 생성합니다.
env = gym.make("Ant-v4", render_mode="human")

# 시뮬레이션을 초기화합니다.
observation, info = env.reset()

print("===== MuJoCo 데이터 추출 실습 시작 =====")

# 200스텝 동안 시뮬레이션을 돌립니다.
for step in range(200):
    # 로봇에게 무작위 행동(모터 토크)을 지시합니다.
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    # [핵심] MuJoCo 물리 엔진의 로우 데이터(Raw Data)에 접근합니다.
    # qpos는 하드웨어의 위치/자세 정보를 담고 있습니다.
    qpos = env.unwrapped.data.qpos
    # qvel은 하드웨어의 속도 정보를 담고 있습니다.
    qvel = env.unwrapped.data.qvel
    
    # 20스텝(약 0.2초)마다 한 번씩만 터미널에 데이터를 깨끗하게 출력해봅니다.
    if step % 20 == 0:
        print(f"\n[Step {step}] -----------------------------------------")
        
        # 1. 로봇 중심(Torso)의 3차원 공간 좌표 (X, Y, Z)
        # qpos의 앞선 3개 원소가 로봇 몸체의 X, Y, Z 위치를 뜻합니다.
        robot_x = qpos[0]
        robot_y = qpos[1]
        robot_z = qpos[2] # 높이
        print(f"🤖 로봇 몸체 위치 -> X: {robot_x:.3f}, Y: {robot_y:.3f}, Z (높이): {robot_z:.3f}")
        
        # 2. 로봇의 관절 각도 (하위 8개 관절)
        # Ant 로봇은 4개의 다리에 각각 2개씩 총 8개의 관절(자유도)을 가집니다.
        # qpos의 7번째 원소부터가 실제 관절들의 회전 각도(Radian)입니다.
        joint_angles = qpos[7:]
        # 소수점 둘째 자리까지만 예쁘게 출력
        readable_angles = [round(angle, 2) for angle in joint_angles]
        print(f"🦵 8개 관절 각도(rad): {readable_angles}")
        
        # 3. 로봇의 관절 회전 속도
        # qvel은 관절들이 얼마나 빠르게 움직이고 있는지(Radian/s)를 나타냅니다.
        joint_velocities = qvel[6:]
        readable_vels = [round(vel, 2) for vel in joint_velocities]
        print(f"⚡ 8개 관절 속도(rad/s): {readable_vels}")

    # 화면을 업데이트하고 잠깐 쉽니다.
    env.render()
    time.sleep(0.01)

env.close()
print("\n===== 실습 종료 =====")