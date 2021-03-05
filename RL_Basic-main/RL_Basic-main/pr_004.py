import numpy as np

class K_armed_bandit:
    def __init__(self):
        self.p1 = [0.65, 0.35]
        self.p3 = [0.9, 0.1]

    def pull(self, arm):
        if arm == 0: # 10, 28을 65, 35 프로의 확률로 
            return np.random.choice([10,28], 1, p = self.p1)
        elif arm ==1: # 5 ~ 13 까지 균일한 랜덤 
            return np.random.choice(range(5,14), 1)
        elif arm == 2:
            return np.random.choice([11,87],1, p = self.p3)
        else:
            print(f"Invalid arm {arm}")
            return 
class Agent:
    def __init__(self, num_arms =3):
        self.num_arms = num_arms 
        self.best_arm = np.argmax(np.random.rand(num_arms))
        print(f"초기 best arm = {self.best_arm}")
    
    def random_ro_predict(self, epsilon):
        if np.random.rand() < epsilon: # epsilon 값 이하 일때 다른걸 써봄 
            return np.random.randint(self.num_arms)
        else:
            return self.best_arm # 아니면 가장 좋은거 씀

env = K_armed_bandit()
agent = Agent()
EPSILON = 0.3 
num_arms = 3

num_iter = [10, 100, 1000, 10000]

for iters in num_iter:
    arm_rewards = np.zeros(num_arms)
    arm_selected = np.zeros(num_arms)

    for _ in range(iters):
        selected = agent.random_ro_predict(EPSILON)
        reward = env.pull(selected)

        arm_rewards[selected] += reward
        arm_selected[selected] += 1

    agent.best_arm = np.argmax(arm_rewards/arm_selected)

    print(f"final best arm은 {agent.best_arm} when {iters}")

    for i, (reward,selected) in enumerate(zip(arm_rewards,arm_selected)):
        print(f"평균 reward of {i} {reward/selected}")
