import numpy as np
import gym
from collections import defaultdict

env = gym.make('CartPole-v0')
alpha = 0.1
gamma = 0.99
num_episodes = 10000
states = np.linspace(-0.2094,0.2094,10)

V = defaultdict(float)

def pi(state):
    action = 0 if state < 5 else 1
    return action

for episode in range(num_episodes):
    observation = env.reset()
    s = np.digitize(observation[2], states) 
    done = False
    while not done:
        a = pi(s)
        observation, r, done, _ = env.step(a)
        s_ = np.digitize(observation[2], states) 
        V[s] += alpha * (r + gamma * V[s_] - V[s]) # 현재 State = 알파 * (리워드 + 감마 * 다음 State - 현재 State)
        s = s_

        if episode % 100 == 0:
            print(f'Episode {episode}')
            for s,v in sorted(V.items()):
                print(f"\tState {s} Value = {v:.2f}")
    

