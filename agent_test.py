import numpy as np
from pettingzoo.mpe import simple_adversary_v2

if __name__ in "__main__":
    env = simple_adversary_v2.env(continuous_actions=True)
    
    env.reset()

    num_agents = env.num_agents
    agents = env.agents
    print(env.num_agents)
    print(env.agents)
    print(env.action_space(env.agents[0]))

    # env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        print(observation)
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()
            print(action)
        env.step(action)
    env.close()

