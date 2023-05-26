# main function that sets up environments
# perform training loop
import os
os.environ["PATH"] += os.pathsep + '/opt/X11/bin'
# import envs
from buffer import ReplayBuffer
from maddpg import MADDPG
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from utilities import transpose_list, transpose_to_tensor

from pettingzoo.mpe import simple_adversary_v2

# keep training awake
from workspace_utils import keep_awake

# for saving gif
import imageio



def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def pre_process(entity, batchsize):
    processed_entity = []
    for j in range(3):
        list = []
        for i in range(batchsize):
            b = entity[i][j]
            list.append(b)
        c = torch.Tensor(list)
        processed_entity.append(c)
    return processed_entity


def main():
    seeding()
    # number of parallel agents
    parallel_envs = 1
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = 2500
    episode_length = 25
    batchsize = 1024
    # how many episodes to save policy and gif
    save_interval = 50
    # t = 0
    
    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 2
    noise_reduction = 0.999

    # how many episodes before update
    episode_per_update = 1

    log_path = os.getcwd()+"/log"
    model_dir= os.getcwd()+"/model_dir"
    
    os.makedirs(model_dir, exist_ok=True)

    env = simple_adversary_v2.env(continuous_actions=True, render_mode='rgb_array', max_cycles=episode_length)
     
    # keep 5000 episodes worth of replay
    buffer = ReplayBuffer(int(5000*episode_length))
    
    # initialize policy and critic
    maddpg = MADDPG()
    logger = SummaryWriter(log_dir=log_path)
    agent0_reward = []
    agent1_reward = []
    agent2_reward = []

    # training loop
    # show progressbar
    import progressbar as pb
    widget = ['episode: ', pb.Counter(),'/',str(number_of_episodes),' ', 
              pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ' ]
    
    timer = pb.ProgressBar(widgets=widget, maxval=number_of_episodes).start()

    # use keep_awake to keep workspace from disconnecting
    for episode in range(0, number_of_episodes):

        timer.update(episode)

        reward_this_episode = np.zeros(3)
        env.reset() #
        agents = env.agents
        num_agents = env.num_agents

        dummy_action = torch.tensor(np.zeros(5,))
        obs = [env.observe(agent) for agent in agents]
       
        for _ in range(num_agents):
            env.step(dummy_action)

        #for calculating rewards for this particular episode - addition of all time steps

        # save info or not
        save_info = (episode) % save_interval == 0
        frames = []
        tmax = 0
        
        # if save_info:
        #     frames.append(env.render('rgb_array'))
        
        for episode_t in range(episode_length):
            
            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            obs_tensor = [torch.tensor(ob, dtype=torch.float32) for ob in obs]
            actions = maddpg.act(obs_tensor, noise=noise)
            actions = [action.clip(0.0, 1.0).detach().numpy() for action in actions]

            noise *= noise_reduction
            
            # step forward one frame
            next_obs, rewards, dones, infos = [], [], [], []
            for i in range(num_agents): 
                next_ob, reward, termination, truncation, info = env.last()
                done = termination or truncation
                if done:
                    break
                next_obs.append(next_ob)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                env.step(actions[i])
            
            if done:
                break
            # print(len(actions), len(actions[0]))
            # raise
            transition = (obs, actions, rewards, next_obs, dones)
            buffer.push(transition)
            
            reward_this_episode += rewards

            obs = next_obs
            # save gif frame
            if save_info:
                frames.append(env.render())
                tmax+=1
        
        if len(buffer) > batchsize and episode % episode_per_update < parallel_envs:
            samples = buffer.sample(batchsize)
            maddpg.update(samples, a_i, logger)
            maddpg.update_targets() #soft update the target network towards the actual networks

        agent0_reward.append(reward_this_episode[0])
        agent1_reward.append(reward_this_episode[1])
        agent2_reward.append(reward_this_episode[2])

        if episode % 100 == 0 or episode == number_of_episodes-1:
            avg_rewards = [np.mean(agent0_reward), np.mean(agent1_reward), np.mean(agent2_reward)]
            agent0_reward = []
            agent1_reward = []
            agent2_reward = []
            print(np.sum(avg_rewards))
            for a_i, avg_rew in enumerate(avg_rewards):
                logger.add_scalar('agent%i/mean_episode_rewards' % a_i, avg_rew, episode)

        #saving model
        save_dict_list =[]
        if save_info:
            for i in range(3):

                save_dict = {'actor_params' : maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params' : maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params' : maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list, 
                           os.path.join(model_dir, 'episode-{}.pt'.format(episode)))
                
            # save gif files
            imageio.mimsave(os.path.join(model_dir, 'episode-{}.gif'.format(episode)), 
                            frames, duration=.04)

    env.close()
    logger.close()
    timer.finish()

if __name__=='__main__':
    main()
