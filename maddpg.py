# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import numpy as np
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
device = torch.device("cpu")


# HIDDEN_IN_ACTOR = 256
# HIDDEN_OUT_ACTOR = 128
# OUT_ACTOR = 5

# # critic input = obs_full + actions = 8+10+10+5+5+5=43
# IN_CRITIC = 43
# HIDDEN_IN_CRITIC = 512
# HIDDEN_OUT_CRITIC = 256

HIDDEN_IN_ACTOR = 256
HIDDEN_OUT_ACTOR = 128
OUT_ACTOR = 5

# critic input = obs_full + actions = 8+10+10+5+5+5=43
IN_CRITIC = 43
HIDDEN_IN_CRITIC = 512
HIDDEN_OUT_CRITIC = 256


class MADDPG:
    def __init__(self, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 28+5+5+5=43
#        
        self.maddpg_agent = [DDPGAgent(8, HIDDEN_IN_ACTOR, HIDDEN_OUT_ACTOR, OUT_ACTOR, IN_CRITIC, HIDDEN_IN_CRITIC, HIDDEN_OUT_CRITIC), 
                             DDPGAgent(10, HIDDEN_IN_ACTOR, HIDDEN_IN_ACTOR, OUT_ACTOR, IN_CRITIC, HIDDEN_IN_CRITIC, HIDDEN_OUT_CRITIC), 
                             DDPGAgent(10, HIDDEN_IN_ACTOR, HIDDEN_IN_ACTOR, OUT_ACTOR, IN_CRITIC, HIDDEN_IN_CRITIC, HIDDEN_OUT_CRITIC)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions
    


    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """

        obs, action, reward, next_obs, done = samples

        obs_full = torch.concatenate(obs, axis=1)
        next_obs_full = torch.concatenate(next_obs, axis=1)
    
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        
        target_critic_input = torch.cat((next_obs_full, target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1).int())
        action = torch.cat(action, dim=1)
        critic_input = torch.cat((obs_full, action), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full, q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




