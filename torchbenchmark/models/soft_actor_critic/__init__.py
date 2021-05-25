import torch
import os
import copy
import pickle
from itertools import chain

from ...util.model import BenchmarkModel
from torchbenchmark.tasks import REINFORCEMENT_LEARNING

from .config import SACConfig
from .envs import load_gym
from .sac import SACAgent
from .replay import PrioritizedReplayBuffer, ReplayBuffer
from .utils import hard_update, soft_update

def learn_standard(
    buffer,
    target_agent,
    agent,
    actor_optimizer,
    critic_optimizer,
    log_alpha_optimizer,
    target_entropy,
    batch_size,
    log_alpha,
    gamma,
    critic_clip,
    actor_clip,
    update_policy=True,
):
    per = isinstance(buffer, PrioritizedReplayBuffer)
    if per:
        batch, imp_weights, priority_idxs = buffer.sample(batch_size)
        imp_weights = imp_weights.to(device)
    else:
        batch = buffer.sample(batch_size)

    # prepare transitions for models
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
    state_batch = state_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    done_batch = done_batch.to(device)

    agent.train()
    ###################
    ## CRITIC UPDATE ##
    ###################
    alpha = torch.exp(log_alpha)
    with torch.no_grad():
        action_dist_s1 = agent.actor(next_state_batch)
        action_s1 = action_dist_s1.rsample()
        logp_a1 = action_dist_s1.log_prob(action_s1).sum(-1, keepdim=True)
        target_action_value_s1 = torch.min(
            target_agent.critic1(next_state_batch, action_s1),
            target_agent.critic2(next_state_batch, action_s1),
        )
        td_target = reward_batch + gamma * (1.0 - done_batch) * (
            target_action_value_s1 - (alpha * logp_a1)
        )

    # update critics
    agent_critic1_pred = agent.critic1(state_batch, action_batch)
    agent_critic2_pred = agent.critic2(state_batch, action_batch)
    td_error1 = td_target - agent_critic1_pred
    td_error2 = td_target - agent_critic2_pred
    critic_loss = 0.5 * (td_error1 ** 2 + td_error2 ** 2)
    if per:
        critic_loss *= imp_weights
    critic_loss = critic_loss.mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(
            chain(agent.critic1.parameters(), agent.critic2.parameters()), critic_clip
        )
    critic_optimizer.step()

    if update_policy:
        ##################
        ## ACTOR UPDATE ##
        ##################
        dist = agent.actor(state_batch)
        agent_actions = dist.rsample()
        logp_a = dist.log_prob(agent_actions).sum(-1, keepdim=True)
        actor_loss = -(
            torch.min(
                agent.critic1(state_batch, agent_actions),
                agent.critic2(state_batch, agent_actions),
            )
            - (alpha.detach() * logp_a)
        ).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        if actor_clip:
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
        actor_optimizer.step()

        ##################
        ## ALPHA UPDATE ##
        ##################
        alpha_loss = (-alpha * (logp_a + target_entropy).detach()).mean()
        log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        log_alpha_optimizer.step()

    if per:
        new_priorities = (abs(td_error1) + 1e-5).cpu().detach().squeeze(1).numpy()
        buffer.update_priorities(priority_idxs, new_priorities)

class Model(BenchmarkModel):
    task = REINFORCEMENT_LEARNING.OTHER_RL
    def __init__(self, device=None, jit=False):
        super(Model, self).__init__()
        self.device = device
        self.jit = jit
        self.args = SACConfig()
        # Construct agent
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.train_env = load_gym(self.args.env_id, self.args.seed)
        self.test_env = load_gym(self.args.env_id, self.args.seed)
        self.obs_shape = self.train_env.observation_space.shape
        self.actions_shape = self.train_env.action_space.shape
        self.agent = SACAgent(self.obs_shape[0], self.actions_shape[0],
                              self.args.log_std_low, self.args.log_std_high)
        if self.args.prioritized_replay:
            buffer_t = PrioritizedReplayBuffer
        else:
            buffer_t = ReplayBuffer
        self.buffer = buffer_t(
            self.args.buffer_size,
            state_shape=self.train_env.observation_space.shape,
            state_dtype=float,
            action_shape=(1,),
        )
        self.learning_method = "Standard"
        self.agent.to(device)
        if not self.args.self_regularized:
            # initialize target networks
            self.target_agent = copy.deepcopy(self.agent)
            self.target_agent.to(device)
            hard_update(self.target_agent.critic1, self.agent.critic1)
            hard_update(self.target_agent.critic2, self.agent.critic2)
            self.target_agent.train()
        self.critic_optimizer = torch.optim.Adam(
            chain(self.agent.critic1.parameters(), self.agent.critic2.parameters(),),
            lr=self.args.critic_lr,
            weight_decay=self.args.critic_l2,
            betas=(0.9, 0.999),
        )
        self.actor_optimizer = torch.optim.Adam(
            self.agent.actor.parameters(),
            lr=self.args.actor_lr,
            weight_decay=self.args.actor_l2,
            betas=(0.9, 0.999),
        )
        self.log_alpha = torch.Tensor([math.log(self.args.init_alpha)]).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.args.alpha_lr, betas=(0.5, 0.999))
        if not self.discrete_actions:
            self.target_entropy = -self.train_env.action_space.shape[0]
        else:
            self.target_entropy = -math.log(1.0 / self.train_env.action_space.n) * 0.98
        if self.args.self_regularized:
            # the critic target improvement ratio is annealed during training
            self.critic_target_imp_slope = (
                self.args.sr_critic_target_improvement_final - self.args.sr_critic_target_improvement_init
            ) / self.args.num_steps
            self.current_target_imp = lambda step: min(
                self.args.sr_critic_target_improvement_init + self.critic_target_imp_slope * step,
                self.args.sr_critic_target_improvement_final,
            )

    def get_module(self):
        pass
        
    def train(self, niter=1):
        # Setup
        self.target_agent.train()
        done = True
        for step in range(niter):
            if done:
                state = self.train_env.reset()
                steps_this_ep = 0
                done = False
            action = self.agent.sample_action(state)
            next_state, reward, done, info = self.train_env.step(action)
            self.buffer.push(state, action, reward, next_state, done)
            state = next_state
            steps_this_ep += 1
            if steps_this_ep >= self.args.max_episode_steps:
                done = True
            for _ in range(self.args.gradient_updates_per_step):
                learn_standard(
                    buffer=self.buffer,
                    target_agent=self.target_agent,
                    agent=self.agent,
                    actor_optimizer=self.actor_optimizer,
                    critic_optimizer=self.critic_optimizer,
                    log_alpha=self.log_alpha,
                    log_alpha_optimizer=self.log_alpha_optimizer,
                    target_entropy=self.target_entropy,
                    batch_size=self.args.batch_size,
                    gamma=self.args.gamma,
                    critic_clip=self.args.critic_clip,
                    actor_clip=self.args.actor_clip,
                    update_policy=step % self.args.actor_delay == 0,
                )

            # move target model towards training model
            if not self.args.self_regularized and (step % self.args.target_delay == 0):
                soft_update(self.target_agent.critic1, self.agent.critic1, tau)
                soft_update(self.target_agent.critic2, self.agent.critic2, tau)
    
    def eval(self, niter=1):
        with torch.no_grad():
            run.evaulate_agent(self.agent, self.env, self.args.eval_episodes, niter, render=False, verbosity=0)

