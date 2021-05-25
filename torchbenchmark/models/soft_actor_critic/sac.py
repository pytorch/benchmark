import argparse
import copy
import math
import os
from itertools import chain

import numpy as np
import tensorboardX
import torch
import torch.nn.functional as F
import tqdm

from . import envs, nets, replay, run, utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACAgent:
    def __init__(
        self,
        obs_space_size,
        act_space_size,
        log_std_low,
        log_std_high,
        actor_net_cls=nets.StochasticActor,
        critic_net_cls=nets.BigCritic,
        hidden_size=1024,
    ):
        self.actor = actor_net_cls(
            obs_space_size,
            act_space_size,
            log_std_low,
            log_std_high,
            dist_impl="pyd",
            hidden_size=hidden_size,
        )
        self.critic1 = critic_net_cls(obs_space_size, act_space_size, hidden_size)
        self.critic2 = critic_net_cls(obs_space_size, act_space_size, hidden_size)

    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic1 = self.critic1.to(device)
        self.critic2 = self.critic2.to(device)

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def save(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic1_path = os.path.join(path, "critic1.pt")
        critic2_path = os.path.join(path, "critic2.pt")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)

    def load(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic1_path = os.path.join(path, "critic1.pt")
        critic2_path = os.path.join(path, "critic2.pt")
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic1.load_state_dict(torch.load(critic1_path))
        self.critic2.load_state_dict(torch.load(critic2_path))

    def forward(self, state, from_cpu=True):
        if from_cpu:
            state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.mean
        self.actor.train()
        if from_cpu:
            act = self.process_act(act)
        return act

    def sample_action(self, state, from_cpu=True):
        if from_cpu:
            state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = act_dist.sample()
        self.actor.train()
        if from_cpu:
            act = self.process_act(act)
        return act

    def process_state(self, state):
        return torch.from_numpy(np.expand_dims(state, 0).astype(np.float32)).to(
            utils.device
        )

    def process_act(self, act):
        return np.squeeze(act.clamp(-1.0, 1.0).cpu().numpy(), 0)


class SACDAgent(SACAgent):
    def __init__(self, obs_space_size, act_space_size):
        self.actor = nets.BaselineDiscreteActor(obs_space_size, act_space_size)
        self.critic1 = nets.BaselineDiscreteCritic(obs_space_size, act_space_size)
        self.critic2 = nets.BaselineDiscreteCritic(obs_space_size, act_space_size)

    def forward(self, state):
        state = self.process_state(state)
        self.actor.eval()
        with torch.no_grad():
            act_dist = self.actor.forward(state)
            act = torch.argmax(act_dist.probs, dim=1)
        self.actor.train()
        return self.process_act(act)


def sac(
    agent,
    buffer,
    train_env,
    test_env,
    num_steps=1_000_000,
    transitions_per_step=1,
    max_episode_steps=100_000,
    batch_size=512,
    tau=0.005,
    actor_lr=1e-4,
    critic_lr=1e-4,
    alpha_lr=1e-4,
    gamma=0.99,
    eval_interval=5000,
    eval_episodes=10,
    warmup_steps=1000,
    actor_clip=None,
    critic_clip=None,
    actor_l2=0.0,
    critic_l2=0.0,
    target_delay=2,
    actor_delay=1,
    save_interval=100_000,
    name="sac_run",
    render=False,
    save_to_disk=True,
    log_to_disk=True,
    verbosity=0,
    gradient_updates_per_step=1,
    init_alpha=0.1,
    discrete_actions=False,
    self_regularized=False,
    sr_max_critic_updates_per_step=20,
    sr_critic_target_improvement_init=0.7,
    sr_critic_target_improvement_final=0.9,
    infinite_bootstrap=True,
    **kwargs,
):
    """
    Train `agent` on `train_env` with Soft Actor Critic algorithm, ane evaluate on `test_env`.

    Reference: https://arxiv.org/abs/1801.01290 and https://arxiv.org/abs/1812.05905

    Also supports discrete action spaces (ref: https://arxiv.org/abs/1910.07207), 
    and self-regularization (ref: https://arxiv.org/abs/2009.08973v1), which
    eliminates the need for target networks and the tau hyperparameter.
    """
    ######################
    ## VALIDATE HPARAMS ##
    ######################
    assert not (self_regularized and discrete_actions)
    assert not self_regularized or (
        sr_critic_target_improvement_final >= sr_critic_target_improvement_init
    )

    if (not discrete_actions) and (not self_regularized):
        learning_method = "Standard"
    elif (discrete_actions) and (not self_regularized):
        learning_method = "Discrete"
    elif (not discrete_actions) and (self_regularized):
        learning_method = "Self Regularized"

    ################
    ## PRINT INFO ##
    ################
    if verbosity:
        print(f"Deep Control Soft Actor Critic with Learning Method: {learning_method}")

    if save_to_disk or log_to_disk:
        save_dir = utils.make_process_dirs(name)
    if log_to_disk:
        # create tb writer, save hparams
        writer = tensorboardX.SummaryWriter(save_dir)
        writer.add_hparams(locals(), {})

    ###########
    ## SETUP ##
    ###########
    agent.to(device)
    agent.train()
    if not self_regularized:
        # initialize target networks
        target_agent = copy.deepcopy(agent)
        target_agent.to(device)
        utils.hard_update(target_agent.critic1, agent.critic1)
        utils.hard_update(target_agent.critic2, agent.critic2)
        target_agent.train()
    # set up optimizers
    critic_optimizer = torch.optim.Adam(
        chain(agent.critic1.parameters(), agent.critic2.parameters(),),
        lr=critic_lr,
        weight_decay=critic_l2,
        betas=(0.9, 0.999),
    )
    actor_optimizer = torch.optim.Adam(
        agent.actor.parameters(),
        lr=actor_lr,
        weight_decay=actor_l2,
        betas=(0.9, 0.999),
    )
    log_alpha = torch.Tensor([math.log(init_alpha)]).to(device)
    log_alpha.requires_grad = True
    log_alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr, betas=(0.5, 0.999))
    if not discrete_actions:
        target_entropy = -train_env.action_space.shape[0]
    else:
        target_entropy = -math.log(1.0 / train_env.action_space.n) * 0.98
    if self_regularized:
        # the critic target improvement ratio is annealed during training
        critic_target_imp_slope = (
            sr_critic_target_improvement_final - sr_critic_target_improvement_init
        ) / num_steps
        current_target_imp = lambda step: min(
            sr_critic_target_improvement_init + critic_target_imp_slope * step,
            sr_critic_target_improvement_final,
        )

    ###################
    ## TRAINING LOOP ##
    ###################
    # warmup the replay buffer with random actions
    run.warmup_buffer(buffer, train_env, warmup_steps, max_episode_steps)
    done = True
    steps_iter = range(num_steps)
    if verbosity:
        steps_iter = tqdm.tqdm(steps_iter)
    for step in steps_iter:
        # collect experience
        for _ in range(transitions_per_step):
            if done:
                state = train_env.reset()
                steps_this_ep = 0
                done = False
            action = agent.sample_action(state)
            next_state, reward, done, info = train_env.step(action)
            if infinite_bootstrap:
                # allow infinite bootstrapping
                if steps_this_ep + 1 == max_episode_steps:
                    done = False
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            steps_this_ep += 1
            if steps_this_ep >= max_episode_steps:
                done = True

        for _ in range(gradient_updates_per_step):
            if learning_method == "Standard":
                learn_standard(
                    buffer=buffer,
                    target_agent=target_agent,
                    agent=agent,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    log_alpha=log_alpha,
                    log_alpha_optimizer=log_alpha_optimizer,
                    target_entropy=target_entropy,
                    batch_size=batch_size,
                    gamma=gamma,
                    critic_clip=critic_clip,
                    actor_clip=actor_clip,
                    update_policy=step % actor_delay == 0,
                )
            elif learning_method == "Discrete":
                learn_discrete(
                    buffer=buffer,
                    target_agent=target_agent,
                    agent=agent,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    log_alpha=log_alpha,
                    log_alpha_optimizer=log_alpha_optimizer,
                    target_entropy=target_entropy,
                    batch_size=batch_size,
                    gamma=gamma,
                    critic_clip=critic_clip,
                    actor_clip=actor_clip,
                    update_policy=step % actor_delay == 0,
                )
            elif learning_method == "Self Regularized":
                learn_self_regularized(
                    buffer=buffer,
                    agent=agent,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    log_alpha=log_alpha,
                    log_alpha_optimizer=log_alpha_optimizer,
                    target_entropy=target_entropy,
                    critic_target_improvement=current_target_imp(step),
                    max_critic_updates_per_step=sr_max_critic_updates_per_step,
                    batch_size=batch_size,
                    gamma=gamma,
                    critic_clip=critic_clip,
                    actor_clip=actor_clip,
                )

            # move target model towards training model
            if not self_regularized and (step % target_delay == 0):
                utils.soft_update(target_agent.critic1, agent.critic1, tau)
                utils.soft_update(target_agent.critic2, agent.critic2, tau)

        if (step % eval_interval == 0) or (step == num_steps - 1):
            mean_return = run.evaluate_agent(
                agent, test_env, eval_episodes, max_episode_steps, render
            )
            if log_to_disk:
                writer.add_scalar("return", mean_return, step * transitions_per_step)

        if step % save_interval == 0 and save_to_disk:
            agent.save(save_dir)

    if save_to_disk:
        agent.save(save_dir)
    return agent


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
    per = isinstance(buffer, replay.PrioritizedReplayBuffer)
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


def learn_self_regularized(
    buffer,
    agent,
    actor_optimizer,
    critic_optimizer,
    log_alpha_optimizer,
    target_entropy,
    critic_target_improvement,
    max_critic_updates_per_step,
    batch_size,
    log_alpha,
    gamma,
    critic_clip,
    actor_clip,
    update_policy=True,
):
    per = isinstance(buffer, replay.PrioritizedReplayBuffer)
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
        y1 = agent.critic1(next_state_batch, action_s1)
        y2 = agent.critic2(next_state_batch, action_s1)
        clipped_double_q_s1 = torch.min(y1, y2)
        td_target = reward_batch + gamma * (1.0 - done_batch) * (
            clipped_double_q_s1 - (alpha * logp_a1)
        )

    critic_loss_initial = None
    for critic_update in range(max_critic_updates_per_step):
        # standard bellman error
        a_critic1_pred = agent.critic1(state_batch, action_batch)
        a_critic2_pred = agent.critic2(state_batch, action_batch)
        td_error1 = td_target - a_critic1_pred
        td_error2 = td_target - a_critic2_pred

        # constraints that discourage large changes in Q(s_{t+1}, a_{t+1}),
        a1_critic1_pred = agent.critic1(next_state_batch, action_s1)
        a1_critic2_pred = agent.critic2(next_state_batch, action_s1)
        a1_constraint1 = y1 - a1_critic1_pred
        a1_constraint2 = y2 - a1_critic2_pred

        elementwise_critic_loss = (
            (td_error1 ** 2)
            + (td_error2 ** 2)
            + (a1_constraint1 ** 2)
            + (a1_constraint2 ** 2)
        )
        if per:
            elementwise_loss *= imp_weights
        critic_loss = 0.5 * elementwise_critic_loss.mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        if critic_clip:
            torch.nn.utils.clip_grad_norm_(
                chain(agent.critic1.parameters(), agent.critic2.parameters()),
                critic_clip,
            )
        critic_optimizer.step()
        if critic_update == 0:
            critic_loss_initial = critic_loss
        elif critic_loss <= critic_target_improvement * critic_loss_initial:
            break

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


def learn_discrete(
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
    per = isinstance(buffer, replay.PrioritizedReplayBuffer)
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
        # create critic targets (clipped double Q learning)
        action_dist_s1 = agent.actor(next_state_batch)
        target_value_s1 = (
            action_dist_s1.probs
            * (
                torch.min(
                    target_agent.critic1(next_state_batch),
                    target_agent.critic2(next_state_batch),
                )
                - (alpha.detach() * action_dist_s1.entropy()).unsqueeze(1)
            )
        ).sum(1, keepdim=True)
        td_target = reward_batch + gamma * (1.0 - done_batch) * (target_value_s1)

    # update critics
    agent_critic1_pred = agent.critic1(state_batch).gather(1, action_batch.long())
    agent_critic2_pred = agent.critic2(state_batch).gather(1, action_batch.long())
    td_error1 = td_target - agent_critic1_pred
    td_error2 = td_target - agent_critic2_pred
    critic_loss = (td_error1 ** 2) + (td_error2 ** 2)
    if per:
        critic_loss *= imp_weights
    critic_loss = 0.5 * critic_loss.mean()
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
        a_dist = agent.actor(state_batch)
        prob_a = a_dist.probs
        vals = torch.min(agent.critic1(state_batch), agent.critic2(state_batch))
        actor_loss = -(
            (prob_a * vals).sum(1) - (alpha.detach() * a_dist.entropy())
        ).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        if actor_clip:
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), actor_clip)
        actor_optimizer.step()

        ##################
        ## ALPHA UPDATE ##
        ##################
        alpha_loss = (alpha * (a_dist.entropy() + target_entropy).detach()).mean()
        log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        log_alpha_optimizer.step()

    if per:
        new_priorities = (abs(td_error1) + 1e-5).cpu().detach().squeeze(1).numpy()
        buffer.update_priorities(priority_idxs, new_priorities)


def add_args(parser):
    parser.add_argument(
        "--num_steps", type=int, default=10 ** 6, help="Number of steps in training"
    )
    parser.add_argument(
        "--transitions_per_step",
        type=int,
        default=1,
        help="env transitions per training step. Defaults to 1, but will need to \
        be set higher for repaly ratios < 1",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=100000,
        help="maximum steps per episode",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="training batch size"
    )
    parser.add_argument(
        "--tau", type=float, default=0.005, help="for model parameter % update"
    )
    parser.add_argument(
        "--actor_lr", type=float, default=1e-4, help="actor learning rate"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=1e-4, help="critic learning rate"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="gamma, the discount factor"
    )
    parser.add_argument(
        "--init_alpha",
        type=float,
        default=0.1,
        help="initial entropy regularization coefficeint.",
    )
    parser.add_argument(
        "--alpha_lr",
        type=float,
        default=1e-4,
        help="alpha (entropy regularization coefficeint) learning rate",
    )
    parser.add_argument(
        "--buffer_size", type=int, default=1_000_000, help="replay buffer size"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=5000,
        help="how often to test the agent without exploration (in episodes)",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="how many episodes to run for when testing",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="warmup length, in steps"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="flag to enable env rendering during training",
    )
    parser.add_argument(
        "--actor_clip",
        type=float,
        default=None,
        help="gradient clipping for actor updates",
    )
    parser.add_argument(
        "--critic_clip",
        type=float,
        default=None,
        help="gradient clipping for critic updates",
    )
    parser.add_argument(
        "--name", type=str, default="sac_run", help="dir name for saves"
    )
    parser.add_argument(
        "--actor_l2",
        type=float,
        default=0.0,
        help="L2 regularization coeff for actor network",
    )
    parser.add_argument(
        "--critic_l2",
        type=float,
        default=0.0,
        help="L2 regularization coeff for critic network",
    )
    parser.add_argument(
        "--target_delay",
        type=int,
        default=2,
        help="How many steps to go between target network updates",
    )
    parser.add_argument(
        "--actor_delay",
        type=int,
        default=1,
        help="How many steps to go between actor updates",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100_000,
        help="How many steps to go between saving the agent params to disk",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="verbosity > 0 displays a progress bar during training",
    )
    parser.add_argument(
        "--gradient_updates_per_step",
        type=int,
        default=1,
        help="how many gradient updates to make per env step",
    )
    parser.add_argument(
        "--prioritized_replay",
        action="store_true",
        help="flag that enables use of prioritized experience replay",
    )
    parser.add_argument(
        "--skip_save_to_disk",
        action="store_true",
        help="flag to skip saving agent params to disk during training",
    )
    parser.add_argument(
        "--skip_log_to_disk",
        action="store_true",
        help="flag to skip saving agent performance logs to disk during training",
    )
    parser.add_argument(
        "--discrete_actions",
        action="store_true",
        help="enable SAC Discrete update function for discrete action spaces",
    )
    parser.add_argument(
        "--log_std_low",
        type=float,
        default=-10.0,
        help="Lower bound for log std of action distribution.",
    )
    parser.add_argument(
        "--log_std_high",
        type=float,
        default=2.0,
        help="Upper bound for log std of action distribution.",
    )
    parser.add_argument(
        "--self_regularized",
        action="store_true",
        help="Self Regularization (no target networks!), as in GRAC",
    )
    parser.add_argument(
        "--sr_max_critic_updates_per_step",
        type=int,
        default=10,
        help="Max critic updates to make per step. The GRAC paper calls this K",
    )
    parser.add_argument(
        "--sr_critic_target_improvement_init",
        type=float,
        default=0.7,
        help="Stop critic updates when loss drops by this factor. The GRAC paper calls this alpha",
    )
    parser.add_argument(
        "--sr_critic_target_improvement_final",
        type=float,
        default=0.9,
        help="Stop critic updates when loss drops by this factor. The GRAC paper calls this alpha",
    )
