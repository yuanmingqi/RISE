import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data

import os
import time
from collections import deque

import numpy as np
import torch

from a2c_ppo_acktr import algo
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.model import Policy

class CNNEmbeddingNetwork(nn.Module):
    def __init__(self, kwargs):
        super(CNNEmbeddingNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(kwargs['in_channels'], 32, (8, 8), stride=(4, 4)), nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=(2, 2)), nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), stride=(1, 1)), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 7 * 7, kwargs['embedding_size']))

    def forward(self, ob):
        x = self.main(ob)

        return x

class MLPEmbeddingNetwork(nn.Module):
    def __init__(self, kwargs):
        super(MLPEmbeddingNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(kwargs['input_dim'], 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, kwargs['embedding_size'])
        )

    def forward(self, ob):
        x = self.main(ob)

        return x

class DisInverseDynamicModel(nn.Module):
    def __init__(self, kwargs):
        super(DisInverseDynamicModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(kwargs['embedding_size'] * 2, 32), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, kwargs['action_shape'])
        )

    def forward(self, ob_emb, next_ob_emb):
        probs = self.main(torch.cat([ob_emb, next_ob_emb], dim=1))
        pred_action = F.softmax(probs, dim=1)

        return pred_action

class DisForwardDynamicModel(nn.Module):
    def __init__(self, kwargs):
        super(DisForwardDynamicModel, self).__init__()
        self.nA = kwargs['nA']
        self.main = nn.Sequential(
            nn.Linear(kwargs['embedding_size'] + kwargs['action_shape'], 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, kwargs['embedding_size'])
        )

    def forward(self, ob_emb, true_action):
        onehot_action = F.one_hot(true_action, num_classes=self.nA)
        pred_next_ob_emb = self.main(torch.cat([ob_emb, onehot_action], dim=1))

        return pred_next_ob_emb

class ConInverseDynamicModel(nn.Module):
    def __init__(self, kwargs):
        super(ConInverseDynamicModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(kwargs['embedding_size'] * 2, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, kwargs['action_shape'])
        )

    def forward(self, ob_emb, next_ob_emb):
        pred_action = self.main(torch.cat([ob_emb, next_ob_emb], dim=1))

        return pred_action

class ConForwardDynamicModel(nn.Module):
    def __init__(self, kwargs):
        super(ConForwardDynamicModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(kwargs['embedding_size'] + kwargs['action_shape'], 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, kwargs['embedding_size'])
        )

    def forward(self, ob_emb, true_action):
        pred_next_ob_emb = self.main(torch.cat([ob_emb, true_action], dim=1))

        return pred_next_ob_emb

class RIDE:
    def __init__(
            self,
            ob_shape,
            action_shape,
            device
    ):
        self.ob_shape = ob_shape
        self.action_shape = action_shape
        self.device = device

        if len(ob_shape) == 3:
            self.embedding_network = CNNEmbeddingNetwork(kwargs={'in_channels': ob_shape[0], 'embedding_size': 128})
            self.idm = DisInverseDynamicModel(kwargs={'embedding_size': 128, 'action_shape':action_shape})
            self.fdm = DisForwardDynamicModel(kwargs={'embedding_size': 128, 'action_shape':action_shape})
        else:
            self.embedding_network = MLPEmbeddingNetwork(kwargs={'input_dim': ob_shape[0], 'embedding_size': 64})
            self.idm = ConInverseDynamicModel(kwargs={'embedding_size': 64, 'action_shape': action_shape[0]})
            self.fdm = ConForwardDynamicModel(kwargs={'embedding_size': 64, 'action_shape': action_shape[0]})

        self.embedding_network.to(self.device)
        self.idm.to(self.device)
        self.fdm.to(self.device)

        self.optimzer_en = optim.Adam(self.embedding_network.parameters(), lr=5e-4)
        self.optimzer_idm = optim.Adam(self.idm.parameters(), lr=5e-4)
        self.optimzer_fdm = optim.Adam(self.fdm.parameters(), lr=5e-4)

    def compute_rewards(self, obs_buffer):
        size = obs_buffer.size()

        obs = obs_buffer[:-1].view(-1, *obs_buffer.size()[2:])
        next_obs = obs_buffer[1:].view(-1, *obs_buffer.size()[2:])

        obs_emb = self.embedding_network(obs.to(self.device))
        next_obs_emb = self.embedding_network(next_obs.to(self.device))

        rewards = torch.norm(obs_emb - next_obs_emb, p=2, dim=1)

        return rewards.view(size[0] - 1, size[1], 1)

    def pseudo_counts(
            self,
            step,
            episodic_memory,
            current_c_ob,
            k=10,
            kernel_cluster_distance=0.008,
            kernel_epsilon=0.0001,
            c=0.001,
            sm=8,
    ):
        counts = torch.zeros(size=(episodic_memory.size()[1], 1))
        for process in range(episodic_memory.size()[1]):
            process_episodic_memory = episodic_memory[:step + 1, process, :]
            ob_dist = [(c_ob, torch.dist(c_ob, current_c_ob)) for c_ob in process_episodic_memory]
            # ob_dist = [(c_ob, torch.dist(c_ob, current_c_ob)) for c_ob in episodic_memory]
            ob_dist.sort(key=lambda x: x[1])
            ob_dist = ob_dist[:k]
            dist = [d[1].item() for d in ob_dist]
            dist = np.array(dist)

            # TODO: moving average
            dist = dist / np.mean(dist)
            dist = np.max(dist - kernel_cluster_distance, 0)
            kernel = kernel_epsilon / (dist + kernel_epsilon)
            s = np.sqrt(np.sum(kernel)) + c

            if np.isnan(s) or s > sm:
                counts[process] = 0.
            else:
                counts[process] = 1 / s

        return counts

    def update(self, obs_buffer, actions_buffer):
        obs = obs_buffer[:-1].view(-1, *obs_buffer.size()[2:])
        next_obs = obs_buffer[1:].view(-1, *obs_buffer.size()[2:])

        if len(self.ob_shape) == 3:
            actions = actions_buffer.view(-1, 1)
        else:
            actions = actions_buffer.view(-1, *actions_buffer.size()[2:])

        dataset = data.TensorDataset(obs, actions, next_obs)

        loader = data.DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=True,
            drop_last=True
        )

        for idx, batch_data in enumerate(loader):
            self.optimzer_en.zero_grad()
            self.optimzer_idm.zero_grad()
            self.optimzer_fdm.zero_grad()

            batch_obs, batch_actions, batch_next_obs = batch_data

            batch_obs_emb = self.embedding_network(batch_obs.to(self.device))
            batch_next_obs_emb = self.embedding_network(batch_next_obs.to(self.device))
            batch_actions = batch_actions.to(self.device)

            if len(self.ob_shape) == 3:
                pred_actions_logits = self.idm(batch_obs_emb, batch_next_obs_emb)
                inverse_loss = F.cross_entropy(pred_actions_logits, batch_actions.squeeze(1))
            else:
                pred_actions = self.idm(batch_obs_emb, batch_next_obs_emb)
                inverse_loss = F.mse_loss(pred_actions, batch_actions)
            pred_next_obs_emb = self.fdm(batch_obs_emb, batch_actions)
            forward_loss = F.mse_loss(pred_next_obs_emb, batch_next_obs_emb)

            total_loss = inverse_loss + forward_loss
            total_loss.backward()

            self.optimzer_en.step()
            self.optimzer_idm.step()
            self.optimzer_fdm.step()

# device = torch.device('cuda:0')
# ride = RIDE(ob_shape=[4, 84, 84], nA=7, device=device)
# obs_buffer = torch.rand(size=[129, 8, 4, 84, 84])
# actions_buffer = torch.randint(low=0, high=6, size=(128, 8, 1))
# ride.update(obs_buffer, actions_buffer)
# rewards = ride.compute_rewards(obs_buffer)
# print(rewards, rewards.size())

def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    # eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    # utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    ''' ride initialization '''
    if envs.observation_space.__class__.__name__ == 'Discrete':
        ride = RIDE(
            ob_shape=envs.observation_space.shape,
            action_shape=envs.action_space.n,
            device=device
        )
        episodic_emb_memory = torch.zeros(size=(args.num_steps + 1, args.num_processes, 128)).to(device)
    elif envs.observation_space.__class__.__name__ == 'Box':
        ride = RIDE(
            ob_shape=envs.observation_space.shape,
            action_shape=envs.action_space.shape,
            device=device
        )
        episodic_emb_memory = torch.zeros(size=(args.num_steps + 1, args.num_processes, 64)).to(device)
    else:
        raise NotImplementedError

    episode_rewards = deque(maxlen=10)
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        episodic_emb_memory[0, :, :] = ride.embedding_network(rollouts.obs[0].to(device))
        pseudo_counts = torch.zeros_like(rollouts.rewards).to(device)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_obs = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            ''' pseudo-count '''
            next_obs_emb = ride.embedding_network(obs.to(device))
            pseudo_counts[step, :, :] = ride.pseudo_counts(step, episodic_emb_memory, next_obs_emb)
            episodic_emb_memory[step + 1, :, :] = next_obs_emb

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_obs, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

            ''' compute intrinsic rewards '''
            intrinsic_rewards = ride.compute_rewards(rollouts.obs)
            rollouts.rewards += intrinsic_rewards * pseudo_counts

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()
        ''' update ride '''
        ride.update(rollouts.obs, rollouts.actions)

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
            or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                'ALGO {}, ENV {}, EPISODE {}, TIME STEPS {}, FPS {} \n MEAN/MEDIAN REWARD {:.3f}|{:.3f}, MIN|MAX REWARDS {:.3f}|{:.3f}\n'.format(
                    args.algo, args.env_name, j, total_num_steps, int(total_num_steps / (end - start)),
                    np.mean(episode_rewards), np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards)
                ))