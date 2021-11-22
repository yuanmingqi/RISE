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

class RE3:
    def __init__(self,
                 ob_shape,
                 action_shape,
                 device
                 ):
        self.ob_shape = ob_shape
        self.action_shape = action_shape
        self.device = device

        if len(ob_shape) == 3:
            self.embedding_network = CNNEmbeddingNetwork(
                kwargs={'in_channels': ob_shape[0], 'embedding_size': 128})
        else:
            self.embedding_network = MLPEmbeddingNetwork(
                kwargs={'input_dim': ob_shape[0], 'embedding_size': 64}
            )

        self.embedding_network.to(self.device)

        for p in self.embedding_network.parameters():
            p.requires_grad = False

    def compute_intrinsic_rewards(self, obs_buffer, k=5):
        size = obs_buffer.size()
        obs = obs_buffer[:size[0] - 1]
        intrinsic_rewards = torch.zeros(size=(size[0] - 1, size[1], 1))

        for process in range(size[1]):
            encoded_obs = self.embedding_network(obs[:, process].to(self.device))
            for step in range(size[0] - 1):
                dist = torch.norm(encoded_obs[step] - encoded_obs, p=2, dim=1)
                H_step = torch.log(dist.sort().values[k + 1] + 1.)
                intrinsic_rewards[step, process, 0] = H_step

        return intrinsic_rewards


# re3 = RE3(ob_shape=[4, 84, 84], device=None)
# obs_buffer = torch.rand(size=[129, 8, 4, 84, 84])
# rewards = re3.compute_rewards(obs_buffer)

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

    if envs.action_space.__class__.__name__ == "Discrete":
        re3 = RE3(
            envs.observation_space.shape,
            envs.action_space.n,
            device
        )
    elif envs.action_space.__class__.__name__ == 'Box':
        re3 = RE3(
            envs.observation_space.shape,
            envs.action_space.shape,
            device
        )
    else:
        raise NotImplementedError

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

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        ''' compute intrinsic rewards '''
        intrinsic_rewards = re3.compute_intrinsic_rewards(rollouts.obs)
        beta_t = args.beta0 * np.power(1. - args.kappa, (j + 1) * args.num_steps)
        rollouts.rewards += beta_t * intrinsic_rewards.to(device)

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

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
