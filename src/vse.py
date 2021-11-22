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

from src.vae import VAE
from src.utils import preprocess

class VSE:
    def __init__(self,
                 ob_shape,
                 action_shape,
                 device
                 ):
        self.ob_shape = ob_shape
        self.action_shape = action_shape
        self.device = device
        self.vae = VAE(
            device=device,
            ob_shape=self.ob_shape,
            latent_dim=512
        )

    def compute_intrinsic_rewards(self, obs_buffer):
        size = obs_buffer.size()
        intrinsic_rewards = torch.zeros(size=(size[0] - 1, size[1], 1))
        with torch.no_grad():
            for process in range(size[1]):
                process_obs = torch.clone(obs_buffer[:, process]).to(self.device)
                _, mu, logvar, _ = self.vae.vae_backbone(process_obs, training=False)
                kld = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - torch.ones_like(logvar), dim=1, keepdim=True)
                intrinsic_rewards[:, process] = kld[:size[0] - 1]

        return intrinsic_rewards

    def update(self, obs_buffer):
        obs = obs_buffer[:-1].view(-1, *obs_buffer.size()[2:])

        dataset = data.TensorDataset(obs)
        loader = data.DataLoader(
            dataset=dataset,
            shuffle=True,
            batch_size=64,
            drop_last=True
        )

        self.vae.vae_backbone.train()
        eps_loss = 0.
        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]

            kld_loss, recon_loss, total_loss = self.vae.train_on_batch(batch_obs, training=True)
            eps_loss += total_loss

        return eps_loss

# device = torch.device('cuda:0')
# obs_buffer = torch.rand(size=[129, 8, 4, 84, 84])
# mmrs = RISE(ob_shape=[4 , 84, 84], device=device, n_clusters=10)
# mmrs.update(obs_buffer)
# rewards = mmrs.compute_rewards(obs_buffer=obs_buffer, lambda_gs=0.1, lambda_ls=0.1)
# print(rewards)

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
        vse = VSE(
            ob_shape=envs.observation_space.shape,
            action_shape=envs.action_space.n,
            device=device
        )
    elif envs.observation_space.__class__.__name__ == 'Box':
        vse = VSE(
            ob_shape=envs.observation_space.shape,
            action_shape=envs.action_space.shape,
            device=device
        )
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


        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_obs = actor_critic.act(
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
            rollouts.insert(obs, recurrent_hidden_obs, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        ''' compute intrinsic rewards '''
        intrinsic_rewards = vse.compute_intrinsic_rewards(rollouts.obs)
        rollouts.rewards += args.beta * intrinsic_rewards.to(device)

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()
        ''' update ride '''
        vse.update(rollouts.obs)

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