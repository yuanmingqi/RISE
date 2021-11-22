from src.rise import train as rise_train
from src.re3 import train as re3_train
from src.vse import train as vse_train
from src.ppo import train as ppo_train
from src.args import get_args

if __name__ == '__main__':
    args = get_args()

    args.use_gae = True
    args.lr = 2.5e-4
    args.clip_param = 0.1
    args.value_loss_coef = 0.5
    args.num_processes = 8
    args.num_steps = 128
    args.num_mini_batch = 4
    args.log_interval = 1
    args.use_linear_lr_decay = True
    args.entropy_coef = 0.01
    args.log_dir = './logs/{}/{}/{}'.format(args.action_space, args.algo, args.env_name)
    args.seed = 0
    args.save_dir = './models/{}/{}/{}'.format(args.action_space, args.algo, args.env_name)

    args.num_env_steps = 10000000

    if args.algo == 'PPO':
        ppo_train(args)
    elif args.algo == 'RE3':
        args.beta0 = 0.1
        args.kappa = 1e-5
        re3_train(args)
    elif args.algo == 'VSE':
        args.beta = 0.1
        vse_train(args)
    elif args.algo == 'RISE':
        args.max_sample_steps = 1e+5
        args.beta0 = 0.1
        args.kappa = 1e-5
        rise_train(args)
    else:
        raise NotImplementedError
