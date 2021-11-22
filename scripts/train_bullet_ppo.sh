for env_name in Ant HalfCheetah Hopper Humanoid InvertedPendulum Walker2D
do
	python main.py --action-space cont --env-name ${env_name}BulletEnv-v0 --algo PPO
done
