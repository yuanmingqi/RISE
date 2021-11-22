for env_name in Assault BattleZone DemonAttack KungFuMaster Riverraid SpaceInvaders
do
	python main.py --action-space dis --env-name ${env_name}NoFrameskip-v4 --algo PPO
done
