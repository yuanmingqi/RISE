# Rényi State Entropy for Accelerating Exploration in Reinforcement Learning

*By Mingqi Yuan, Mon-on Pun and Dong Wang*

<div align='center'>
    <img src= 'https://github.com/yuanmingqi/RISE/blob/main/examples/rise.png'>
</div>

**RISE is a generalized state entropy maximization method for providing high-quality intrinsic rewards. 
If you find this repository is useful in your research, please cite the [[paper]](https://www.researchgate.net/publication/356407441):**

```
@article{yuan2022r,
  title={R$\backslash$'enyi State Entropy for Exploration Acceleration in Reinforcement Learning},
  author={Yuan, Mingqi and Pun, Man-on and Wang, Dong},
  journal={arXiv preprint arXiv:2203.04297},
  year={2022}
}
```

# Installation

* Get the repository with git:
```
git clone https://github.com/yuanmingqi/RISE.git
```

* Run the following command to get dependencies:

```
pip install -r requirements.txt
```

* Install the maze environment following:
```html
https://github.com/MattChanTK/gym-maze
```

# Training
Run the following command to train the model:
```shell
python main.py --action-space dis --env-name SpaceInvadersNoFrameskip-v4 --algo RISE 
```
Or use shell:
```shell
sh scripts/train_atari_rise.sh
```

