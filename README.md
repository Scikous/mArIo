# mArIo
A neural network utilizing reinforcement learning (Deep Q Learning in this case) for playing Super Marios Bros. (NES)

## Table of Contents
- [Dependencies](#dependencies)
- [Features](#features)
- [Usage](#usage)
- [Credits](#credits)

# Dependencies
> :warning: Gym versions later than 0.23.1 are incompatible with gym-super-mario_bros 7.4.0
```
pip install -r requirements.txt
```

# Features
```
- Mathematically correct implementation of Deep Q Learning
- Custom Convolutional Neural Network (CNN)
- Easy training and testing
```

# Usage
Simply run the env.py for training a model or env_eval.py for evaluating (testing) a model

# Credits
Some logic and code used from the following:
```
Sourish07 (Sourish Kundu): https://github.com/Sourish07/Super-Mario-Bros-RL
jereminuer (Jeremi Nuer): https://github.com/jereminuer/DQN_Cart-Pole

@misc{gym-super-mario-bros,
  author = {Christian Kauten},
  howpublished = {GitHub},
  title = {{S}uper {M}ario {B}ros for {O}pen{AI} {G}ym},
  URL = {https://github.com/Kautenja/gym-super-mario-bros},
  year = {2018},
}
```