# DRL-Project-Continuous-Control

## Project Details
This project is based on a Unity environment to design, train, and evaluate deep reinforcement learning algorithms.
The environment used in this project is the Reacher environment.

<p align="center">
 <img src="reacher.gif"/>
    <br>
    <em><b>Unity ML-Agents Reacher Environment</b></em>
</p>

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this project, two separate versions of the Unity environment are provided:

* The first version contains a single agent.
* The second version contains 20 identical agents, each with its own copy of the environment.

The second version is useful to distribute the task of gathering experience.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes in the first version. The second version is solved, when the average over 100 consecutive episodes of the average scores over all 20 agents (obtained with each episode) is at least +30.

## Getting Started

### Project File Structure
The project is structured as follows:

📦project<br>
 ┣ 📂Reacher_Linux_1  **`(contains the one-agent Reacher environment for Linux based systems)`** <br>
 ┣ 📂Reacher_Linux_20  **`(contains the twenty-agent Reacher environment for Linux based systems)`** <br>
 ┣ 📂Reacher_Windows_x86_64_1  **`(contains the one-agent Reacher environment for Windows 64-bit based systems)`** <br>
 ┣ 📂Reacher_Windows_x86_64_20  **`(contains the twenty-agent environment for Windows 64-bit based systems)`** <br>
 ┣ 📂models  **`(contains the actor and critic states of successfully trained agents)`** <br>
 ┃ ┣ checkpoint_actor.pth<br>
 ┃ ┣ checkpoint_critic.pth<br>
 ┃ ┗ ... <br>
 ┣ 📂python **`(files required to set up the environment)`** <br>
 ┣ 📂score_plots **`(contains the score plots of successfully trained agents)`** <br>
 ┃ ┣ score_plot_1.png<br>
 ┃ ┣ training_output_1.png<br>
 ┃ ┗ ...<br>
 ┣ .gitignore <br>
 ┣ config.py  <br>
 ┣ config.yml <br>
 ┣ ddpg_agent.py **`(Unity agent for reacher environment)`**<br> 
 ┣ main.py **`(Python script to run a trained agent or to train a new one)`**<br>
 ┣ model.py **`(Actor and Critic networks)`**<br>
 ┣ reacher.gif <br>
 ┣ README.md <br>
 ┗ Report.md <br>
 
### Installation and Dependencies

The code of this project was tested on Linux (Ubuntu 20.04) and Windows 11. To get the code running on your local system, follow these steps which are base on Anaconda and pip:

1.  `conda create --name reacher python=3.8 -c conda-forge`
2.  `conda activate reacher`
3.  Create a directory where you want to save this project
4.  `git clone https://github.com/rp-dippold/DRL-Project-Continuous-Control.git`
5.  `cd python`
6.  `pip install .`
7.  `python -m ipykernel install --user --name reacher --display-name "reacher"`
8.  Install Pytorch:
    * [CPU]: `pip install torch torchvision torchaudio`
    * [GPU]: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`.\
    Depending on your GPU and cudnn version a different pytorch version my be required. Please refer to 
    https://pytorch.org/get-started/locally/.


## Instructions
To run the code go into the directory where you installed the repository. First of all, open the file `config.yml` and check if `reacher_env` refers to the correct reacher environment. Set the reacher environment as follows:

* **`Windows 11`**: "./Reacher_Windows_x86_64_[n]/Reacher.exe" (set [n] to 1 or 20 = number of agents)
* **`Linux`**: "./Reacher_Linux_[n]/Reacher.x86_64" (set [n] to 1 or 20 = number of agents)

**Note that the first start of the Unity environment &mdash; as described below  &mdash; may take up to 30 seconds; all following
start times are much shorter. So please be patient!**

#### Training an Agent
Before training an agent you should adapt the respective hyperparameters in config.yml. The current values allow to train an agent that can reach an average score of +30.0 over 100 consecutive episodes.

To start training just enter the following command: `python main.py train`

If you want to watch the agent during training enter: `python main.py train --watch`

At the end of the training a window pops up that shows the scores of the agent for each episode. After closing this 
window, the program stops.

If the agent was trained successfully its weights are saved in the root directory and not in directory `models`. \
The filenames are: `checkpoint_actor.pth` and `checkpoint_critic.pth`.

#### Running the Environment with a Smart Agent
Depending on whether you want see the actions of a trained model in a one-agent or twenty-agent environment set the 
reacher_env variable in config.yml accordingly (see above). The following command runs the environment:

`python main.py run --actor_params <path to stored actor weights> --critic_params <path to the stored critic weights>`

`<path to stored xxx weights>` is the path to the directory plus the name of the `checkpoint_xxx.pth` file, e.g.
`./models/checkpoint_actor.pth`.
