import pandas as pd
import matplotlib.pyplot as plt

Data_rew = pd.read_csv('run-.-tag-rollout_ep_rew_mean.csv', header=None)
Data_len = pd.read_csv('run-.-tag-rollout_ep_len_mean.csv', header=None)
Data_exr = pd.read_csv('run-.-tag-rollout_exploration_rate.csv', header=None)
Data_loss = pd.read_csv('run-.-tag-train_loss.csv', header=None)

step_rew = Data_rew.iloc[:, 1].values
value_rew = Data_rew.iloc[:, 2].values

step_len = Data_len.iloc[:, 1].values
value_len = Data_len.iloc[:, 2].values

step_exr = Data_exr.iloc[:, 1].values
value_exr = Data_exr.iloc[:, 2].values

step_loss = Data_loss.iloc[:, 1].values
value_loss = Data_loss.iloc[:, 2].values

LineWidth = 2

plt.subplot(2, 2, 1)
plt.plot(step_rew, value_rew, linewidth=LineWidth)
plt.ylabel('Episode length')
plt.xlabel('Step')
plt.title('Mean episode reward')
plt.grid(True, which='both', linestyle='--')
plt.minorticks_on()

plt.subplot(2, 2, 2)
plt.plot(step_len, value_len, linewidth=LineWidth)
plt.ylabel('Episode length')
plt.xlabel('Step')
plt.title('Mean episode length')
plt.grid(True, which='both', linestyle='--')
plt.minorticks_on()

plt.subplot(2, 2, 3)
plt.plot(step_exr, value_exr, linewidth=LineWidth)
plt.ylabel('Episode length')
plt.xlabel('Step')
plt.title('Exploration Rate')
plt.grid(True, which='both', linestyle='--')
plt.minorticks_on()

plt.subplot(2, 2, 4)
plt.plot(step_loss, value_loss, linewidth=LineWidth)
plt.ylabel('Episode length')
plt.xlabel('Step')
plt.title('Train Loss')
plt.grid(True, which='both', linestyle='--')
plt.minorticks_on()

plt.suptitle('Torque-based energy consumption without continuation after lost detection')
plt.tight_layout()
plt.show()
