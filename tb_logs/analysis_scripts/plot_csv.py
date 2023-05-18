
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_csv_logs(reward, length, exploration, loss):

    step_rew = [row[0] for row in reward]
    value_rew = [row[2] for row in reward]

    step_len = [row[0] for row in length]
    value_len = [row[2] for row in length]

    step_exr = [row[0] for row in exploration]
    value_exr = [row[2] for row in exploration]

    step_loss = [row[0] for row in loss]
    value_loss = [row[2] for row in loss]

    LineWidth = 2
    
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Additional LaTeX packages if needed

    plt.subplot(2, 2, 1)
    plt.plot(step_rew, value_rew, linewidth=LineWidth)
    plt.ylabel(r'\text{Episode length}')
    plt.xlabel(r'\text{Step}')
    plt.title(r'\text{Mean episode reward}')
    plt.grid(True, which='both', linestyle='--')
    plt.minorticks_on()

    plt.subplot(2, 2, 2)
    plt.plot(step_len, value_len, linewidth=LineWidth)
    plt.ylabel(r'\text{Episode length}')
    plt.xlabel(r'\text{Step}')
    plt.title(r'\text{Mean episode length}')
    plt.grid(True, which='both', linestyle='--')
    plt.minorticks_on()

    plt.subplot(2, 2, 3)
    plt.plot(step_exr, value_exr, linewidth=LineWidth)
    plt.ylabel(r'\text{Episode length}')
    plt.xlabel(r'\text{Step}')
    plt.title(r'\text{Exploration Rate}')
    plt.grid(True, which='both', linestyle='--')
    plt.minorticks_on()

    plt.subplot(2, 2, 4)
    plt.plot(step_loss, value_loss, linewidth=LineWidth)
    plt.ylabel(r'\text{Episode length}')
    plt.xlabel(r'\text{Step}')
    plt.title(r'\text{Train Loss}')
    plt.grid(True, which='both', linestyle='--')
    plt.minorticks_on()

    plt.suptitle(r'\text{Torque-based energy consumption without continuation after lost detection}')
    plt.tight_layout()
    plt.show()