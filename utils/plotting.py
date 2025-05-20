"""
Plotting utilities for simulation, experimental, and neural network output data.
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_demo_data():
    # Experimental data
    x1_exp = np.linspace(0, 2, 6)
    y1_exp = x1_exp
    x2_exp = np.linspace(0, 2, 6)
    y2_exp = x2_exp + 2
    x3_exp = np.linspace(0, 2, 6)
    y3_exp = x3_exp + 4

    # Simulation data
    x1 = np.linspace(0, 2, int(1e3))
    y1 = x1
    x2 = np.linspace(0, 2, int(1e3))
    y2 = x2 + 2
    x3 = np.linspace(0, 2, int(1e3))
    y3 = x3 + 4

    # Neural network data
    x1n = np.linspace(0, 2, 10)
    y1n = x1n
    x2n = np.linspace(0, 2, 10)
    y2n = x2n + 2
    x3n = np.linspace(0, 2, 10)
    y3n = x3n + 4

    return {
        "exp": [(x1_exp, y1_exp), (x2_exp, y2_exp), (x3_exp, y3_exp)],
        "sim": [(x1, y1), (x2, y2), (x3, y3)],
        "nn": [(x1n, y1n), (x2n, y2n), (x3n, y3n)]
    }

def plot_figure1(data, save=True):
    blue = (0.00, 0.00, 0.55)
    red = (0.65, 0.16, 0.16)
    green = (0.00, 0.39, 0.00)
    colors = [blue, red, green]
    markers = ['x', 'x', 'x']
    exp_markers = ['^', 's', 'v']

    fig, ax = plt.subplots()
    # Simulation
    for i, (x, y) in enumerate(data["sim"]):
        ax.plot(x, y, color=colors[i])
    # Neural-network predictions
    for i, (x, y) in enumerate(data["nn"]):
        ax.scatter(x, y, marker=markers[i], edgecolor='k', facecolor=colors[i], s=60)
    # Experiments
    for i, (x, y) in enumerate(data["exp"]):
        ax.scatter(x, y, marker=exp_markers[i], edgecolor='k', facecolor=colors[i], s=60)

    ax.set_xlabel(r'$\lambda$/nm', fontsize=14, fontname='Times New Roman')
    ax.set_ylabel(r'$\Delta p$/Pa', fontsize=14, fontname='Times New Roman')
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 6])
    ax.tick_params(labelsize=14)
    ax.set_box_aspect(1)
    ax.grid(False)
    fig.tight_layout()
    if save:
        fig.savefig('results/figure1.eps', format='eps')
        fig.savefig('results/figure1.jpg', format='jpg')
        fig.savefig('results/figure1.tif', format='tiff')
    return fig, ax

def plot_figure2(data, save=True):
    blue = (0.00, 0.00, 0.55)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # (a)
    axs[0, 0].plot(data["sim"][0][0], data["sim"][0][1], color=blue)
    axs[0, 0].set_title('(a)', fontsize=14, fontname='Times New Roman', loc='left')
    axs[0, 0].set_xlabel(r'$\lambda$/nm', fontsize=14, fontname='Times New Roman')
    axs[0, 0].set_ylabel(r'$\Delta p$/Pa', fontsize=14, fontname='Times New Roman')
    axs[0, 0].set_xlim([0, 2])
    axs[0, 0].set_ylim([0, 6])
    axs[0, 0].tick_params(labelsize=14)
    axs[0, 0].set_box_aspect(1)
    axs[0, 0].grid(False)
    # (b)
    axs[0, 1].plot(data["sim"][0][0], data["sim"][1][1], color=blue)
    axs[0, 1].set_title('(b)', fontsize=14, fontname='Times New Roman', loc='left')
    axs[0, 1].set_xlabel(r'$\lambda$/nm', fontsize=14, fontname='Times New Roman')
    axs[0, 1].set_ylabel(r'$\Delta p$/Pa', fontsize=14, fontname='Times New Roman')
    axs[0, 1].set_xlim([0, 2])
    axs[0, 1].set_ylim([0, 6])
    axs[0, 1].tick_params(labelsize=14)
    axs[0, 1].set_box_aspect(1)
    axs[0, 1].grid(False)
    # (c)
    axs[1, 0].plot(data["sim"][0][0], data["sim"][2][1], color=blue)
    axs[1, 0].set_title('(c)', fontsize=14, fontname='Times New Roman', loc='left')
    axs[1, 0].set_xlabel(r'$\lambda$/nm', fontsize=14, fontname='Times New Roman')
    axs[1, 0].set_ylabel(r'$\Delta p$/Pa', fontsize=14, fontname='Times New Roman')
    axs[1, 0].set_xlim([0, 2])
    axs[1, 0].set_ylim([0, 6])
    axs[1, 0].tick_params(labelsize=14)
    axs[1, 0].set_box_aspect(1)
    axs[1, 0].grid(False)
    # Hide the unused subplot
    axs[1, 1].axis('off')
    fig.tight_layout()
    if save:
        fig.savefig('results/figure2.eps', format='eps')
        fig.savefig('results/figure2.jpg', format='jpg')
        fig.savefig('results/figure2.tif', format='tiff')
    return fig, axs

if __name__ == "__main__":
    data = generate_demo_data()
    plot_figure1(data)
    plot_figure2(data)
    plt.show()