
import matplotlib
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import argparse

def plot_combined_graph(args, ax, show_ylabel=True):

    input_directory = f'{args.root_directory}'
    print('Input Directory', input_directory)

    if os.path.exists(input_directory):
        print("The directory exists.")
    else:
        print("The directory does not exist.")

    # ax.figure(figsize=(10, 5))
    max_, min_ = 0, 100
    max_spar, min_spar = 0, 100

    for root, dirs, files in os.walk(input_directory):
        #for dir in dirs:
            for filename in files:
                if filename.endswith('log'):

                    file_path = os.path.join(root, filename)
                    final_test_acc_list = []
                    adj_list = []

                    with open(file_path, 'r') as infile:
                        for line in infile:
                            final_test_acc_match = re.search(r'Test Acc:([\d.]+)', line)
                            spa = re.search(r'spa:\[([\d.]+)%\]', line)
                            if final_test_acc_match and spa:
                                final_test_acc = float(final_test_acc_match.group(1))
                                adj = float(spa.group(1))
                                final_test_acc_list.append(final_test_acc)
                                adj_list.append(adj)

                    # print(final_test_acc_list)
                    
                    graph_sparcity = adj_list
                    
                    print(graph_sparcity)
                    
                    max_ = max(max_, max(final_test_acc_list))
                    min_ = min(min_, min(final_test_acc_list))
                    min_spar = min(min_spar, min(graph_sparcity))
                    max_spar = max(max_spar, max(graph_sparcity))
                    if 'new_url' in file_path:
                        ax.plot(graph_sparcity, final_test_acc_list, linestyle='-', linewidth=3, label=f'{filename}_new')
                    else:
                        ax.plot(graph_sparcity, final_test_acc_list, linestyle='-', linewidth=3, label=f'{filename}_old')
            ax.set_xlabel('Graph Sparsity (%)', fontsize=22)

    # Set x-axis ticks and labels
    ax.xaxis.set_ticks(np.arange(0, max_spar + 3, 5))
    ax.xaxis.set_tick_params(rotation=45, labelsize=20)

    ax.tick_params(axis='x', rotation=45, labelsize=20)
    margin = 1
    ax.set_xlim(min_spar - margin, max_spar + margin + 0.5)

    min_y = (min_-15) - ((min_ - 15) % 2)
    max_y = min_y + np.ceil((max_ - min_y) / 5) * 5
    if max_y > max_:
        diff = max_y - max_
        if diff >= 3:
            max_y -= 3
            min_y += 2
    ax.set_ylim(min_y, max_y)
    ax.set_yticks(np.linspace(min_y, max_y, 6))
    ax.tick_params(axis='y', labelsize=20)

    ax.grid(axis='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=20, loc='lower left')  # Reduce fontsize from 20 to 16
    

def plot_single_figure(args):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Print the input arguments
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Root Directory: {args.root_directory}")
    print(f"Output Directory: {args.output_directory}")

    plot_combined_graph(args, ax)

    ax.set_ylabel(f'Accuracy ({args.dataset})', fontsize=22)

    ax_right = ax.twinx()
    ax_right.set_yticks([])
    plt.show()

    output_directory = f'{args.output_directory}/'
    os.makedirs(output_directory, exist_ok=True)
    output_pdf = os.path.join(output_directory, f'Inductive_{args.dataset}.pdf')
    output_jpg = os.path.join(output_directory, f'Inductive_{args.dataset}.jpg')

    fig.savefig(output_pdf, bbox_inches='tight')
    fig.savefig(output_jpg, bbox_inches='tight', dpi=300)

def initialize_args():
    # Define the input arguments
    class Args:
        pass

    args = Args()
    args.model = "Inductive"
    args.dataset = "DD"
    args.root_directory = "/home/yifan/conda/Inductive_Lottery_Ticket_Learning_Supplementary_Material-2/TUDataset/log"
    args.output_directory = "/home/yifan/conda/Inductive_Lottery_Ticket_Learning_Supplementary_Material-2/TUDataset/log"

    return args

# main to plot single figure
if __name__ == "__main__":
    args = initialize_args()

    # Plot the first subfigure
    plot_single_figure(args)

