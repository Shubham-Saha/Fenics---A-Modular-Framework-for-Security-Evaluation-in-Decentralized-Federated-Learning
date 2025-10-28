# plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import numpy as np
from dfl_simulator.utils import detect_convergence

def visualize_data_distribution(train_datasets, num_nodes, class_names, output_dir, logger):
    num_classes = len(class_names)
    class_counts = np.zeros((num_nodes, num_classes))  # Assuming 10 classes in FashionMNIST

    # Aggregate class counts for each worker
    for i, dataset in enumerate(train_datasets):
        # Extract targets for the current worker
        targets = np.array(dataset.dataset.targets)[dataset.indices]
        for cls in range(num_classes):
            class_counts[i, cls] = np.sum(targets == cls)

    # Set up the plot
    plt.figure(figsize=(20, 10))
    bar_width = 0.8 / num_nodes  # Adjust bar width based on number of workers
    indices = np.arange(num_classes)  # Number of classes

    # Generate a color palette for different workers
    palette = sns.color_palette("Dark2", num_nodes)  # Other options like: 'Set2' 

    # Plot bars for each worker
    for i in range(num_nodes):
        plt.bar(indices + i * bar_width, class_counts[i], width=bar_width, label=f'Worker {i}', color=palette[i])

    # Labeling and aesthetics
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    #plt.title('Data Distribution Across Workers', fontsize=16)
    plt.xticks(indices + bar_width * (num_nodes / 2), class_names, rotation=45, ha='right', fontsize=12)
    plt.legend(title='Workers', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    data_dist_path = os.path.join(output_dir, 'data_distribution.pdf')
    plt.savefig(data_dist_path, format='pdf')
    plt.close()
    logger.info(f"Data distribution plot saved as '{data_dist_path}'.")

def plot_metrics_with_convergence(metrics, rounds_range, total_execution_time, output_dir, logger):
    # Prepare data for training metrics
    avg_metrics_train = {
        'Accuracy': [],
        'F1 Score': [],
        'Precision': [],
        'Recall': []
    }
    # Prepare data for testing metrics
    avg_metrics_test = {
        'Accuracy': [],
        'F1 Score': [],
        'Precision': [],
        'Recall': []
    }
    
    for rnd in rounds_range:
        acc_train, f1_train, prec_train, rec_train = [], [], [], []
        acc_test, f1_test, prec_test, rec_test = [], [], [], []
        for node in metrics:
            if len(metrics[node]['train_accuracy']) >= rnd:
                acc_train.append(metrics[node]['train_accuracy'][rnd-1])
                f1_train.append(metrics[node]['train_f1_score'][rnd-1])
                prec_train.append(metrics[node]['train_precision'][rnd-1])
                rec_train.append(metrics[node]['train_recall'][rnd-1])
            if len(metrics[node]['accuracy']) >= rnd:
                acc_test.append(metrics[node]['accuracy'][rnd-1])
                f1_test.append(metrics[node]['f1_score'][rnd-1])
                prec_test.append(metrics[node]['precision'][rnd-1])
                rec_test.append(metrics[node]['recall'][rnd-1])
        # Compute average metrics over all nodes for this round
        avg_metrics_train['Accuracy'].append(np.nanmean(acc_train) if acc_train else 0)
        avg_metrics_train['F1 Score'].append(np.nanmean(f1_train) if f1_train else 0)
        avg_metrics_train['Precision'].append(np.nanmean(prec_train) if prec_train else 0)
        avg_metrics_train['Recall'].append(np.nanmean(rec_train) if rec_train else 0)
        
        avg_metrics_test['Accuracy'].append(np.nanmean(acc_test) if acc_test else 0)
        avg_metrics_test['F1 Score'].append(np.nanmean(f1_test) if f1_test else 0)
        avg_metrics_test['Precision'].append(np.nanmean(prec_test) if prec_test else 0)
        avg_metrics_test['Recall'].append(np.nanmean(rec_test) if rec_test else 0)

    # Detect convergence based on test accuracy
    convergence_round = detect_convergence(avg_metrics_test['Accuracy'])
    if convergence_round:
        logger.info(f"Convergence detected at round {convergence_round}")
    else:
        logger.info("Convergence not detected within the given rounds.")

    # Log the averaged training and testing metrics before plotting
    logger.info("\n=== Averaged Training Metrics per Round ===")
    for idx, rnd in enumerate(rounds_range):
        logger.info(f"Round {rnd}: Accuracy={avg_metrics_train['Accuracy'][idx]:.4f}, "
                    f"F1 Score={avg_metrics_train['F1 Score'][idx]:.4f}, "
                    f"Precision={avg_metrics_train['Precision'][idx]:.4f}, "
                    f"Recall={avg_metrics_train['Recall'][idx]:.4f}")
    
    logger.info("\n=== Averaged Testing Metrics per Round ===")
    for idx, rnd in enumerate(rounds_range):
        logger.info(f"Round {rnd}: Accuracy={avg_metrics_test['Accuracy'][idx]:.4f}, "
                    f"F1 Score={avg_metrics_test['F1 Score'][idx]:.4f}, "
                    f"Precision={avg_metrics_test['Precision'][idx]:.4f}, "
                    f"Recall={avg_metrics_test['Recall'][idx]:.4f}")
    
    # Plot Training Metrics Bar Chart
    x = np.arange(len(rounds_range))  # the label locations
    num_metrics = 4
    width = 0.2  # Adjusted for better visibility
    
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(7, 4))
    # Use a colorblind-friendly palette from seaborn
    palette = sns.color_palette("Set2", num_metrics)
    
    # Compute offsets for bar positions
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    
    plt.bar(x + offsets[0], avg_metrics_train['Accuracy'], width, label='Accuracy', color=palette[0])
    plt.bar(x + offsets[1], avg_metrics_train['F1 Score'], width, label='F1 Score', color=palette[1])
    plt.bar(x + offsets[2], avg_metrics_train['Precision'], width, label='Precision', color=palette[2])
    plt.bar(x + offsets[3], avg_metrics_train['Recall'], width, label='Recall', color=palette[3])
    
    plt.ylabel('Metric Value')
    plt.xlabel('Round')
    #plt.title('Average Training Metrics per Round')
    plt.xticks(x, [f'Round {rnd}' for rnd in rounds_range])
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    training_bar_path = os.path.join(output_dir, 'average_training_metrics_bar_chart.pdf')
    plt.savefig(training_bar_path, format='pdf')
    plt.close()
    logger.info(f"Average training metrics bar chart saved as '{training_bar_path}'.")

    # Plot Testing Metrics Bar Chart
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(7, 4))
    plt.bar(x + offsets[0], avg_metrics_test['Accuracy'], width, label='Accuracy', color=palette[0])
    plt.bar(x + offsets[1], avg_metrics_test['F1 Score'], width, label='F1 Score', color=palette[1])
    plt.bar(x + offsets[2], avg_metrics_test['Precision'], width, label='Precision', color=palette[2])
    plt.bar(x + offsets[3], avg_metrics_test['Recall'], width, label='Recall', color=palette[3])
    
    plt.ylabel('Metric Value')
    plt.xlabel('Round')
    #plt.title('Average Testing Metrics per Round')
    plt.xticks(x, [f'Round {rnd}' for rnd in rounds_range])
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    testing_bar_path = os.path.join(output_dir, 'average_testing_metrics_bar_chart.pdf')
    plt.savefig(testing_bar_path, format='pdf')
    plt.close()
    logger.info(f"Average testing metrics bar chart saved as '{testing_bar_path}'.")

    # Plot Training Metrics Line Plot with Total Execution Time
    plt.figure(figsize=(7, 4))
    rounds = list(rounds_range)
    # Use a colorblind-friendly palette from seaborn
    palette = sns.color_palette("Set2", 4)

    plt.plot(rounds, avg_metrics_train['Accuracy'], marker='o', label='Accuracy(%)', color=palette[0])
    plt.plot(rounds, avg_metrics_train['F1 Score'], marker='s', label='F1 Score(%)', color=palette[1])
    plt.plot(rounds, avg_metrics_train['Precision'], marker='^', label='Precision(%)', color=palette[2])
    plt.plot(rounds, avg_metrics_train['Recall'], marker='d', label='Recall(%)', color=palette[3])

    # Mark convergence point on training line plot
    '''if convergence_round and convergence_round in rounds_range:
        plt.axvline(x=convergence_round, color='red', linestyle='--', label='Convergence Point')'''

    # Annotate total execution time only on training line plot
    plt.text(0.95, 0.95, f'Total Execution Time: {total_execution_time:.2f} sec',
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.5))

    plt.ylabel('Metric Value')
    plt.xlabel('Round')
    #plt.title('Average Training Metrics per Round (Line Plot)')
    plt.xticks(rounds)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    training_line_path = os.path.join(output_dir, 'average_training_metrics_line_plot.pdf')
    plt.savefig(training_line_path, format='pdf')
    plt.close()
    logger.info(f"Average training metrics line plot saved as '{training_line_path}'.")

    # Plot Testing Metrics Line Plot
    plt.figure(figsize=(7, 4))
    x_offset = 0.1  # Small offset to separate overlapping lines
    '''plt.plot(rounds, avg_metrics_test['Accuracy'], marker='o', label='Accuracy', color=palette[0])
    plt.plot(rounds, avg_metrics_test['F1 Score'], marker='s', label='F1 Score', color=palette[1])
    plt.plot(rounds, avg_metrics_test['Precision'], marker='^', label='Precision', color=palette[2])
    plt.plot(rounds, avg_metrics_test['Recall'], marker='d', label='Recall', color=palette[3])'''

    # plt.plot(rounds, avg_metrics_test['Accuracy'], marker='o', label='Accuracy(%)', color=palette[0])
    # plt.plot([x + x_offset for x in rounds], avg_metrics_test['F1 Score'], marker='s', label='F1 Score(%)', color=palette[1])
    # plt.plot(rounds, avg_metrics_test['Precision'], marker='^', label='Precision(%)', color=palette[2])
    # plt.plot([x - x_offset for x in rounds], avg_metrics_test['Recall'], marker='d', label='Recall(%)', color=palette[3])

    # # Add convergence line to testing plot
    # if convergence_round and convergence_round in rounds_range:
    #     plt.axvline(x=convergence_round, color='blue', linestyle='--', label='Convergence Point')

    # plt.ylabel('Metric Value')
    # plt.xlabel('Round')
    # Define brighter colors with higher saturation
    bright_palette = ['#00C957', '#FF5500', '#8A2BE2', '#FF1493']  # Bright green, orange, blue, pink

    # Plot with thicker, more visible lines
    plt.plot(rounds, avg_metrics_test['Accuracy'], marker='o', label='Accuracy(%)', 
            color=bright_palette[0], linewidth=2.0, markersize=6)
    plt.plot([x + x_offset for x in rounds], avg_metrics_test['F1 Score'], marker='s', label='F1 Score(%)', 
            color=bright_palette[1], linewidth=2.0, markersize=6)
    plt.plot(rounds, avg_metrics_test['Precision'], marker='^', label='Precision(%)', 
            color=bright_palette[2], linewidth=2.0, markersize=6)
    plt.plot([x - x_offset for x in rounds], avg_metrics_test['Recall'], marker='d', label='Recall(%)', 
            color=bright_palette[3], linewidth=2.0, markersize=6)

    # Add convergence line to testing plot
    if convergence_round and convergence_round in rounds_range:
        plt.axvline(x=convergence_round, color='#4169E1', linestyle='--', label='Convergence Point', linewidth=1.5)

    plt.ylabel('Metric Value', fontweight='bold')
    plt.xlabel('Round', fontweight='bold')
    #plt.title('Average Testing Metrics per Round (Line Plot)')
    plt.xticks(rounds)
    plt.ylim(0, 1)
    plt.legend(markerscale=1.5, prop={'size': 10},loc='lower right', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    testing_line_path = os.path.join(output_dir, 'average_testing_metrics_line_plot.pdf')
    plt.savefig(testing_line_path, format='pdf')
    plt.close()
    logger.info(f"Average testing metrics line plot saved as '{testing_line_path}'.")

def plot_loss_line(metrics, rounds_range, output_dir, logger):
    avg_loss_per_round = []
    avg_train_loss_per_round = []
    for rnd in rounds_range:
        losses = []
        train_losses = []
        for node in metrics:
            if len(metrics[node]['loss']) >= rnd:
                losses.append(metrics[node]['loss'][rnd-1])
            if len(metrics[node]['train_loss']) >= rnd:
                train_losses.append(metrics[node]['train_loss'][rnd-1])
        avg_loss = np.nanmean(losses) if losses else np.nan
        avg_train_loss = np.nanmean(train_losses) if train_losses else np.nan
        avg_loss_per_round.append(avg_loss)
        avg_train_loss_per_round.append(avg_train_loss)
    
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(7, 4))
    plt.plot(rounds_range, avg_train_loss_per_round, marker='o', color='blue', label='Average Training Loss')
    plt.plot(rounds_range, avg_loss_per_round, marker='o', color='red', label='Average Test Loss')
    #plt.title('Average Loss over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, 'average_loss_over_rounds.pdf')
    plt.savefig(loss_plot_path, format='pdf')
    plt.close()
    logger.info(f"Average loss plot saved as '{loss_plot_path}'.")

def plot_training_aggregation_times(rounds_range, total_training_times, total_aggregation_times, total_execution_time, output_dir, logger):
    #plt.figure(figsize=(12, 6))
    plt.figure(figsize=(7, 4))
    plt.plot(rounds_range, total_training_times, marker='o', label='Total Training Time (s)', color='darkgreen')
    plt.plot(rounds_range, total_aggregation_times, marker='x', label='Total Aggregation Time (s)', color='steelblue')
    
    # Annotate total execution time only on training and aggregation times plot
    plt.text(0.95, 0.95, f'Total Execution Time: {total_execution_time:.2f} sec',
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.5))
    
    #plt.title('Training and Aggregation Times per Round')
    plt.xlabel('Round')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    times_plot_path = os.path.join(output_dir, 'training_aggregation_times_per_round.pdf')
    plt.savefig(times_plot_path, format='pdf')
    plt.close()
    logger.info(f"Training and aggregation times plot saved as '{times_plot_path}'.")

def plot_additional_metrics(rounds_range, cpu_usages, round_times, output_dir, logger):
    # Plot CPU Usage over Rounds
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(7, 4))
    plt.plot(rounds_range, cpu_usages, marker='o', label='CPU Usage (%)', color='darkorange')
    #plt.title('CPU Usage over Rounds')
    plt.xlabel('Round')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    cpu_plot_path = os.path.join(output_dir, 'cpu_usage_over_rounds.pdf')
    plt.savefig(cpu_plot_path, format='pdf')
    plt.close()
    logger.info(f"CPU usage plot saved as '{cpu_plot_path}'.")

    # Plot Round Times
    #plt.figure(figsize=(10, 6))
    plt.figure(figsize=(7, 4))
    plt.plot(rounds_range, round_times, marker='o', label='Round Time (s)', color='purple')
    #plt.title('Time Taken per Round')
    plt.xlabel('Round')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    round_time_plot_path = os.path.join(output_dir, 'round_times_over_rounds.pdf')
    plt.savefig(round_time_plot_path, format='pdf')
    plt.close()
    logger.info(f"Round times plot saved as '{round_time_plot_path}'.")
