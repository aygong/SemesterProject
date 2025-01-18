import matplotlib.pyplot as plt
import numpy as np

def plot_SL_results(path_plot, path_result, filename, until_epoch=None):
    """
    Plot the results of the supervised learning, namely the training loss and accuracy, and validation loss and MAE
    """
    res = dict(np.load(path_result + filename + '.npz'))
    if until_epoch:
        for k,v in res.items():
            res[k] = v[:until_epoch]
    fig,axs = plt.subplots(2,2, figsize=(10,6), sharex=True)
    axs[0,0].plot(range(len(res['train_policy_performance'])), res['train_policy_performance'], c='g')
    axs[0,0].set_title('Train policy accuracy')
    axs[0,0].set_ylabel('Accuracy')
    axs[0,1].plot(range(len(res['train_value_performance'])), res['train_value_performance'], c='r')
    axs[0,1].set_title('Train cost-to-go MAE')
    axs[0,1].set_ylabel('MAE')
    axs[1,0].plot(range(len(res['valid_policy_performance'])), res['valid_policy_performance'], c='g')
    axs[1,0].set_title('Validation policy accuracy')
    axs[1,0].set_xlabel('Epochs')
    axs[1,0].set_ylabel('Accuracy')
    axs[1,1].plot(range(len(res['valid_value_performance'])), res['valid_value_performance'], c='r')
    axs[1,1].set_title('Validation cost-to-go MAE')
    axs[1,1].set_xlabel('Epochs')
    axs[1,1].set_ylabel('MAE')
    
    plt.tight_layout()
    plt.savefig(path_plot + filename + '.svg')


def compare_accuracy(path_plot, path_result, file_list, title, plot_filename, labels=None, until_epoch=None):
    """
    Show all the accuracies of the file in file_list on the same plot for comparison.
    """
    fig,axs = plt.subplots(1,2, figsize=(10,6), sharex=True)
    for i,filename in enumerate(file_list):
        res = dict(np.load(path_result + filename + '.npz'))
        if until_epoch:
            for k,v in res.items():
                res[k] = v[:until_epoch]
        if labels:
            label = labels[i]
        else:
            label = filename[6:-17]

        axs[0].plot(range(len(res['train_policy_performance'])), res['train_policy_performance'], label=label)
        axs[0].set_title('Train policy accuracy')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Accuracy')

        axs[1].plot(range(len(res['valid_policy_performance'])), res['valid_policy_performance'], label=label)
        axs[1].set_title('Validation policy accuracy')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')

    plt.legend()
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(path_plot + plot_filename + '.svg')