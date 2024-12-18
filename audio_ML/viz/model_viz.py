def plot_train_acc(train_df):
    """Create (colored) scatter plot with optional regression line.
    """
    # ax.set_title(f'Pred Filter = {pred_filter}')
    # ax.set_xlabel('Confidence Midpoint')
    # ax.set_ylabel('Accuracy')

    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    ax.set_title('Loss over time')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.plot(train_df['epoch'],train_df['loss'])

    ax = axs[1]
    ax.set_title('Accuracy over time')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy')
    ax.plot(train_df['epoch'],train_df['accuracy'])
    fig.tight_layout()
    return fig