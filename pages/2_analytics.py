import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import numpy as np
# load data
instrument_map = pd.read_csv('./resources/instrument_map.csv')
train_df = pd.read_csv('./resources/viz/train_df.csv')
calib_df = pd.read_csv('./resources/viz/calib_df.csv',index_col=None)

with open('resources/viz/cm.pkl', 'rb') as file:
    cm = pickle.load(file)

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


def calculate_bucket_metrics(df, pred_filter=None):
    """
    Buckets `prob` in steps of 0.05, and calculates the percentage of `label` matching `pred` for each bucket.
    Optionally filters rows where `pred` equals a specific value.

    Args:
        df (pd.DataFrame): A DataFrame with columns `label`, `pred`, and `prob`.
        pred_filter (optional): A value to filter the `pred` column. If None, no filtering is applied.

    Returns:
        pd.DataFrame: A DataFrame with columns `bucket`, `match_percentage`, `count`, and `bucket_midpoint`.
    """
    # Drop rows with NaNs in relevant columns
    df = df.dropna(subset=['label', 'pred', 'prob'])
    
    # Optional filtering based on `pred_filter`
    if pred_filter != 'all':
        df = df[df['pred'] == pred_filter]

    # Create buckets for `prob`
    bins = [i * 0.05 for i in range(21)]
    df['bucket'] = pd.cut(df['prob'], bins=bins, right=False)

    # Group by buckets and calculate metrics
    grouped = df.groupby('bucket').agg(
        match_percentage=('label', lambda x: 100 * (x == df.loc[x.index, 'pred']).mean()),
        count=('label', 'size')
    ).reset_index()

    # Add bucket midpoints
    grouped['bucket_midpoint'] = grouped['bucket'].apply(lambda x: x.mid * 100)

    return grouped


def plot_calibration(grouped_df, instrument):
    # Create a new figure
    fig = plt.figure(figsize=(8, 5))  # Set figure size

    # Create the plot on the figure
    plt.scatter(
        grouped_df['bucket_midpoint'],              # X-axis: bucket midpoints
        grouped_df['match_percentage'],            # Y-axis: match percentage
        s=grouped_df['count'] * 10,                     # Bubble size: proportional to count
        alpha=0.6,                                 # Transparency for better visualization
        label='Count of predictions',
        color='blue'                               # Bubble color
    )

    # Plot the y=x line
    plt.plot(
        grouped_df['bucket_midpoint'], 
        grouped_df['bucket_midpoint'], 
        linestyle='--', 
        color='black', 
        label="Perfect Calibration"
    )

    # Add labels, title, legend, and grid
    plt.xlabel('Confidence (Percentage)')
    plt.ylabel('Accuracy (Percentage)')
    plt.title(f'Calibration plot for {instrument} predictions')
    plt.ylim(0, 140)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.text(
        x=10,                                   # Position near the left edge of the chart
        y=130,                                  # Position near the top
        s='Underconfident',                     # Text to display
        fontsize=12,                            # Font size for annotation
        color='red',                            # Text color
        ha='left',                              # Align text to the left
        va='top'                                # Align text to the top
    )

    plt.text(
        x=75,                                   # Position near the left edge of the chart
        y=20,                                   # Position near the top
        s='Overconfident',                      # Text to display
        fontsize=12,                            # Font size for annotation
        color='red',                            # Text color
        ha='left',                              # Align text to the left
        va='top'                                # Align text to the top
    )

    # Return the figure object
    return fig

#### Display Analytics ####

# data set characteristics
## Histogram by instrument

# training accuracy
st.header("Training Accuracy over Time")
fig = plot_train_acc(train_df)
st.write(fig)


# confusion matrix
import matplotlib
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

fig, ax = plt.subplots()
# create list of instruments
instruments = instrument_map['instrument_name'].unique().tolist()

im, cbar = heatmap(cm, instruments, instruments, ax=ax,
                   cmap="YlGn", cbarlabel="count")
texts = annotate_heatmap(im, valfmt="{x:2d}")

fig.tight_layout()
st.header('Confusion Matrix for Validation Set (N = 1,527)')
st.write(fig)
# st.write(cm)
# calibration
st.header("Calibration Plots by Instrument")

leftcol, rightcol = st.columns([2, 1])

# add 'all' to instruments for checkbox
instruments.insert(0,'all')


with rightcol:  # plot setup selectors on the right
    instrument = st.selectbox("Instrument", instruments)
    if instrument == 'all':
        target_instrument = 'all'
    else:
        target_instrument = instrument_map.loc[instrument_map['instrument_name']==instrument]['target_instrument'].iloc[0]

with leftcol:
    grouped_df = calculate_bucket_metrics(df = calib_df,pred_filter=target_instrument)
    fig = plot_calibration(grouped_df,instrument)
    st.write(fig)




