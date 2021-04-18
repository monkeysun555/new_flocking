import numpy as np
import matplotlib
import matplotlib.pyplot as plt
PLOT_SEG = 31       #22
TYPES = [3,4,1,2,100]

INTERVALS = [1,2,3,4,5]
def get_data(path):
    data = {}
    line_counter = 0
    t, interval, row = None, None, None
    with open(path, 'r') as fr:
        for line in fr:
            line = line.strip('\n').split()
            if line_counter%17 == 0:
                assert len(line) == 2
                t, interval = int(line[0]), int(line[1])
                row_num = 0
                if t not in data:
                    data[t] = [np.zeros((16, 32)) for _ in range(5)]
            else:
                line = [float(x) for x in line]
                for col in range(32):
                    data[t][interval][row_num,col] = line[col]
                row_num += 1
            line_counter += 1
    return data

def translate_ylabel(t):
    if t == 1:
        return 'LSTM$_{t}$'
    elif t == 2:
        return 'LSTM$_{c}$'
    elif t == 3:
        return 'TLP'
    elif t == 4:
        return 'Co-TLP'
    elif t == 100:
        return 'GT'

def plot_heatmap(data):

    fig, axes = plt.subplots(5, 5, figsize=(10, 8))

    # Replicate the above example with a different font size and colormap.
    for i in range(len(TYPES)):
        for j in range(len(INTERVALS)):
            y_label, x_label = None, None 
            t, interval = TYPES[i], INTERVALS[j]
            if j == 0:
                y_label = translate_ylabel(t)
            if i == 4:
                x_label = interval
            test_data = data[t][j]
            ax = axes[i][j]
            bar = False
            if j == 4:
                bar = True
            im = heatmap(test_data, row_labels = y_label, col_labels = x_label, ax=ax, bar= bar)
            # annotate_heatmap(im, valfmt="{x:.1f}", size=7)

    # Create some new data, give further arguments to imshow (vmin),
    # use an integer format on the annotations and provide some colors.

    # data = np.random.randint(2, 100, size=(7, 7))
    # y = ["Book {}".format(i) for i in range(1, 8)]
    # x = ["Store {}".format(i) for i in list("ABCDEFG")]
    # im, _ = heatmap(data, y, x, ax=ax2, vmin=0,
    #                 cmap="magma_r", cbarlabel="weekly sold copies")
    # annotate_heatmap(im, valfmt="{x:d}", size=7, threshold=20,
    #                  textcolors=("red", "white"))

    # Sometimes even the data itself is categorical. Here we use a
    # `matplotlib.colors.BoundaryNorm` to get the data into classes
    # and use this to colorize the plot, but also to obtain the class
    # labels from an array of classes.

    # data = np.random.randn(6, 6)
    # y = ["Prod. {}".format(i) for i in range(10, 70, 10)]
    # x = ["Cycle {}".format(i) for i in range(1, 7)]

    # qrates = list("ABCDEFG")
    # norm = matplotlib.colors.BoundaryNorm(np.linspace(-3.5, 3.5, 8), 7)
    # fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])

    # im, _ = heatmap(data, y, x, ax=ax3,
    #                 cmap=plt.get_cmap("PiYG", 7), norm=norm,
    #                 cbar_kw=dict(ticks=np.arange(-3, 4), format=fmt),
    #                 cbarlabel="Quality Rating")

    # annotate_heatmap(im, valfmt=fmt, size=9, fontweight="bold", threshold=-1,
    #                  textcolors=("red", "black"))

    # We can nicely plot a correlation matrix. Since this is bound by -1 and 1,
    # we use those as vmin and vmax. We may also remove leading zeros and hide
    # the diagonal elements (which are all 1) by using a
    # `matplotlib.ticker.FuncFormatter`.

    # corr_matrix = np.corrcoef(harvest)
    # im, _ = heatmap(corr_matrix, vegetables, vegetables, ax=ax4,
    #                 cmap="PuOr", vmin=-1, vmax=1,
    #                 cbarlabel="correlation coeff.")


    # def func(x, pos):
    #     return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")

    # annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)
    left_pos = 0.08
    vertial_h = [0.79, 0.61, 0.455, 0.295, 0.165]
    for i in range(len(TYPES)):
        t = TYPES[i]
        v_height = vertial_h[i]
        fig.text(left_pos, v_height, translate_ylabel(t) , rotation='vertical', fontsize=22)
    
    bot_height = 0.1
    horizontal_pos = [0.2, 0.36, 0.52, 0.68, 0.84]

    for i in range(len(TYPES)):
        fig.text(horizontal_pos[i], bot_height, str(i+1) ,  ha='center', fontsize=22)

    fig.text(0.52, 0.04, 'Prediction Interval (s)', ha='center', fontsize=24)
    # plt.tight_layout()
    plt.show()
    input()
    fig.savefig('./figures/fov_prediction/hm.eps', format='eps', dpi=250, figsize=(10, 8))

def heatmap(data, row_labels = None, col_labels = None, ax=None, bar = False,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, vmin=0, vmax=0.006, **kwargs)

    # Create colorbar
    # if bar:
        # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    # ax.set_xticks(np.arange(data.shape[1]))
    # ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    # if col_labels:
        # ax.set_xticklabels(col_labels)
        # ax.text(0.5, 0.025, col_labels, ha='center', fontsize=16)
        # ax.text(0.05, 0.5, 'KL Divergence ', va='center', rotation='vertical', fontsize=24)
    # if row_labels:
        # ax.set_yticklabels(row_labels)
        # ax.text(0.5, 0.025, 'Prediction Interval (s)', ha='center', fontsize=24)
        # ax.text(-0.1, 0.45, row_labels, va='center', rotation='vertical', fontsize=16)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=False, 
                   labeltop=False, labelbottom=False, 
                   right=False,left=False,
                   labelleft=False,labelright = False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.tick_params(which="minor", bottom=False, left=False)

    return im


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
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
    #         text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
    #         texts.append(text)

    return texts

def main():
    path = './fov_prediction/pred_dis/' + str(PLOT_SEG) + '.txt'

    plot_data = get_data(path)
    for x in plot_data[1]:
        print(sum(x))
    plot_heatmap(plot_data)

if __name__ == '__main__':
    main()