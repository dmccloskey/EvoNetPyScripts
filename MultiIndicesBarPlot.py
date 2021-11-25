import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from matplotlib.lines import Line2D
from itertools import cycle, chain

def dicrete_plot(df, index_columns, metric_columns, error_columns, unit=1.0, scale=1.0, border=2.0, legend_offset=1.0, capsize=10):
    # Initial assertions:
    if len(index_columns) == 0:
        raise ValueError('Empty list given, index_columns must be non-empty list')
    if len(metric_columns) == 0 or len(error_columns) == 0:
        raise ValueError('Empty list given, metric_columns and error_columns must be non-empty list')
    if len(metric_columns) != len(error_columns):
        raise ValueError('metric_columns and error_columns must be lists of equal size')

    # Extract labels:
    labels = {}
    for index_name in index_columns:
        labels[index_name] = np.sort(df[index_name].unique())

    # Specify GridSpec:
    widths = np.array([unit] * len(labels) + [6*unit])
    heights = np.array([unit] * len(df) + [legend_offset*unit])
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, axs = plt.subplots(
        ncols=len(widths), nrows=len(heights), 
        constrained_layout=False, gridspec_kw=gs_kw, 
        figsize=(scale * widths.sum(), scale * heights.sum()))

    # Define color mappings:
    from matplotlib.cm import get_cmap
    colors = {}
    cmaps = cycle(['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', ])
    for c, ((label, items), cmap_name) in enumerate(zip(labels.items(), cmaps)):
        cmap = get_cmap(cmap_name)
        inx_colors = {}
        for value in items:
            inx_colors[value] = cmap(0.3 + items.tolist().index(value) * 0.2)
        colors[label] = inx_colors

    for r, row in enumerate(axs):
        for c, ax in enumerate(row):
            ax.set(xticks=[], yticks=[])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            if r == len(axs) - 1:
                ax.patch.set_alpha(0.)
                continue
            if c < len(labels):
                var_name = list(labels.keys())[c]
                var_value = df[var_name][r]
                ax.add_patch(pch.Rectangle((0., 0.), 1., 1., 
                    facecolor=colors[var_name][var_value], edgecolor='black', label=var_value))
                ax.spines['top'].set_linewidth(border)
                ax.spines['right'].set_linewidth(border)
                ax.spines['bottom'].set_linewidth(border)
                ax.spines['left'].set_linewidth(border)
            else:
                cmap = get_cmap('Greys')
                ax.patch.set_alpha(0.)
                inter_ = 0.1
                slope_ = (1. - inter_) / len(metric_columns)
                for i, (metric_name, error) in enumerate(zip(metric_columns, error_columns)):
                    clr = cmap(inter_ + (i + 1) * slope_)
                    ax.barh([i], [df[metric_name][r]], xerr=[[0.], [df[error][r]]], 
                        color=clr, edgecolor='black', height=1., capsize=capsize, label=metric_name)
                handles, lbs = ax.get_legend_handles_labels()

    # Build the legend:
    sizes = [ll.shape[0] for ln, ll in labels.items()]
    largest_label_inx = np.argmax(sizes)

    ## Categorical variables:
    hndl = []
    for i, (lbl, colors_) in enumerate(colors.items()):
        size, empty = len(colors_), []
        if i >= 1:
            m = sizes[largest_label_inx] - sizes[i - 1]
            if m > 0:
                empty = [Line2D([], [], label='', alpha=0.)] * m
        hndl += empty + [Line2D([], [], label=lbl, alpha=0.)] + [
            pch.Patch(facecolor=color, edgecolor="k", label=label, alpha=0.7) 
            for label, color in colors_.items()
        ]

    ## Continuous variables:
    empty = []
    m = sizes[largest_label_inx] - sizes[-1] + 1
    if m > 0:
        empty = [Line2D([], [], label='', alpha=0.)] * m
    hndl += empty + handles + [
        Line2D([], [], label='', alpha=0.)] * (sizes[largest_label_inx] - len(metric_columns))
        
    legend = fig.legend(handles=hndl, loc='lower center', handlelength=scale*1.4, handleheight=scale*1.6, ncol=len(index_columns) + 1, labelspacing=.0)
    legend.get_frame().set_alpha(0.)
    plt.subplots_adjust(hspace=0.1, wspace=0.)
    return fig

def main_aggregatedData(data_dir, data_filename, headers_filename):
    """Run main script"""

    # read in the headers
    data_used = pd.read_csv(headers_filename)
    #data_used = data[data["used_"]==True]
    index_columns = list(data_used.loc[:, "indices"])
    metric_columns = list(data_used.loc[:, "metrics"])
    error_columns = list(data_used.loc[:, "errors"])
    headers = list(chain(index_columns, metric_columns, error_columns))

    # read in the data
    df = pd.read_csv(data_filename, usecols=headers )

    # make the plot
    fig = dicrete_plot(df, index_columns, metric_columns, error_columns, scale=1.5, legend_offset=0.5, capsize=5)

    # Export to svg file:
    fig.savefig('MultiIndicesBarPlot.svg')

def main_individualData(data_dir, data_filename, headers_filename):
    """Run main script"""

    # read in the headers
    data_used = pd.read_csv(headers_filename)
    index_columns = list(data_used.loc[:, "indices"])
    metric_columns = list(data_used.loc[:, "metrics"])
    headers = list(chain(index_columns, metric_columns))

    # read in the data
    df = pd.read_csv(data_filename, usecols=headers )

    # make the empty data frame for the aggregate statistics
    metric_means = [item + "_mean" for item in metric_columns]
    metric_errors = [item + "_error" for item in metric_columns]
    agg_stats = pd.DataFrame(columns=index_columns)

    # calculate the aggregated statistics (average and standard deviation)
    for metric in metric_columns:
        df_tmp = df.groupby(index_columns).agg(
            metric_means=pd.NamedAgg(column=metric, aggfunc="mean"),
            metric_errors=pd.NamedAgg(column=metric, aggfunc="std"))
        df_tmp = df_tmp.rename(columns={"metric_means": metric + "_mean", "metric_errors": metric + "_error"})
        agg_stats = agg_stats.merge(df_tmp, on=index_columns, how="outer")       

    # make the plot
    fig = dicrete_plot(agg_stats, index_columns, metric_means, metric_errors, scale=1.5, legend_offset=0.5, capsize=5)

    # Export to svg file:
    fig.savefig('MultiIndicesBarPlot.svg')

# Run main
if __name__ == "__main__":

    data_dir = ""

    # Input files (aggregated data)
    #data_filename = data_dir + "MultiIndicesBarPlotInput_aggregated.csv"
    #headers_filename = data_dir + "MultiIndicesBarPlotHeaders_aggregated.csv"
    #main_aggregatedData(data_dir, data_filename, headers_filename)
    
    # Input files (individual data)
    data_filename = data_dir + "MultiIndicesBarPlotInput.csv"
    headers_filename = data_dir + "MultiIndicesBarPlotHeaders.csv"

    main_individualData(data_dir, data_filename, headers_filename)