import pandas as pd
import numpy as np
import io, argparse
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import matplotlib.ticker as tck
from matplotlib.lines import Line2D
from itertools import cycle, chain

def dicrete_plot(df, index_columns, metric_columns, error_columns, 
        unit=1.0, scale=1.0, border=2.0, legend_offset=1.0, capsize=10, barplot_margin=1.0, enable_minor_grid=True):

    # Initial assertions:
    if len(index_columns) == 0:
        raise ValueError('Empty list given, index_columns must be non-empty list')
    if len(metric_columns) == 0 or len(error_columns) == 0:
        raise ValueError('Empty list given, metric_columns and error_columns must be non-empty list')
    if len(metric_columns) != len(error_columns):
        raise ValueError('metric_columns and error_columns must be lists of equal size')

    # Extract labels:
    labels = {index_name: np.sort(df[index_name].unique()) for index_name in index_columns}

    # Specify GridSpec:
    widths = np.array([unit] * len(labels) + [6*unit])
    heights = np.array([unit] * len(df) + [legend_offset*unit])
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, axs = plt.subplots(
        ncols=len(widths), nrows=len(heights), subplot_kw=dict(frameon=False),
        constrained_layout=False, gridspec_kw=gs_kw, 
        figsize=(scale * widths.sum(), scale * heights.sum()))

    # Define color mappings:
    from matplotlib.cm import get_cmap
    cmaps = cycle(['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', ])
    def map_disc_(items, cmap_name):
        cmap_ = get_cmap(cmap_name)
        inter_ = 0.3
        slope_ = (1. - inter_) / len(items)
        return {value: cmap_(inter_ + items.tolist().index(value) * slope_) for value in items}
    colors = {label: map_disc_(items, cmap_name) \
        for (label, items), cmap_name in zip(labels.items(), cmaps)}

    # Find maximum value for metrics including error:
    max_m_e = pd.DataFrame({m + ' + ' + e: df[m] + df[e] \
        for i, (m, e) in enumerate(zip(metric_columns, error_columns))}).to_numpy().max()

    # Apply settings for axes:
    for r, row in enumerate(axs):
        for c, ax in enumerate(row):
            for _, spine in ax.spines.items():
                spine.set_visible(False)
            if r == len(axs) - 1:
                ax.set(xticks=[], yticks=[])
                ax.patch.set_alpha(0.)
                continue
            if c < len(labels):
                ax.set(xticks=[], yticks=[])
                var_name = list(labels.keys())[c]
                var_value = df[var_name][r]
                ax.add_patch(pch.Rectangle((0., 0.), 1., 1., 
                    facecolor=colors[var_name][var_value], edgecolor='black', label=var_value))
                for _, spine in ax.spines.items():
                    spine.set_linewidth(border)
            else:
                ax.set_xlim(0., max_m_e + barplot_margin)
                ax.set(yticks=[])
                ax.tick_params(grid_linestyle='solid', color='grey', zorder=0)
                if enable_minor_grid:
                    ax.tick_params(which='minor', grid_linestyle='dashed', grid_dashes=(3, 6), color='grey', zorder=0)
                    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
                ax.grid(which='both')
                if r < len(axs) - 2: 
                    ax.tick_params(labelbottom=False, length=10)
                if r == len(axs) - 2: 
                    ax.tick_params(labelbottom=True)
                    ax.spines['bottom'].set_visible(True)
                cmap = get_cmap('Greys')
                ax.patch.set_alpha(0.)
                inter_ = 0.1
                slope_ = (1. - inter_) / len(metric_columns)
                for i, (metric_name, error) in enumerate(zip(metric_columns, error_columns)):
                    clr = cmap(inter_ + (i + 1) * slope_)
                    ax.barh([i], [df[metric_name][r]], xerr=[[0.], [df[error][r]]], 
                        color=clr, edgecolor='black', height=1., capsize=capsize, label=metric_name, zorder=10.)
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

def main_aggregatedData(data_filename, headers_filename, output_filename):
    """Run main script"""

    # read in the headers
    data_used = pd.read_csv(headers_filename)
    #data_used = data[data["used_"]==True]
    index_columns = list(data_used.loc[:, "indices"])
    metric_columns = list(data_used.loc[:, "metrics"])
    error_columns = list(data_used.loc[:, "errors"])

    # read in the data
    df = pd.read_csv(data_filename, usecols=headers )

    # make the plot
    fig = dicrete_plot(df, index_columns, metric_columns, error_columns, scale=1.5, legend_offset=0.5, capsize=5)

    # Export to svg file:
    fig.savefig(output_filename)

def main_individualData(data_filename, headers_filename, output_filename, ascending=True, top_n=None):
    """Run main script
    
    Expected input headers are the following:
    indices
    metrics
    sort
    """

    # read in the headers
    data_used = pd.read_csv(headers_filename)
    index_columns = list(data_used.loc[:, "indices"].dropna())
    metric_columns = list(data_used.loc[:, "metrics"].dropna())
    sort_columns = list(data_used.loc[:, "sort"].dropna())

    headers = list(chain(index_columns, metric_columns, sort_columns))

    # read in the data
    df = pd.read_csv(data_filename, usecols=headers )

    # make the empty data frame for the aggregate statistics
    metric_means = [item + "_mean" for item in metric_columns]
    metric_errors = [item + "_error" for item in metric_columns]
    metric_sort = [item + "_error" for item in sort_columns]
    agg_stats = pd.DataFrame(columns=index_columns)

    # calculate the aggregated statistics (average and standard deviation)
    for metric in list(chain(metric_columns, sort_columns)):
        df_tmp = df.groupby(index_columns).agg(
            metric_means=pd.NamedAgg(column=metric, aggfunc="mean"),
            metric_errors=pd.NamedAgg(column=metric, aggfunc="std"))
        df_tmp = df_tmp.rename(columns={"metric_means": metric + "_mean", "metric_errors": metric + "_error"})
        agg_stats = agg_stats.merge(df_tmp, on=index_columns, how="outer")

    # sort
    agg_stats = agg_stats.sort_values(by=metric_sort, axis=0, ascending=ascending).reset_index(drop=True)

    # make the plot
    if top_n:
        fig = dicrete_plot(agg_stats.head(top_n), index_columns, metric_means, metric_errors, scale=1.5, legend_offset=0.5, capsize=5)
    else:        
        fig = dicrete_plot(agg_stats, index_columns, metric_means, metric_errors, scale=1.5, legend_offset=0.5, capsize=5)

    # Export to svg file:
    fig.savefig(output_filename)

def main(args):

    # Input files (aggregated data, default)
    #data_filename = "MultiIndicesBarPlotInput_aggregated.csv"
    #headers_filename = "MultiIndicesBarPlotHeaders_aggregated.csv"
    
    # Input files (individual data, default)
    data_filename = "MultiIndicesBarPlotInput.csv"
    headers_filename = "MultiIndicesBarPlotHeaders.csv"
    output_filename = "MultiIndicesBarPlot.svg"
    ascending = True
    top_n = 0

    ## Parse command line arguments
    #if args.data_filename:
    #    data_filename = args.data_filename
    #if args.headers_filename:
    #    headers_filename = args.headers_filename
    #if args.output_filename:
    #    output_filename = args.output_filename
    if args.ascending:
        ascending = args.ascending
    if args.top_n:
        top_n = args.top_n
    print("input data: " + data_filename)
    print("input headers: " + headers_filename)
    print("output: " + output_filename)
    print("ascending: " + str(ascending))
    print("top_n: " + str(top_n))

    #main_aggregatedData(data_filename, headers_filename)
    main_individualData(data_filename, headers_filename, output_filename, ascending, top_n)

# Run main
if __name__ == "__main__":

    # Initialize parser
    parser = argparse.ArgumentParser()
 
    # Adding optional argument
    parser.add_argument("-d", "--data", dest="data_filename", help = "Input data svg")
    parser.add_argument("-l", "--headers", dest="headers_filename", help = "Input headers csv")
    parser.add_argument("-o", "--output", dest="output_filename", help = "Output svg")
    parser.add_argument("-a", "--ascending", type=bool, dest="ascending", help = "True: sort by ascending order; False: sort by descending order")
    parser.add_argument("-n", "--topn", type=int, dest="top_n", help = "Top N rows to use for plotting; 0 = use all")
 
    # Read arguments from command line
    args = parser.parse_args()
    main(args)