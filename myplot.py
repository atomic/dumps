import pandas
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import math


def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def colorbar_index(ncolors, cmap, data):
    """Put the colorbar labels in the correct positions
        using unique levels of data as tickLabels
    """

    cmap = cmap_discretize(cmap, ncolors)
    mappable = matplotlib.cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors + 0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(np.unique(data))


def cmap_xmap(function, cmap):
    """ Applies function, on the indices of colormap cmap. Beware, function
    should map the [0, 1] segment to itself, or you are in for surprises.

    See also cmap_xmap.
    """
    cdict = cmap._segmentdata
    function_to_map = lambda x: (function(x[0]), x[1], x[2])
    for key in ('red', 'green', 'blue'):
        cdict[key] = map(function_to_map, cdict[key])
        cdict[key].sort()
        assert (cdict[key][0] < 0 or cdict[key][-1] > 1), "Resulting indices extend out of the [0, 1] segment."

    return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)

def plot_categorical_map(ax, df_map, colorbar=True, nlargest=None, fig=None):
    #
    # alternatively:
    # sns.heatmap(X.vegetation.values.reshape([m,n])
    #           , yticklabels=False
    #           , xticklabels=False
    #           , cmap=sns.color_palette('Set1', fuel_sets))
    #
    if nlargest:
        codes   = pd.Series(df_map.values.flatten()).value_counts().nlargest(nlargest).index.tolist()
        codes.append(-1)
    else:
        codes   = sorted(list(set(df_map.values.flatten())))
    mapping = dict([ (code,i) for i,code in enumerate(codes)])
    getcode = lambda e: mapping[e] if e in mapping else mapping[-1]
    encoded = df_map.applymap(getcode)
    k       = len(codes)
    c       = cmap_discretize('jet', k)
    pos     = ax.imshow(encoded, interpolation='nearest', cmap=c)
    if colorbar:
        if fig:
            cb  = fig.colorbar(pos, ax=ax)
        else:
            cb  = ax.colorbar()
        cb.set_ticks(list(range(k)))
        cb.set_ticklabels(codes)

def plot_feature(fig, ax, data, name, m, n):
    df_toplot = pd.DataFrame( data[name].values.reshape([m,n]) )
    if name == 'fuel_model':
        plot_categorical_map(ax, df_toplot , fig=fig)
    elif name == 'vegetation':
        plot_categorical_map(ax, df_toplot , nlargest=10, fig=fig)
    elif name == 'elevation':
        ax.imshow( df_toplot, vmin=0, vmax=2000)
    else:
        ax.imshow( df_toplot, vmin=0 )          # for most data, minimum should be 0, except vegetation
    ax.set_title(name, fontsize=30, color='black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    
def quick_plot(feature, m, n, vmin=None, vmax=None):
    plt.figure(figsize=(14,7))
    plt.imshow(feature.values.reshape([m,n]), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()

def plot_all_data(data, col=3):
    m,n = data['m'], data['n']
    data = data['data']
    features = data.columns.tolist()
    row      = int(math.ceil(len(features) / col))
    f, axarr = plt.subplots( row + 1, col, sharey=False,figsize=(28,18))
    for i,name in enumerate(features):
        plot_feature(f, axarr[ int(i/col), i % col], data, name, m, n)
    plt.show()

def compare_hist(axx, A, B, title=None, threshold=0, width=1):
    df_AB = pd.concat([A.value_counts(), B.value_counts()], axis=1, keys=['A','B'])
    df_AB = df_AB / df_AB.sum()
    df_AB = df_AB[df_AB > threshold].dropna()
    plt.figure(figsize=(25,6))
    df_AB.plot.bar(ax=plt.gca(), width=width, alpha=0.7)
    plt.tick_params(axis='x', colors='black', labelsize=15)
    plt.tick_params(axis='y', colors='black', labelsize=15)
    plt.title(title, color='black', fontsize=25)
    plt.show()
