import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(df, cmap='coolwarm', annot=False, fmt='.2f', save_filename=None):
    """
    Plot a heatmap from a DataFrame and optionally save it to a file.

    Args:
        df (pd.DataFrame): The DataFrame to be plotted.
        cmap (str): The colormap to use (default is 'coolwarm').
        annot (bool): Whether to annotate the cells with the data values (default is False).
        fmt (str): The format specifier for the data values (default is '.2f').
        save_filename (str): The filename to save the plot. If None, the plot is displayed but not saved.
    """
    # 检查行名和列名是否相等
    if set(df.columns) == set(df.index):
        sorted_columns = sorted(df.columns)
        df = df[sorted_columns]
        df = df.reindex(sorted_columns)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap=cmap, annot=annot, fmt=fmt)
    plt.xlabel('dataset1 label')
    plt.ylabel('dataset2 label')
    plt.title('SCOTCH label transfer')

    if save_filename:
        plt.savefig(save_filename)
    else:
        plt.show()