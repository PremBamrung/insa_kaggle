import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def show_missing(df, save=False, title="", size=(15, 10), show_all=False, dpi=300, sorting_missing=True):
    """

    Plot the missing values in a Panda Serie with their percentage. Horizontal bar ascending, saveable.

    """
    percent_missing = df.isnull().sum() * 100 / len(df)
    # show all columns, even those without nan
    if show_all == False:
        percent_missing = percent_missing[percent_missing != 0]
        column = percent_missing.index
    else:
        column = df.columns
    missing_value_df = pd.DataFrame(
        {'column_name': column, 'percent_missing': percent_missing})

    # sort by missing values or by column name
    if sorting_missing:
        missing_value_df = missing_value_df.sort_values(
            "percent_missing", ascending=False)
    else:
        missing_value_df = missing_value_df.sort_values(
            "column_name", ascending=False)

    ax = missing_value_df.plot(kind="barh", figsize=(size), legend=None)
    plt.title("Missing values", fontsize=19)
    plt.xlabel("Missing Values %", fontsize=18)
    plt.ylabel("Columns", fontsize=18)

    # set individual bar labels using above list
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width() + .3, i.get_y() + .2,
                str(round(i.get_width(), 3)) + '%')

    if save == True:
        plt.savefig(f'{title}.png', dpi=dpi)


def show_value(series, save=False, title="", size=(15, 10)):
    """

    Plot the different values in a Panda Serie with their percentage. Horizontal bar ascending, saveable.

    """

    fig = series.value_counts().sort_values(ascending=False).plot(
        kind='barh', figsize=(size), legend=None)
    plt.title("Class distribution", fontsize=19)
    plt.xlabel("Sample", fontsize=18)
    plt.ylabel("Columns", fontsize=18)
    for i in fig.patches:
        # get_width pulls left or right; get_y pushes up or down
        fig.text(i.get_width() + .3, i.get_y() + .2,
                 str(round(i.get_width() / len(series) * 100, 2)) + '%')
    if save == True:
        plt.savefig(f'{title}.png', dpi=300)


def z_score(input_serie, threshold=3,return_df=False):
    """

    Outliers removal using z score. Take a Panda Serie as input and return a Panda Serie without the outliers

    """
    df = pd.DataFrame()
    df["input"] = pd.Series(input_serie)
    mean = df.input.mean()
    std = df.input.std()
    df["z_score"] = df.input.apply(lambda x: (x - mean) / std)
    output_serie = df.input.where(np.abs(df.z_score) < threshold)
    output_serie.dropna(inplace=True)
    if return_df:
    	return df
    else:
    	return  output_serie


def iqr(series, fences=False):
    """

    Outliers removal using IQR or Tukey Fence. Take a Panda Serie as input and return a Panda Serie without the outliers

    """
    series = sorted(series)
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    series = pd.Series(series)
#     return series[series.between(lower, upper)]
    if fences:
        return lower, upper
    else:
        return series.apply(lambda x: x if (x > lower and x < upper) else np.nan)


def compare(list_df):
    """

    Compare two dataframe or series using the describe method of pandas. The input is a list of
    dataframe or series

    """
    df = pd.DataFrame()
    for index, l in enumerate(list_df):
        df[f"{variablename(l)}"] = l.describe()
    return df


def variablename(var):
    """

    Return the name of a variable

    """
    return [tpl[0] for tpl in filter(lambda x: var is x[1], globals().items())]
# series= [10,12,12,13,12,11,14,13,15,10,10,10,10,12,14,13, 12,1000000, 100000000000001,11,12,15,12,13,12,11,14,13,15,10,15,12,10,14,13,15,10,100000000000000000000]
# print(z_score(series))
# print(iqr(series))


def outliers_detect(variable, compare=False):
    """

    Plot series without outlier using z score and IQR/Tukey Fence

    """
    zs_var = z_score(variable)
    iqr_var = iqr(variable)

    plt.subplot(3, 1, 1)
    sns.boxplot(zs_var)
    plt.title(f"Z score ({round(len(zs_var.dropna())/len(variable)*100,2)})")
    plt.xlabel("")

    plt.subplot(3, 1, 2)
    sns.boxplot(iqr_var)
    plt.title(
        f"IQR/Tukey Fence ({round(len(iqr_var.dropna())/len(variable)*100,2)})")

    plt.subplot(3, 1, 3)
    sns.boxplot(variable)
    plt.title("Original data")

    plt.tight_layout()
    plt.show()

    if compare:
        display(compare((zs_var, iqr_var, variable)))


def get_ata(lo):
    """

    Get the ATA from LO_CHECKEED (2 first digits). Use directly on a serie without apply.

    """

    if isinstance(lo, pd.Series):
        lo = lo.astype(str).tolist()
    ata_lst = []
    for l in lo:
        ata = ''
        for c in l:
            if c.isdigit() and len(ata) < 2:
                ata = ata + c
        ata_lst.append(ata)
    return ata_lst


def get_ID(line):
    """

    Get the MSN ID from LO_CHECKED_EF. Use as apply function

    """
    result = []
    if line is not np.nan:
        sp0 = line.split(",")
        for n in sp0:
            sp1 = n.split("-")
            if len(sp1) == 2:
                result_id = [int(x)
                             for x in range(int(sp1[0]), int(sp1[1]) + 1)]
                result.append(result_id)
            else:
                result.append([int(sp1[0])])
        result = [item for sublist in result for item in sublist]

    else:
        result = line

    return result


def sum_ascii(string):
    """

    Sum the ascii values of each caracter in a string.
    Usefull to detect outliers in a list a string that has the same structure (MSN001,MSN0002...)

    """
    sum = 0
    for s in string:
        sum = sum + ord(s)
    return sum


def df_find(df, string):
    """

    Looking for elements that has a string in the entire df

    """

    test = df[df.applymap(lambda x: string in str(x)).any(axis=1)]
    return test


""" df_find list version"""
# servants=["Okita","Frankenstein","Yorimitsu"]
# result=pd.DataFrame()
# for servant in servants:
#     result=pd.concat([result,ut.df_find(df,servant)])
# result

""" count the number of time the string is found in a row"""
# def nb_count(row,string):
#     return int(row.str.count(string).sum())

# find=["Merlin","Kaleidoscope"]
# toto=df.copy()
# for i in find:
#     toto=df_find(toto,i)
#     toto[f"nb_{i}"]=toto.apply(nb_count,args=(i,),axis=1)
# toto.sort_values("Price($)",inplace=True)
# toto


""" Change all the column with dtype A to dtype B """
# df[df.select_dtypes(include=['float']).columns]=df[df.select_dtypes(include=['float']).columns].astype(int)



def reduce_mem_usage(df, verbose=True):
    """

    Reduce the memory usage of a df by redducing  number type 

    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased from {:5.2f} to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem,end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df