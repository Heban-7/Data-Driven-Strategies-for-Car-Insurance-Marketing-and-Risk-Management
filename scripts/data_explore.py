import pandas as pd
def data_explore(df):
    """
    Explore the data to gain insights about the data.
    """
    print("Dimensions of the dataset")
    print(df.shape)

    print("\n\nView summary of the dataset")
    print(df.info())

    print("\n\nSumamry Statistics of numerical columns")
    print(df.describe())

    print("\n\nSummary Statistics for Character columns")
    print(df.describe(include=['object']))

    print("\n\nSummary Statistics for all the columns")
    print(df.describe(include='all'))
