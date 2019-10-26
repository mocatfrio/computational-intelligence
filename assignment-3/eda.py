# Import packages
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
# import seaborn as sns # used for plot interactive graph
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

"""
Visualize Correlation
"""
# using matshow matplotlib
# def visualize_correlation(df, method, title):
#     path = os.getcwd() + '/figure/correlation/'
#     plt.matshow(df.corr(method=method), vmax=1, vmin=-1, cmap='PRGn')
#     plt.title(title, size=12)
#     plt.colorbar()
#     # plt.show()
#     plt.savefig(path + title + '.png')
#     print(title + '.png saved!')

# using heatmap seaborn
def visualize_correlation(df, method, title):
    path = os.getcwd() + '/eda/correlation/'
    f, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(df.corr(method=method), annot=True, fmt='.3f', ax=ax)
    ax.set_title(title)
    # plt.show()
    plt.savefig(path + title + '.png')
    print(title + '.png saved!')

"""
Visualize Data
- Over day (all field)
  - plot
  - hist (distribution)
- Over year (certain field)
  - plot
  - hist (distribution)
- Over month
"""
def visualize_data(vtime, vtype, df, path, title, col=None, years=None, year=None):
  create_directory(path)
  fig, ax = plt.subplots(figsize=(18,18))
  if vtime == 'd':
    for i in range(len(df.columns)):
      plt.subplot(len(df.columns), 1, i+1)
      name = df.columns[i]
      if vtype == 'plot':
        plt.plot(df[name])
      elif vtype == 'hist':
        df[name].hist(bins=200)
      plt.title(name, y=0, loc = 'right')
      plt.yticks([])
  elif vtime == 'y':
    for i in range(len(years)):
      plt.subplot(len(years), 1, i+1)
      year = years[i]
      data = df[str(year)]
      if vtype == 'plot':
        plt.plot(data[col])
      elif vtype == 'hist':
        data[col].hist(bins = 200)
      plt.title(str(year), y = 0, loc = 'left')
  elif vtime == 'm':
    months = [i for i in range(1,13)]
    for i in range(len(months)):
      ax = plt.subplot(len(months), 1, i+1)
      month = year + '-' + str(months[i])
      try:
        data = df[month]
        data[col].hist(bins = 100)
        ax.set_xlim(0,5)
        plt.title(month, y = 0, loc = 'right')
      except:
        break
  # plt.show()
  ax.set_title(title)
  fig.tight_layout()
  fig.savefig(path + title + '.png')
  print(path + title + '.png saved!')

"""
Automatically make directory
"""
def create_directory(path):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
  # constant path
  PATH = os.getcwd()
  DATASET_PATH = PATH + '/dataset/'
  EDA_PATH = PATH + '/eda/'

  fill_methods = ['custom', 'mean', 'median', 'mode', 'bfill', 'ffill', 'linear', 'polynomial']
  years = ['2007', '2008', '2009', '2010']

  for i in range(0, len(fill_methods)):
    fill_methods[i] = 'power_consumption_' + fill_methods[i]

  for data_name in fill_methods:
    df = pd.read_csv(DATASET_PATH + data_name + '.csv', 
                      parse_dates = True, index_col = 'datetime', low_memory = False)
    # visualize_correlation(df, 'spearman', data_name)
    
    # resampling
    # sum karena total konsumsi power
    df_resampled = df.resample('D').sum()
    visualize_data('d', 'plot', df_resampled, EDA_PATH + 'plot-per-day/', data_name)
    for col_name in df_resampled.columns.to_list():
      visualize_data('d', 'hist', df_resampled, EDA_PATH + 'hist-dist-per-day-' + col_name + '/', data_name, col_name)
      visualize_data('y', 'plot', df_resampled, EDA_PATH + 'plot-per-year-' + col_name + '/', data_name, col_name, years)
      visualize_data('y', 'hist', df_resampled, EDA_PATH + 'hist-dist-per-year-' + col_name + '/', data_name, col_name, years)

      for year in years:
        visualize_data('m', 'hist', df_resampled, EDA_PATH + 'hist-dist-per-month-' + year + '-' + col_name + '/', data_name, col_name, None, year)