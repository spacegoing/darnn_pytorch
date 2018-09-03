# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
# todo: torch must be imported first, otherwise segmentation fault
# do not know why ver: 0.4.0
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def read_data(input_path, debug=True):
  """
    Args:
        input_path (str): directory to nasdaq dataset.

    Returns:
        X (DataFrame): features.
        y (DataFrame): ground truth.

  """
  df = pd.read_csv(input_path, nrows=250 if debug else None)
  X = df.iloc[:, :-1]
  Y = df.iloc[:, -1].to_frame()
  return X, Y


class PreprocDataset:

  def init_dataset(self, X, Y, train_ratio, timesteps, pred_timesteps):
    """
    self.X_train_df, self.Y_train_df, self.Ygt_train_df
    self.X_test_df, self.Y_test_df, self.Ygt_test_df
    """
    # normalize dataset
    self.X_scaler, X, self.Y_scaler, Y = self.normalize_featmat(X, Y)

    # split train test dataset
    rows = X.shape[0]
    train_size = int(rows * train_ratio)
    X_train = X.iloc[:train_size, :]
    X_test = X.iloc[train_size:, :]
    Y_train = Y.iloc[:train_size, :]
    Y_test = Y.iloc[train_size:, :]

    # generate timeseries training data
    self.X_train_df, self.Y_train_df, self.Ygt_train_df = self.gen_dataset(
        X_train, Y_train, timesteps, pred_timesteps)
    self.X_test_df, self.Y_test_df, self.Ygt_test_df = self.gen_dataset(
        X_test, Y_test, timesteps, pred_timesteps)

  def normalize_featmat(self, X_df, Y_df):
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()

    X_scaler.fit(X_df)
    Y_scaler.fit(Y_df)
    return X_scaler, pd.DataFrame(
        X_scaler.transform(X_df)), Y_scaler, pd.DataFrame(
            Y_scaler.transform(Y_df))

  def gen_dataset(self, X_df, Y_df, timesteps, pred_timesteps):
    """
    y_{t+pred_timesteps} = p(y_t,...,y_{timesteps-1}, x_t,...,x_{timesteps-1})

    len = df.shape[0]
    X_lag_df: (len-timesteps-pred_timesteps+1, feat_dim*timesteps)
    Y_lag_df: (len-timesteps-pred_timesteps+1, feat_dim*timesteps)
    Ygt_df: (len-timesteps-pred_timesteps+1, 1)

    test:
      txdf = pd.DataFrame(np.arange(100))
      tydf = pd.DataFrame(np.arange(100,200))
      X_df = txdf
      Y_df = tydf
    """

    def get_lag_df(df, timesteps, pred_timesteps):
      """
      x_{timesteps-1},...,x_t totally `timesteps` x_i

      -pred_timesteps sample won't have Ygt, so filtered out
      """
      df_col = [df.shift(i) for i in range(timesteps - 1, -1, -1)]
      lag_df = pd.concat(df_col, axis=1).iloc[timesteps - 1:-pred_timesteps, :]
      return lag_df

    X_lag_df = get_lag_df(X_df, timesteps, pred_timesteps)
    Y_lag_df = get_lag_df(Y_df, timesteps, pred_timesteps)

    # -pred_timesteps sample won't have Ygt, so filtered out
    Ygt_df = Y_df.shift(-pred_timesteps).iloc[timesteps - 1:-pred_timesteps, :]
    return X_lag_df, Y_lag_df, Ygt_df


class Trainset(Dataset):

  def __init__(self, pre_dataset: PreprocDataset, timesteps):
    super().__init__()
    self.X = pre_dataset.X_train_df
    self.Y = pre_dataset.Y_train_df
    self.Ygt = pre_dataset.Ygt_train_df
    self.timesteps = timesteps

  def __getitem__(self, idx):
    return {
        'X': self.get_np_mat(self.X, idx),
        'Y': self.get_np_mat(self.Y, idx),
        'Ygt': self.Ygt.iat[idx, 0]
    }

  def __len__(self):
    return self.X.shape[0]

  def get_np_mat(self, df, idx):
    '''
    reshape df (1, feat_dim * timesteps) -> (timesteps, feat_dim)
    '''
    return df.iloc[idx].values.reshape(self.timesteps, -1)


class Nas100Dataset:

  def get_data_loader(self, opt):
    # train_ratio = 0.7
    # timesteps = 9  # t, t-1, ... t-8
    # pred_timesteps = 1
    # batchsize = 128
    X, Y = read_data("../nasdaq/nasdaq100_padding.csv", opt.debug)

    train_ratio = opt.train_ratio
    timesteps = opt.timesteps
    pred_timesteps = opt.pred_timesteps
    batchsize = opt.batchsize
    shuffle = opt.shuffle
    num_workers = opt.num_workers
    pin_memory = opt.pin_memory

    pre_dataset = PreprocDataset()
    pre_dataset.init_dataset(X, Y, train_ratio, timesteps, pred_timesteps)

    # recover normalized
    pre_dataset.X_scaler.inverse_transform(
        pre_dataset.X_train_df.values[0].reshape(timesteps, -1))

    train_dataset = Trainset(pre_dataset, timesteps)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory)

    return pre_dataset, train_loader


if __name__ == "__main__":
  train_ratio = 0.7
  timesteps = 9  # t, t-1, ... t-8
  pred_timesteps = 1
  batchsize = 128

  X, Y = read_data("../nasdaq/nasdaq100_padding.csv")
  pre_dataset = PreprocDataset()
  pre_dataset.init_dataset(X, Y, train_ratio, timesteps, pred_timesteps)

  # recover normalized
  pre_dataset.X_scaler.inverse_transform(
      pre_dataset.X_train_df.values[0].reshape(timesteps, -1))

  train_dataset = Trainset(pre_dataset, timesteps)
  train_loader = DataLoader(
      train_dataset,
      batch_size=batchsize,
      shuffle=False,
      num_workers=1,
      pin_memory=True)

  for i, d in enumerate(train_loader):
    if i % 100 == 0:
      print(i)

  # # double check
  # i=-1
  # aa=pre_dataset.X_train_df.iloc[i].values.reshape(timesteps,-1)
  # print((aa==d['X'][i,:,:]).sum())

  # aaa=pre_dataset.X_scaler.inverse_transform(aa)
  # print((aaa[0] == X.iloc[0]).sum())
  # neq = np.where((aaa[0] == X.iloc[0])==False)[0]
  # for i in neq:
  #   print("%f %f" % (aaa[0,i], X.iloc[0,i]))

  # opt.train_ratio = 0.7
  # opt.timesteps = 9
  # opt.pred_timesteps = 1
  # opt.batchsize = 128
  # opt.shuffle = False
  # opt.num_workers = 1
  # opt.pin_memory = True

