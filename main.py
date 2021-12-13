import numpy as np

from read_data import ds_games, ds_train, ds_test

n_train = len(ds_train)
n_test = len(ds_test)

n_words = np.zeros((25, 1))
train_accuracy = np.zeros((25, 1))
test_accuracy = np.zeros((25, 1))
alpha = 1


def main():
    print(ds_games)


if __name__ == '__main__':
    main()
