from read_data import read_games, read_train_dataset, read_test_dataset

games = read_games()
train_dataset = read_train_dataset()
test_dataset = read_test_dataset()


def main():
    print(games)


if __name__ == '__main__':
    main()
