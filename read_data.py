import io

import pandas as pd


def read_games():
    """Reads the game descriptions"""
    df = pd.read_csv(filepath_or_buffer="data/game_overview.csv", index_col=0)

    for i in range(0, len(df.index)):
        tags_str = df['tags'][i]
        tags_str = tags_str[1:len(tags_str) - 1]

        df_tags = pd.read_csv(io.StringIO(tags_str), header=None, quotechar="'")
        tags = df_tags.values.flatten().tolist()

        for j in range(0, len(tags)):
            tags[j] = tags[j].replace(" '", "").replace("'", "")

        df.at[df.index[i], 'tags'] = tags

    return df


def read_review_dataset(filename):
    df = pd.read_csv(filepath_or_buffer="data/" + filename, index_col=0)
    return df


def read_train_dataset():
    """Reads the training dataset"""
    return read_review_dataset("train.csv")


def read_test_dataset():
    """Reads the test dataset"""
    return read_review_dataset("test.csv")
