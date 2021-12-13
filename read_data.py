import io

import pandas as pd

from settings import *


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

        game = Game()
        game.title = df.index[i]
        game.developer = df['developer'][i]
        game.publisher = df['publisher'][i]
        game.tags = tags
        game.overview = df['overview'][i]

        games.append(game)
        for tag in tags:
            game_tags.add(tag)

        df.at[df.index[i], 'tags'] = tags

    return df


def read_review_dataset(filename):
    df = pd.read_csv(filepath_or_buffer="data/" + filename)
    return df


def read_train_dataset():
    """Reads the training dataset"""
    df = read_review_dataset("train.csv")

    for i in range(0, len(df.index)):
        user_review = UserReview()
        user_review.id = df['review_id'][i]
        user_review.title = df['title'][i]
        user_review.year = df['year'][i]
        user_review.user_review = df['user_review'][i]
        user_review.suggested = df['user_suggestion'][i] == '1'

        training_dataset.append(user_review)

    return df


def read_test_dataset():
    """Reads the test dataset"""
    df = read_review_dataset("train.csv")

    for i in range(0, len(df.index)):
        user_review = UserReview()
        user_review.id = df['review_id'][i]
        user_review.title = df['title'][i]
        user_review.year = df['year'][i]
        user_review.user_review = df['user_review'][i]

        training_dataset.append(user_review)

    return df


ds_games = read_games()
print(ds_games["developer"])
ds_train = read_train_dataset()
ds_test = read_test_dataset()
