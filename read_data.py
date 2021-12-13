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


def read_review_dataset(filename, selected_game_tags):
    df = pd.read_csv(filepath_or_buffer="data/" + filename)
    tags = []

    for i in range(0, len(df.index)):
        game_title = df["title"][i]

        found_game_tag = -1
        for game in games:
            if game.title == game_title:
                for index in range(0, len(selected_game_tags)):
                    if selected_game_tags[index] in game.tags:
                        found_game_tag = index
                        break

        tags.append(found_game_tag)

    df["tag"] = tags
    filtered_df = df[df["tag"] != -1]

    return filtered_df


def read_train_dataset(selected_game_tags):
    """Reads the training dataset"""
    df = read_review_dataset("train.csv", selected_game_tags)

    for index, row in df.iterrows():
        user_review = UserReview()
        user_review.id = index
        user_review.title = row['title']
        user_review.year = row['year']
        user_review.user_review = row['user_review']
        user_review.suggested = row['user_suggestion'] == '1'

        training_dataset.append(user_review)

    return df


def read_test_dataset(selected_game_tags):
    """Reads the test dataset"""
    df = read_review_dataset("train.csv", selected_game_tags)

    for index, row in df.iterrows():
        user_review = UserReview()
        user_review.id = index
        user_review.title = row['title']
        user_review.year = row['year']
        user_review.user_review = row['user_review']
        user_review.suggested = row['user_suggestion'] == '1'

        training_dataset.append(user_review)

    return df
