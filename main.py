import matplotlib.pyplot as plt
import numpy as np
import sklearn.feature_extraction.text as txt
import sklearn.naive_bayes as nb
from sklearn import metrics
import itertools
import pprint

from read_data import game_tags, game_tag_count, read_test_dataset, read_train_dataset, read_games, \
    get_game_tags_in_test

selected_game_tags = ["Action RPG", "Card Game"]

ds_games = read_games()
ds_train = read_train_dataset(selected_game_tags)
ds_test = read_test_dataset(selected_game_tags)
get_game_tags_in_test()
n_train = len(ds_train)
n_test = len(ds_test)


def print_game_tags():
    print("Game tags found in dataset:")
    print(sorted(game_tags))
    pprint.pprint(game_tag_count)


print_game_tags()

n_words = np.zeros((25, 1))
train_accuracy = np.zeros((25, 1))
test_accuracy = np.zeros((25, 1))
alpha = 1

# Searching the optimal keywords list
for i in range(25):
    min_df = (i + 1) * 0.01
    vectorize_data = txt.CountVectorizer(stop_words='english', min_df=min_df)
    DT_train = vectorize_data.fit_transform(ds_train["user_review"])
    n_words[i] = DT_train.shape[1]
    clf_MNB = nb.MultinomialNB(alpha=alpha)
    clf_MNB.fit(DT_train, ds_train["tag"])
    train_accuracy[i] = clf_MNB.score(DT_train, ds_train["tag"])
    DT_test = vectorize_data.transform(ds_test["user_review"])
    test_accuracy[i] = clf_MNB.score(DT_test, ds_test["tag"])

# Accuracy plot
fig = plt.figure(1)
plt.title('Accuracy plot for Naiv Bayes classifier')
plt.xlabel('Number of words')
plt.ylabel('Accuracy')
plt.plot(n_words, train_accuracy, c='blue', label='training')
plt.plot(n_words, test_accuracy, c='red', label='test')
plt.legend(loc="lower right")
plt.show()

# Vectorizing using the optimal parameter
min_df = 0.13
vectorizer = txt.CountVectorizer(stop_words='english', min_df=min_df)
DT_train = vectorizer.fit_transform(ds_train["user_review"])
vocabulary_list = vectorizer.get_feature_names_out()
vocabulary = np.asarray(vocabulary_list)  # vocabulary in 1D array
print()
print(vocabulary)
print()

# Fitting the final model
clf_MNB = nb.MultinomialNB(alpha=alpha)
clf_MNB.fit(DT_train, ds_train["tag"])
ds_train_pred = clf_MNB.predict(DT_train)

# Performance metrics
print(metrics.classification_report(ds_train["tag"], ds_train_pred, target_names=selected_game_tags))

train_conf_mat = metrics.confusion_matrix(ds_train["tag"], ds_train_pred)


# Visualisation of the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Plot non-normalized confusion matrix
plt.figure(2)
plot_confusion_matrix(train_conf_mat, classes=selected_game_tags, title='Confusion matrix for training set')

# Transforming the test dataset
DT_test = vectorizer.transform(ds_test["user_review"])
ds_test_pred = clf_MNB.predict(DT_test)

print(metrics.classification_report(ds_test["tag"], ds_test_pred, target_names=selected_game_tags))

test_conf_mat = metrics.confusion_matrix(ds_test["tag"], ds_test_pred)
test_proba = clf_MNB.predict_proba(DT_test)

# Plot non-normalized confusion matrix
plt.figure(3)
plot_confusion_matrix(test_conf_mat, classes=selected_game_tags, title='Confusion matrix for test set')

# Computing false and true positive rate and AUC
fpr_train, tpr_train, _ = metrics.roc_curve(ds_train["tag"], ds_train_pred)
roc_auc_train = metrics.auc(fpr_train, tpr_train)
fpr_test, tpr_test, _ = metrics.roc_curve(ds_test["tag"], ds_test_pred)
roc_auc_test = metrics.auc(fpr_test, tpr_test)

plt.figure(4)
lw = 4
plt.plot(fpr_train, tpr_train, color='blue', lw=lw, label='train (AUC = %0.2f)' % roc_auc_train)
plt.plot(fpr_test, tpr_test, color='red', lw=lw, label='test (AUC = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()

