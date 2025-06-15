import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter


class HelperFunctions:
    def __init__(self):
        return

    @staticmethod
    def identify_duplicate_questions(
        df: pd.DataFrame,
        dataset_name: str = 'training'
    ) -> None:
        """This function is responsible for displaying data of dataframe

        Args:
            df (pd.DataFrame): dataframe to display data of
        """

        duplicates = round(df['is_duplicate'].mean()*100, 2)
        qids = pd.Series(
            df['qid1'].tolist() + df['qid2'].tolist()
        )
        unique_qids = np.unique(qids)
        multiple_qids = np.sum(qids.value_counts() > 1)

        print(f"Total number of question pairs for {dataset_name}: {len(df)}")
        print(f"Duplicate pairs: {duplicates}%")
        print(
            f"Total number of questions in {dataset_name} data: "
            f"{len(unique_qids)}"
        )
        print(
            f"Number of questions which appear multiple times: {multiple_qids}"
        )

        plt.figure(figsize=(12, 5))
        plt.hist(qids.value_counts(), bins=50)
        plt.yscale('log')
        plt.title('Log-Histogram of question appearance counts')
        plt.xlabel('Number of occurences of question')
        plt.ylabel('Number of questions')
        plt.show()

    @staticmethod
    def text_analysis(
        train_df: pd.DataFrame, test_df: pd.DataFrame, pal: tuple
    ) -> tuple:
        """This function shows plot of train and test dataframes text analysis

        Args:
            train_df (pd.DataFrame): train dataframe
            test_df (pd.DataFrame): test dataframe
        """

        train_qs = pd.Series(
            train_df['question1'].tolist() + train_df['question2'].tolist()
        ).astype(str)
        test_qs = pd.Series(
            test_df['question1'].tolist() + test_df['question2'].tolist()
        ).astype(str)

        dist_train = train_qs.apply(len)
        dist_test = test_qs.apply(len)

        plt.figure(figsize=(15, 10))
        plt.hist(
            dist_train, bins=200,
            range=[0, 200],
            color=pal[2],
            density=True,   # <-- use this!
            label='train'
        )
        plt.hist(
            dist_test,
            bins=200,
            range=[0, 200],
            color=pal[1],
            density=True,   # <-- use this!
            alpha=0.5,
            label='test'
        )

        print(
            f"mean-train: {dist_train.mean():.2f} "
            f"std-train: {dist_train.std():.2f} "
            f"max-train: {dist_train.max()} "
            f"mean-test: {dist_test.mean():.2f} "
            f"std-test: {dist_test.std():.2f} "
            f"max-test: {dist_test.max()}"
        )

        # Return the two Series
        return train_qs, test_qs

    @staticmethod
    def word_count(
        train_qs: pd.Series, test_qs: pd.Series, pal: tuple
    ) -> None:
        dist_train = train_qs.apply(lambda x: len(x.split(' ')))
        dist_test = test_qs.apply(lambda x: len(x.split(' ')))

        plt.figure(figsize=(15, 10))
        plt.hist(
            dist_train,
            bins=50,
            range=[0, 50],
            color=pal[2],
            density=True,
            label='train'
        )
        plt.hist(
            dist_test,
            bins=50,
            range=[0, 50],
            color=pal[1],
            density=True,
            alpha=0.5,
            label='test'
        )
        plt.title(
            'Normalised histogram of word count in questions',
            fontsize=15
        )
        plt.legend()
        plt.xlabel('Number of words', fontsize=15)
        plt.ylabel('Probability', fontsize=15)

        print(
            f" mean-train {dist_train.mean()}"
            f" std-train {dist_train.std()}"
            f" mean-test {dist_test.mean()}"
            f" std-test {dist_test.std()}"
            f" max-train {dist_train.max()}"
            f" max-test {dist_test.max()}"
        )

    @staticmethod
    def semantic_analysis(qs_series: pd.Series) -> None:
        qmarks = np.mean(qs_series.apply(lambda x: '?' in x))
        math = np.mean(qs_series.apply(lambda x: '[math]' in x))
        fullstop = np.mean(qs_series.apply(lambda x: '.' in x))
        capital_first = np.mean(qs_series.apply(lambda x: x.isupper()))
        strings_with_capitals = np.mean(
            qs_series.apply(lambda x: max([y.isupper() for y in x]))
        )
        numbers = np.mean(qs_series.apply(
            lambda x: max([y.isdigit() for y in x]))
        )

        print(f"Questions with question marks: {qmarks * 100:.2f}%")
        print(f"Questions with [math] tags: {math * 100:.2f}%")
        print(f"Questions with full stops: {fullstop * 100:.2f}%")
        print(
            f"Questions with "
            f"capitalised first letters: {capital_first * 100:.2f}%"
        )
        print(
            f"Questions"
            f" with capital letters: {strings_with_capitals * 100: .2f}%"
        )
        print(f"Questions with numbers: {numbers * 100:.2f}%")

    @staticmethod
    def feature_analysis(df: pd.DataFrame):
        stops = set(stopwords.words("english"))

        plt.figure(figsize=(15, 5))

        train_word_match = df.apply(
            lambda row: HelperFunctions.word_match_share(row, stops),
            axis=1,
        )
        plt.hist(train_word_match[df['is_duplicate'] == 0],
                 bins=20, density=True, label='Not Duplicate')
        plt.hist(train_word_match[df['is_duplicate'] == 1],
                 bins=20, density=True, alpha=0.7, label='Duplicate')
        plt.legend()
        plt.title('Label distribution over word_match_share', fontsize=15)
        plt.xlabel('word_match_share', fontsize=15)

        return train_word_match

    @staticmethod
    def word_match_share(row, stops):
        q1words = {}
        q2words = {}

        for word in str(row['question1']).lower().split():
            if word not in stops:
                q1words[word] = 1

        for word in str(row['question2']).lower().split():
            if word not in stops:
                q2words[word] = 1

        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes
            # a few questions that are nothing but stopwords
            return 0

        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

        R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / \
            (len(q1words) + len(q2words))

        return R

    # If a word appears only once, we ignore it completely (likely a typo)
    # Epsilon defines a smoothing constant,
    # which makes the effect of extremely rare words smaller
    @staticmethod
    def get_weight(count: int, eps: int = 1000, min_count: int = 2):
        """
        Get word weight in the text

        Args:
            count (int): Number of word appearances in text.
            eps (int, optional): smoothing constant.
            min_count (int, optional): Minimal word count in text.
                                        Defaults to 2.

        Returns:
            word weight
        """
        if count < min_count:
            return 0
        else:
            return 1 / (count + eps)

    @staticmethod
    def show_common_words_and_weights(qs_series: pd.Series):
        """
        Show words weights is text

        Args:
            qs_series (pd.Series): questions series
        """

        words = (" ".join(qs_series)).lower().split()
        counts = Counter(words)
        weights = {
            word: HelperFunctions.get_weight(count)
            for word, count in counts.items()
        }

        print('Most common words and weights: \n')
        print(sorted(weights.items(),
              key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
        print('\nLeast common words and weights: ')
        print(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])

        return weights

    @staticmethod
    def tfidf_word_match_share(row, weights):
        q1words = {}
        q2words = {}
        stops = set(stopwords.words("english"))

        for word in str(row['question1']).lower().split():
            if word not in stops:
                q1words[word] = 1
        for word in str(row['question2']).lower().split():
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions
            # that are nothing but stopwords
            return 0

        shared_weights = [
            weights.get(w, 0) for w in q1words.keys()
            if w in q2words] + [weights.get(w, 0) for w in q2words.keys()
                                if w in q1words
                                ]
        total_weights = [weights.get(w, 0) for w in q1words] + \
            [weights.get(w, 0) for w in q2words]

        R = np.sum(shared_weights) / np.sum(total_weights)
        return R

    @staticmethod
    def show_tlidf_match_share(df: pd.DataFrame, weights):
        tfidf_train_word_match = df.apply(
            lambda row: HelperFunctions.tfidf_word_match_share(row, weights),
            axis=1
        )
        plt.figure(figsize=(15, 5))
        plt.hist(tfidf_train_word_match[df['is_duplicate'] == 0].fillna(0),
                 bins=20, density=True, label='Not Duplicate')
        plt.hist(tfidf_train_word_match[df['is_duplicate'] == 1].fillna(0),
                 bins=20, density=True, alpha=0.7, label='Duplicate')
        plt.legend()
        plt.title(
            'Label distribution over tfidf_word_match_share',
            fontsize=15
        )
        plt.xlabel('word_match_share', fontsize=15)
        plt.show()

        return tfidf_train_word_match
