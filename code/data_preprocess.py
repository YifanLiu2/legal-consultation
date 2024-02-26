import os
import argparse
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def tokenize_and_stem(text_series):
    """Tokenize and stem a pandas Series of text."""
    return text_series.apply(lambda x: [stemmer.stem(word.lower()) for word in word_tokenize(x)])


def most_freq(tokens_series, top_n):
    """Find the most frequent tokens in a Series of token lists."""
    all_tokens = np.concatenate(tokens_series.values)
    freq_dist = nltk.FreqDist(all_tokens)
    return set([word for word, _ in freq_dist.most_common(top_n)])


def initialize_word_sets(filepath):
    """Initialize the attorney and client's frequent word list."""
    try:
        att_label = pd.read_csv(filepath)[:200]
    except FileNotFoundError as e:
        print(f"Error reading files: {e}")
        return
    att_post = att_label[att_label.true_label == 1]['PostText']
    cli_post = att_label[att_label.true_label == 0]['PostText']

    att_tokens = tokenize_and_stem(att_post)
    cli_tokens = tokenize_and_stem(cli_post)

    att_set_30 = most_freq(att_tokens, 30)
    att_set_50 = most_freq(att_tokens, 50)
    cli_set_30 = most_freq(cli_tokens, 30)
    cli_set_50 = most_freq(cli_tokens, 50)

    att_words = list(att_set_50 - cli_set_30)
    cli_words = list(cli_set_50 - att_set_30)

    return att_words, cli_words


def count_att_cli_words(tokens):
    """Determines the label based on the count of attorney and client words in tokens. Return 1 if attorney words are more frequent."""
    attorney_count = sum(token in ATT_WORDS for token in tokens)
    client_count = sum(token in CLI_WORDS for token in tokens)
    if attorney_count > client_count:
        return 1
    elif attorney_count < client_count:
        return 0
    else:
        return int(len(tokens) > 10)


def convert_format(d):
    """Convert invalid date format."""
    try:
        return f"{d[4:6]}/{d[7:9]}/{d[1:3]} {d[10:-1]}"
    except TypeError:
        return np.nan


def clean_dates(df, column):
    """Convert the date to datetime object."""
    df[column] = df[column].apply(convert_format)
    df[column] = pd.to_datetime(df[column], errors='coerce')


def prepare_datasets(filepath, nrows):
    """
    Reads post, question, and attorney time entry datasets from a specified filepath, merge them together, and
    cleans date columns.

    Args:
        filepath (str): The directory path where the CSV files are located.
        nrows (int | None): The number of rows to load from question dataset.

    Returns:
        tuple: Merged posts DataFrame.

    Raises:
        FileNotFoundError: If any of the CSV files are not found at the specified filepath.
    """
    # Read data
    try:
        question = pd.read_csv(os.path.join(filepath, "questions.csv"), index_col=0, nrows=nrows)
        attorney_time = pd.read_csv(os.path.join(filepath, "attorneytimeentries.csv"), index_col=0, usecols=["AttorneyUno", "EnteredOnUtc"])
    except FileNotFoundError as e:
        print(f"Error reading files: {e}")
        return
    question = question[["StateAbbr", "QuestionUno", "CategoryUno", "TakenByAttorneyUno"]]
    question = question[question['TakenByAttorneyUno'].notna()]
    question = question[question['StateAbbr'] == "IN"]

    chunks = []
    for chunk in pd.read_csv(os.path.join(filepath, "questionposts.csv"), index_col=0, chunksize=10000, usecols=["QuestionUno", "PostText", "CreatedUtc"]):
        filtered_chunk = chunk.merge(question, on='QuestionUno', how='inner')
        chunks.append(filtered_chunk)
    post = pd.concat(chunks, ignore_index=True)
    
    post = post.merge(attorney_time, left_on='TakenByAttorneyUno', right_on='AttorneyUno', how='left')

    # Clean date
    clean_dates(post, "CreatedUtc")
    clean_dates(post, "EnteredOnUtc")

    return post


def preprocess_text(text_series):
    """
    Preprocess texts with following steps:
    # TODO: Updates this method
    Args:
        text_series (pd.Series): A Series of text strings to preprocess.

    Returns:
        pd.Series: A Series where each text string is replaced by a single string with stemmed and lowercased words.
    """
    return text_series.apply(lambda text: ' '.join([stemmer.stem(word.lower()) for word in word_tokenize(text)]))


def label_posts(post):
    """
    Identifies post as attorney or client based on three criteria:
    1. Time-based matching with attorney time entries.
    2. Identification of the first post for each question.
    3. Comparison of attorney and client word counts.

    Args:
        post (pd.DataFrame): DataFrame containing posts.

    Updates:
        Updates the 'WhosePost' column in the 'post' DataFrame in place.
    """
    # Time-based matching with attorney time entries
    time_diff = post['EnteredOnUtc'] - post['CreatedUtc']
    within_time_frame = time_diff.abs() <= pd.Timedelta(minutes=1)
    post['WhosePost'] = np.where(within_time_frame, 1, np.nan)

    # Identification of the first post for each question
    first_posts = post.groupby('QuestionUno')['CreatedUtc'].idxmin()
    post.loc[post.index.isin(first_posts), 'WhosePost'] = 0

    # Comparison of attorney and client word counts
    unlabeled = post['WhosePost'].isna()
    post.loc[unlabeled, 'WhosePost'] = post.loc[unlabeled, 'WhosePost'].apply(count_att_cli_words)


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    nrows = args.nrows

    # Init gloable var
    global ATT_WORDS, CLI_WORDS
    ATT_WORDS, CLI_WORDS = initialize_word_sets(os.path.join(input_path, "att_label.csv"))

    # Prepare datasets
    print("Prepare Data\n")
    post= prepare_datasets(input_path, nrows=nrows)

    # Clean text
    print("Clean Data\n")
    post["CleanedText"] = preprocess_text(post["PostText"])

    # Label posts
    print("Label Posts\n")
    label_posts(post)

    # Save new post DataFrame without index
    post = post[["QuestionUno", "CleanedText", "WhosePost", "CategoryUno"]]
    post.to_csv(os.path.join(output_path, "cleaned_posts.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and label post data.")
    parser.add_argument('-i', '--input_path', type=str, help='Path to the input directory containing the data files.')
    parser.add_argument('-o', '--output_path', type=str, help='Path to the output directory where the processed files will be saved.')
    parser.add_argument('-n', '--nrows', type=int, default=None, help='Optional: Number of rows to read from question dataset.')

    args = parser.parse_args()
    main(args)



