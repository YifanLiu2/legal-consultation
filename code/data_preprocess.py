import os
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def most_freq(tokens_series, top_n):
    """Find the most frequent tokens in a Series of token lists."""
    all_tokens = np.concatenate(tokens_series.values)
    freq_dist = nltk.FreqDist(all_tokens)
    return set([word for word, _ in freq_dist.most_common(top_n)])


def initialize_word_sets(filepath):
    """Initialize the attorney and client's frequent word set."""
    try:
        att_label = pd.read_csv(filepath)[:200]
    except FileNotFoundError as e:
        print(f"Error reading files: {e}")
        return
    att_post = att_label[att_label.true_label == 1]['PostText']
    cli_post = att_label[att_label.true_label == 0]['PostText']

    att_tokens = preprocess_text(att_post)
    cli_tokens = preprocess_text(cli_post)

    att_set_30 = most_freq(att_tokens, 30)
    att_set_50 = most_freq(att_tokens, 50)
    cli_set_30 = most_freq(cli_tokens, 30)
    cli_set_50 = most_freq(cli_tokens, 50)

    att_words = Counter(att_set_50 - cli_set_30)
    cli_words = Counter(cli_set_50 - att_set_30)

    return att_words, cli_words


def count_att_cli_words(text):
    """Determines the label based on the count of attorney and client words in tokens. Return 1 if attorney words are more frequent."""
    word_counter = Counter(text)
    attorney_count = len(word_counter & ATT_WORDS)
    client_count = len(word_counter & CLI_WORDS)
    if attorney_count > client_count:
        return 1
    elif attorney_count < client_count:
        return 0
    else:
        return int(len(word_counter) > 10)


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


def prepare_datasets(filepath: str, nrows: int | None) -> tuple[pd.DataFrame]:
    """
    Reads post, question, and attorney time entry datasets from a specified filepath;
    merge question and post dataframe;
    cleans date columns.

    Args:
        filepath (str): The directory path where the CSV files are located.
        nrows (int | None): The number of rows to load from question dataset, positive integer.

    Returns:
        tuple: Merged posts DataFrame and attorney time DataFrame.

    Raises:
        FileNotFoundError: If any of the CSV files are not found at the specified filepath.
    """
    # Read data
    try:
        question = pd.read_csv(os.path.join(filepath, "questions.csv"), index_col=0)
        if nrows:
            question = question.sample(n=5)
        attorney_time = pd.read_csv(os.path.join(filepath, "attorneytimeentries.csv"), index_col=0, usecols=["AttorneyUno", "EnteredOnUtc"])
    except FileNotFoundError as e:
        print(f"Error reading files: {e}")
        return
    question = question[["StateAbbr", "QuestionUno", "Category", "TakenByAttorneyUno"]]
    question = question[question['TakenByAttorneyUno'].notna()]

    chunks = []
    try:
        for chunk in pd.read_csv(os.path.join(filepath, "questionposts.csv"), chunksize=10000, usecols=["Id", "QuestionUno", "PostText", "CreatedUtc"]):
            filtered_chunk = chunk.merge(question, on='QuestionUno', how='inner')
            chunks.append(filtered_chunk)
        post = pd.concat(chunks, ignore_index=True)
    except FileNotFoundError as e:
        print(f"Error reading files: {e}")
        return

    # Clean date
    clean_dates(post, "CreatedUtc")
    clean_dates(attorney_time, "EnteredOnUtc")

    return post, attorney_time


def preprocess_text(text_series: pd.Series) -> pd.Series:
    """
    Preprocess texts with following steps:
    This function keeps tokens in list format for each post.

    Args:
        text_series (pd.Series): A Series of text strings to preprocess.

    Returns:
        pd.Series: A Series where each text string is replaced by a list of stemmed and lowercased words.
    """
    text_series = text_series.astype(str)
    tokenized_docs = []
    for doc in tqdm(nlp.pipe(text_series, n_process=-1), total=len(text_series)):
        tokens = [token.lemma_.lower() for token in doc if not token.is_space]
        tokenized_docs.append(tokens)
    return pd.Series(tokenized_docs)


def label_posts(post: pd.DataFrame, attorney_time: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies post as attorney or client based on three criteria:
    1. Time-based matching with attorney time entries.
    2. Identification of the first post for each question.
    3. Comparison of attorney and client word counts.

    Args:
        post (pd.DataFrame): DataFrame containing posts.
        attorney_time (pd.DataFrame): DataFrame containing attorney time entries.

    Returns:
        pd.DataFrame: a new post dataframe with WhosePost label.
    """
    # Time-based matching with attorney time entries
    merged = post[['Id', 'TakenByAttorneyUno', 'CreatedUtc']].merge(attorney_time, left_on='TakenByAttorneyUno', right_on='AttorneyUno', how='left')
    merged['TimeDiff'] = (merged['EnteredOnUtc'] - merged['CreatedUtc']).abs()
    min_time_diff = merged.groupby(['Id'])['TimeDiff'].min().reset_index()
    min_time_diff['WhosePost'] = np.where(min_time_diff['TimeDiff'] <= pd.Timedelta(minutes=1), 1, np.nan)
    post = post.merge(min_time_diff[['Id', 'WhosePost']], on='Id', how='left')

    # Identification of the first post for each question
    first_posts = post.groupby('QuestionUno')['CreatedUtc'].idxmin()
    post.loc[post.index.isin(first_posts), 'WhosePost'] = 0

    # Comparison of attorney and client word counts
    unlabeled = post['WhosePost'].isna()
    post.loc[unlabeled, 'WhosePost'] = post.loc[unlabeled, 'CleanedText'].apply(count_att_cli_words)

    # Join the token list back to text
    post['CleanedText'] = post['CleanedText'].apply(lambda d: " ".join(token for token in d))

    return post

def main(args):
    input_path = args.input_path
    output_path = args.output_path
    nrows = args.nrows

    # Init gloable var
    global ATT_WORDS, CLI_WORDS
    ATT_WORDS, CLI_WORDS = initialize_word_sets(os.path.join(input_path, "att_label.csv"))

    # Prepare datasets
    print("Prepare Data\n")
    post, attorney_time = prepare_datasets(input_path, nrows=nrows)

    # Clean text
    print("Clean Data\n")
    post["CleanedText"] = preprocess_text(post["PostText"])

    # Label posts
    print("Label Posts\n")
    post = label_posts(post, attorney_time)

    # Save new post DataFrame without index
    post = post[["Id", "QuestionUno", "Category", "CleanedText", "WhosePost"]]
    post.to_csv(os.path.join(output_path, "cleaned_posts.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and label post data.")
    parser.add_argument('-i', '--input_path', type=str, help='Path to the input directory containing the data files.')
    parser.add_argument('-o', '--output_path', type=str, help='Path to the output directory where the processed files will be saved.')
    parser.add_argument('-n', '--nrows', type=int, default=None, help='Optional: Number of rows to read from question dataset.')

    args = parser.parse_args()
    main(args)
