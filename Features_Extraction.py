from textstat import flesch_reading_ease, gunning_fog, automated_readability_index, coleman_liau_index, syllable_count
from textblob import TextBlob
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk import pos_tag

df = pd.read_csv('SEEDNet_Dataset')

# Function to calculate polarity and subjectivity scores
def calculate_sentiment_scores(text):
    blob = TextBlob(text)
    polarity_score = blob.sentiment.polarity
    subjectivity_score = blob.sentiment.subjectivity
    return polarity_score, subjectivity_score


df[['Polarity_Score', 'Subjectivity_Score']] = df['tweet'].apply(lambda x: pd.Series(calculate_sentiment_scores(x)))

# ---------------------------------------------------------------------------------------------------------------------

# Function to count negative words
def Negative_Words_Count(text, threshold=-0.5):
    neg_word_list = []
    tokenized_words = word_tokenize(text)

    for word in tokenized_words:
        testimonial = TextBlob(word)
        if testimonial.sentiment.polarity <= threshold:
            neg_word_list.append(word)

    return len(neg_word_list)


# Function to count positive words
def Positive_Words_Count(text, threshold=0.5):
    pos_word_list = []
    tokenized_words = word_tokenize(text)

    for word in tokenized_words:
        testimonial = TextBlob(word)
        if testimonial.sentiment.polarity >= threshold:
            pos_word_list.append(word)

    return len(pos_word_list)


df['Negative_Words_Count'] = df['tweet'].apply(lambda x: Negative_Words_Count(x))
df['Positive_Words_Count'] = df['tweet'].apply(lambda x: Positive_Words_Count(x))

# ---------------------------------------------------------------------------------------------------------------------

# Load the AVAD dataset from the CSV file
avad_df = pd.read_csv('AVAD_Dataset_Extracting_features.csv')

# Function to calculate average AVAD scores for a tweet
def calculate_average_avad(tweet):
    words = tweet.lower().split()

    total_valence = 0
    total_arousal = 0
    total_dominance = 0
    word_count = 0

    for word in words:
        word_info = avad_df[avad_df['Word'] == word]
        if not word_info.empty:
            total_valence += word_info['valence'].values[0]
            total_arousal += word_info['arousal'].values[0]
            total_dominance += word_info['dominance'].values[0]
            word_count += 1

    average_valence = total_valence / word_count if word_count > 0 else 0
    average_arousal = total_arousal / word_count if word_count > 0 else 0
    average_dominance = total_dominance / word_count if word_count > 0 else 0

    return average_valence, average_arousal, average_dominance


# Apply the function to the 'tweet' column and create a new column for average AVAD scores
df[['Average_Valence', 'Average_Arousal', 'Average_Dominance']] = df['tweet'].apply(
    lambda x: pd.Series(calculate_average_avad(x)))

df['Average_AVAD'] = df[['Average_Valence', 'Average_Arousal', 'Average_Dominance']].mean(axis=1)


def calculate_average_anew_scores(text):
    words = word_tokenize(text.lower())

    total_valence = 0
    total_arousal = 0
    total_dominance = 0
    word_count = 0

    for word in words:
        word_info = avad_df[avad_df['Word'] == word]
        if not word_info.empty:
            total_valence += word_info['valence'].values[0]
            total_arousal += word_info['arousal'].values[0]
            total_dominance += word_info['dominance'].values[0]
            word_count += 1

    average_valence = total_valence / word_count if word_count > 0 else 0
    average_arousal = total_arousal / word_count if word_count > 0 else 0
    average_dominance = total_dominance / word_count if word_count > 0 else 0

    return average_valence, average_arousal, average_dominance


df[['Average_Valence', 'Average_Arousal', 'Average_Dominance']] = df['tweet'].apply(
    lambda x: pd.Series(calculate_average_anew_scores(x)))


# ---------------------------------------------------------------------------------------------------------------------


def count_parts_of_speech(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)

    noun_count = len([word for word, tag in tagged_words if tag.startswith('N')])
    adjective_count = len([word for word, tag in tagged_words if tag.startswith('J')])
    adverb_count = len([word for word, tag in tagged_words if tag.startswith('R')])
    verb_count = len([word for word, tag in tagged_words if tag.startswith('V')])

    return noun_count, adjective_count, adverb_count, verb_count


df[['Noun_Count', 'Adjective_Count', 'Adverb_Count', 'Verb_Count']] = df['tweet'].apply(
    lambda x: pd.Series(count_parts_of_speech(x)))


# ---------------------------------------------------------------------------------------------------------------------


def calculate_flesch_reading_ease(text):
    return flesch_reading_ease(text)


def calculate_gunning_fog_index(text):
    return gunning_fog(text)


def calculate_automated_readability_index(text):
    return automated_readability_index(text)


def calculate_coleman_liau_index(text):
    return coleman_liau_index(text)


def calculate_syllable_count(text):
    return syllable_count(text)


# Apply the functions to the DataFrame
df['Flesch_Reading_Ease'] = df['tweet'].apply(calculate_flesch_reading_ease)
df['Gunning_Fog_Index'] = df['tweet'].apply(calculate_gunning_fog_index)
df['Automated_Readability_Index'] = df['tweet'].apply(calculate_automated_readability_index)
df['Coleman_Liau_Index'] = df['tweet'].apply(calculate_coleman_liau_index)
df['Syllable_Count'] = df['tweet'].apply(calculate_syllable_count)


# ---------------------------------------------------------------------------------------------------------------------


lexicon_df = pd.read_csv('lexicon_dict_EDBase.csv')

word_dictionary = dict(zip(lexicon_df['words'], range(len(lexicon_df))))


# Function to extract words from a tweet that are in the dictionary
def extract_words_in_dictionary(tweet, dictionary):
    tweet_words = tweet.lower().split()
    return [word for word in tweet_words if word in dictionary]


# Add a new column 'Extracted_Words' to tweets_df with the extracted words
df['lexicon_Words'] = df['tweet'].apply(lambda x: extract_words_in_dictionary(x, word_dictionary))


# Function to count the extracted words and return the count
def count_extracted_words(extracted_words):
    return len(extracted_words)


# Add a new column 'Extracted_Word_Count' to count the extracted words
df['lexicon_Count'] = df['lexicon_Words'].apply(count_extracted_words)


def count_to_binary(count):
    return 1 if count >= 1 else 0


# Add a new column 'Count_Binary' based on the extracted word count
df['Count_Binary'] = df['lexicon_Count'].apply(count_to_binary)


df.to_csv('SEEDNet_Dataset_features.csv')
