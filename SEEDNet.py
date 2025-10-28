import pandas as pd
import re
import networkx as nx
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch_geometric.data import Data
import warnings
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")
from collections import Counter
import math

# Read csv file
df = pd.read_csv("SEEDNet_Dataset_features.csv")

# Mapping of label strings to numeric values
label_mapping = {
    'High': 3,
    'Moderate': 2,
    'Mild': 1,
    'None': 0
}

# Replace the label strings with numeric values using the label_mapping
df['label'] = df['label'].replace(label_mapping)

df = df.sample(frac=1, random_state=42)
df.rename(columns={' tweet': 'tweet'}, inplace=True)
df.reset_index(drop=True, inplace=True)

# randomize the rows
df = df.sample(frac=1, random_state=42)


def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    string = emoji_pattern.sub(r'', string)

    return string.strip().lower()


df['clean_tweet'] = df['tweet'].apply(lambda x: clean_string(x))

# Remove rows with duplicate tweets
df = df.drop_duplicates(subset='tweet')

dictionary = pd.read_csv("lexicon_dict_EDBase.csv")


def extract_lexicon_words(tweet, dictionary):
    words = tweet.split()  # Split the tweet into individual words
    common_words = [word for word in words if word in dictionary.values]  # Find the common words with the dictionary
    return common_words


df['lexicon_words'] = df['clean_tweet'].apply(lambda x: extract_lexicon_words(x, dictionary))

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


# Define a function to tokenize and encode text data
def encode_text(text):
    # Tokenize the input text and generate BERT embeddings
    encodings = tokenizer.encode_plus(text, truncation=True, padding='max_length',
                                      max_length=128)  # length is fix at 128, truncate words more than 128 and add 0 padding for less than 128

    # Convert the input_ids to a tensor and add a batch dimension
    input_ids = torch.tensor(encodings['input_ids']).unsqueeze(0)

    # Convert the attention_mask to a tensor and add a batch dimension
    attention_mask = torch.tensor(encodings['attention_mask']).unsqueeze(0)

    # Generate BERT embeddings using the specified input_ids and attention_mask
    bert_embeddings = bert_model(torch.tensor(input_ids), attention_mask=torch.tensor(attention_mask))[
        1].detach().numpy()

    # Return the processed input_ids, attention_mask, and BERT embeddings
    return input_ids, attention_mask, bert_embeddings


# Apply the function to create 'input_ids' and 'attention_mask' columns
df[['input_ids', 'attention_mask', 'bert_embeddings']] = df['clean_tweet'].apply(lambda x: pd.Series(encode_text(x)))

# Adding Ids
df['ids'] = range(len(df))

"""Creating Graph"""

columns_to_extract = ['Automated_readability_index', 'polarity', 'subjectivity', 'positive_word', 'negative_word',
                      'C_NOUN', 'C_verb', 'C_adj', 'C_adv', 'Syllable_count', 'Coleman_liau_index', 'Gunning_fog',
                      'Flesch_reading_ease', 'Average_VAD',
                      'Domi0ce', 'Arousal', 'Valence', 'ED_related', 'count_eating']
# Extract the columns to be scaled
data_to_scale = df[columns_to_extract]

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler on the data and transform the data
scaled_data = scaler.fit_transform(data_to_scale)

# Replace the original columns with the scaled values
df[columns_to_extract] = scaled_data

# Split the data into training (80%) and testing (20%) sets using stratified sampling
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Initiate graph
graph = nx.Graph()

# Create a dictionary to store attributes for each ID
attributes_dict = {}

# Iterate through the rows of the DataFrame and create a dictionary
for _, row in train_df.iterrows():
    tweet = row['clean_tweet']
    attributes = {
        'Automated_readability_index': row['Automated_readability_index'],
        'polarity': row['polarity'],
        'subjectivity': row['subjectivity'],
        'positive_word': row['positive_word'],
        'negative_word': row['negative_word'],
        'C_NOUN': row['C_NOUN'],
        'C_verb': row['C_verb'],
        'C_adj': row['C_adj'],
        'C_adv': row['C_adv'],
        'Syllable_count': row['Syllable_count'],
        'Coleman_liau_index': row['Coleman_liau_index'],
        'Gunning_fog': row['Gunning_fog'],
        'Flesch_reading_ease': row['Flesch_reading_ease'],
        'Average_VAD': row['Average_VAD'],
        'Domi0ce': row['Domi0ce'],
        'Arousal': row['Arousal'],
        'Valence': row['Valence'],
        'ED_related': row['ED_related'],
        'count_eating': row['count_eating'],
        'label': row['label'],
        'bert_embeddings': row['bert_embeddings']
    }
    attributes_dict[tweet] = attributes

# Add nodes to the graph with attributes
for tweet, attributes in attributes_dict.items():
    # add each node with all the features
    graph.add_node(tweet, **attributes, node_type="tweet")

# Establish edges between nodes based on common lexicon words
for idx1, row1 in train_df.iterrows():
    lex_words_node1 = set(row1['lexicon_words'])
    tweet1_id = row1['clean_tweet']

    for idx2, row2 in train_df.iterrows():
        # Skip the same row (i.e., comparing a tweet with itself)
        if idx1 == idx2:
            continue

        lex_words_node2 = set(row2['lexicon_words'])
        tweet2_id = row2['clean_tweet']

        # Check if both tweets have at least one common lexicon word and avoid adding duplicate edges
        if len(set(lex_words_node1).intersection(set(lex_words_node2))) >= 1:
            graph.add_edge(tweet1_id, tweet2_id)

print(f"Train Graph data: {graph}")

labels = train_df.label.values
labels = torch.tensor(labels)

# Create a mapping from node names to unique integer indices
node_to_index = {node: i for i, node in enumerate(graph.nodes())}

# Convert edge list to list of tuples using the mapping
edge_list = [(node_to_index[edge[0]], node_to_index[edge[1]]) for edge in graph.edges()]

# Convert edge list to tensor
edge_index = torch.tensor(edge_list).t().contiguous()

node_features = []

for node in graph.nodes():
    scalar_features = [
        graph.nodes[node]['Automated_readability_index'],
        graph.nodes[node]['polarity'],
        graph.nodes[node]['subjectivity'],
        graph.nodes[node]['positive_word'],
        graph.nodes[node]['negative_word'],
        graph.nodes[node]['C_NOUN'],
        graph.nodes[node]['C_verb'],
        graph.nodes[node]['C_adj'],
        graph.nodes[node]['C_adv'],
        graph.nodes[node]['Syllable_count'],
        graph.nodes[node]['Coleman_liau_index'],
        graph.nodes[node]['Gunning_fog'],
        graph.nodes[node]['Flesch_reading_ease'],
        graph.nodes[node]['Average_VAD'],
        graph.nodes[node]['Domi0ce'],
        graph.nodes[node]['Arousal'],
        graph.nodes[node]['Valence'],
        graph.nodes[node]['ED_related'],
        graph.nodes[node]['count_eating']
    ]

    # Convert the 'bert_embeddings' to a flat array
    flattened_bert_embeddings = graph.nodes[node]['bert_embeddings'].flatten()

    # Combine scalar features and flattened 'bert_embeddings'
    combined_features = scalar_features + flattened_bert_embeddings.tolist()

    # Convert the combined_features to a NumPy array
    features_np = np.array(combined_features, dtype=np.float64)

    # Convert the NumPy array to a PyTorch tensor
    features_tensor = torch.tensor(features_np)

    # Append the tensor to the node_features list
    node_features.append(features_tensor)

# Convert the list of tensors to a single tensor
x1 = torch.stack(node_features)

# Convert it to float
x1 = x1.float()

bert_embeddings = torch.stack([torch.tensor(tensor) for tensor in train_df['bert_embeddings'].values])
bert_embeddings = bert_embeddings.squeeze(1)
data = Data(x=x1, edge_index=edge_index)

num_features = x1[0].shape[0]
# print("total number of features are: ", num_features)


class CombinedModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, heads=8):
        super(CombinedModel, self).__init__()
        # Define Graph Convolutional Layers
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.attention = GATConv(32, 16, heads=heads, dropout=0.1)

        # Define components for BERT processing
        self.dropout = nn.Dropout(0.1)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.linear = nn.Linear(768, 16)

        # Final Dense Layer for combining graph and text features
        self.concat_fc = nn.Linear(144, 64)  # Concatenated output to dense layer

        # Output layer: Outputs num_classes logits (for num_classes classes)
        self.output = nn.Linear(64, num_classes)

    def forward(self, graph_data, bert_embeddings):
        # Process graph data
        x, edge_index = graph_data.x, graph_data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x1 = self.attention(x, edge_index)

        # Process BERT embeddings
        sequence_output = bert_embeddings
        multihead_output, _ = self.multihead_attention(sequence_output, sequence_output, sequence_output)
        multihead_output = self.dropout(multihead_output)
        x2 = self.linear(multihead_output)

        # Combine graph and BERT features
        concatenated_output = torch.cat((x1, x2), dim=1)
        combined_output = F.relu(self.concat_fc(concatenated_output))

        # Get logits for each class
        logits = self.output(combined_output)

        # Apply softmax activation layer
        out = F.softmax(logits, dim=1)

        return out


# Define hyperparameters
num_classes = 4
heads = 8  # Number of heads for multi-head attention

# Create an instance of the combined model
combined_model = CombinedModel(num_features, num_classes, heads)
combined_output = combined_model(data, bert_embeddings)


# Create a soft labels for the given labels.
def soft_encoded_vector(label, penalty, num_classes):
    classes = list(range(num_classes))
    cost = lambda val, label: penalty * (abs(label - val))
    expr = lambda val, label: math.exp(-cost(val, label))
    total_sum = sum([expr(val, label) for val in classes])
    soft_vector = torch.tensor([(expr(val, label) / total_sum) for val in classes], dtype=torch.float32)
    return soft_vector


def convert_labels_to_vec(labels, penalty, num_classes):
    soft_labels = torch.stack([soft_encoded_vector(label.item(), penalty, num_classes) for label in labels])
    return soft_labels


learning_rate = 4e-5
# Define the optimizer
optimizer = torch.optim.Adam(combined_model.parameters(), lr=learning_rate)

# Define the loss function as KLDivLoss
criterion = nn.KLDivLoss(reduction="batchmean")

# Lists to store training history
loss_history = []

# Number of training epochs
num_epochs = 50
penalty = 2

for epoch in range(num_epochs):
    # Set the model in training mode
    combined_model.train()

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass: Get model predictions (log probabilities)
    pred = combined_model(data, bert_embeddings)

    # Prepare labels with soft encoding
    # Note: You should have your labels prepared in the variable `labels`
    soft_labels = convert_labels_to_vec(labels.numpy(), penalty, num_classes).to(torch.float32)

    # Calculate the loss
    loss = criterion(pred, soft_labels)

    # Backpropagation
    loss.backward()

    # Update model parameters
    optimizer.step()

    # Store loss in history list
    loss_history.append(loss.item())

    # Print and log training progress
    if (((epoch + 1) % 10) == 0) or epoch == 0:
        print(f'Epoch: {epoch + 1:03d}, Loss: {loss.item():.4f}')

# Get the predicted classes
_, predicted_classes = torch.max(pred, 1)
y_true = labels.numpy()  # Convert to numpy array if necessary


def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


# Convert predicted_classes tensor to numpy for evaluation
y_pred = predicted_classes.cpu().numpy()
evaluate(y_true, y_pred)

"""Testing Graph
We already have the training graph, "graph," which was created using 80% of the data. The model was trained with it. In the next step, we will use the same graph and add the testing nodes to it, which constitute the remaining 20% of the data, and then we use these testing nodes on the graph for evaluation.
"""

# Total number of train nodes
train_nodes = len(graph.nodes())

# Create a dictionary to store attributes for each ID
attributes_dict = {}

# Iterate through the rows of the DataFrame and create a dictionary
for _, row in test_df.iterrows():
    tweet = row['clean_tweet']
    attributes = {
        'Automated_readability_index': row['Automated_readability_index'],
        'polarity': row['polarity'],
        'subjectivity': row['subjectivity'],
        'positive_word': row['positive_word'],
        'negative_word': row['negative_word'],
        'C_NOUN': row['C_NOUN'],
        'C_verb': row['C_verb'],
        'C_adj': row['C_adj'],
        'C_adv': row['C_adv'],
        'Syllable_count': row['Syllable_count'],
        'Coleman_liau_index': row['Coleman_liau_index'],
        'Gunning_fog': row['Gunning_fog'],
        'Flesch_reading_ease': row['Flesch_reading_ease'],
        'Average_VAD': row['Average_VAD'],
        'Domi0ce': row['Domi0ce'],
        'Arousal': row['Arousal'],
        'Valence': row['Valence'],
        'ED_related': row['ED_related'],
        'count_eating': row['count_eating'],
        'label': row['label'],
        'bert_embeddings': row['bert_embeddings']
    }
    attributes_dict[tweet] = attributes

# Add nodes to the graph with attributes
for tweet, attributes in attributes_dict.items():
    # add each node with all the features
    graph.add_node(tweet, **attributes, node_type="tweet")

# Establish edges between nodes based on common lexicon words
for idx1, row1 in test_df.iterrows():
    lex_words_node1 = set(row1['lexicon_words'])
    tweet1_id = row1['clean_tweet']

    for idx2, row2 in test_df.iterrows():
        # Skip the same row (i.e., comparing a tweet with itself)
        if idx1 == idx2:
            continue

        lex_words_node2 = set(row2['lexicon_words'])
        tweet2_id = row2['clean_tweet']

        # Check if both tweets have at least one common lexicon word and avoid adding duplicate edges
        if len(set(lex_words_node1).intersection(set(lex_words_node2))) >= 1:
            graph.add_edge(tweet1_id, tweet2_id)

# Total number of nodes
total_nodes = len(graph.nodes())

# print(f"Total Graph data: {graph}")

labels = test_df.label.values
labels = torch.tensor(labels)

# Create a mapping from node names to unique integer indices
node_to_index = {node: i for i, node in enumerate(graph.nodes())}

# Convert edge list to list of tuples using the mapping
edge_list = [(node_to_index[edge[0]], node_to_index[edge[1]]) for edge in graph.edges()]

# Convert edge list to tensor
edge_index = torch.tensor(edge_list).t().contiguous()

node_features = []

for node in graph.nodes():
    scalar_features = [
        graph.nodes[node]['Automated_readability_index'],
        graph.nodes[node]['polarity'],
        graph.nodes[node]['subjectivity'],
        graph.nodes[node]['positive_word'],
        graph.nodes[node]['negative_word'],
        graph.nodes[node]['C_NOUN'],
        graph.nodes[node]['C_verb'],
        graph.nodes[node]['C_adj'],
        graph.nodes[node]['C_adv'],
        graph.nodes[node]['Syllable_count'],
        graph.nodes[node]['Coleman_liau_index'],
        graph.nodes[node]['Gunning_fog'],
        graph.nodes[node]['Flesch_reading_ease'],
        graph.nodes[node]['Average_VAD'],
        graph.nodes[node]['Domi0ce'],
        graph.nodes[node]['Arousal'],
        graph.nodes[node]['Valence'],
        graph.nodes[node]['ED_related'],
        graph.nodes[node]['count_eating']
    ]

    # Convert the 'bert_embeddings' to a flat array
    flattened_bert_embeddings = graph.nodes[node]['bert_embeddings'].flatten()

    # Combine scalar features and flattened 'bert_embeddings'
    combined_features = scalar_features + flattened_bert_embeddings.tolist()

    # Convert the combined_features to a NumPy array
    features_np = np.array(combined_features, dtype=np.float64)

    # Convert the NumPy array to a PyTorch tensor
    features_tensor = torch.tensor(features_np)

    # Append the tensor to the node_features list
    node_features.append(features_tensor)

# Convert the list of tensors to a single tensor
x1 = torch.stack(node_features)

# Convert it to float
x1 = x1.float()

bert_embeddings = torch.stack([torch.tensor(tensor) for tensor in df['bert_embeddings'].values])
bert_embeddings = bert_embeddings.squeeze(1)
data = Data(x=x1, edge_index=edge_index)

out = combined_model(data, bert_embeddings)

# Get the predicted classes
_, predicted_classes = torch.max(out, 1)
y_true = labels.numpy()  # Convert to numpy array if necessary

# Getting results only for testing data
test_pred = predicted_classes[train_nodes:]

print("Testing Data: ")
# Make predictions for the test set
evaluate(y_true, test_pred)
