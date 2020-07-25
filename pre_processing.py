import pandas as pd
# this is to extract the data from that .tgz file
import tarfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# get all of the data out of that .tgz
yelp_reviews = tarfile.open('data/yelp_review_polarity_csv.tgz')
yelp_reviews.extractall('data')
yelp_reviews.close()

# check out what the data looks like before you get started
# look at the training data set
train_df = pd.read_csv('data/yelp_review_polarity_csv/train.csv', header=None)
print(train_df.head())

# look at the test data set
test_df = pd.read_csv('data/yelp_review_polarity_csv/test.csv', header=None)
print(test_df.head())

# convert training and test data to standard 0 and 1 labels
train_df[0] = (train_df[0] == 2).astype(int)
test_df[0] = (test_df[0] == 2).astype(int)

# look at data after conversion
print(train_df.head())
print(test_df.head())

# format data to match what Bert expects
bert_df = pd.DataFrame({
    'id': range(len(train_df)),
    'label': train_df[0],
    'alpha': ['q']*train_df.shape[0],
    'text': train_df[1].replace(r'\n', ' ', regex=True)
})

# split training data into train file and dev file
train_bert_df, dev_bert_df = train_test_split(bert_df, test_size=0.01)

# take a look at the newly formatted data
train_bert_df.head()

# format the test data
test_bert_df = pd.DataFrame({
    'id': range(len(test_df)),
    'text': test_df[1].replace(r'\n', ' ', regex=True)
})

# look at the formatted test data
test_bert_df.head()

# save train, dev, and test data as tsv files
train_bert_df.to_csv('data/train.tsv', sep='\t', index=False, header=False)
dev_bert_df.to_csv('data/dev.tsv', sep='\t', index=False, header=False)
test_bert_df.to_csv('data/test.tsv', sep='\t', index=False, header=False)
