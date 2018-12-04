import csv
import json
import re
import string
import tensorflow as tf
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb #The purpose of this dataset is to use the word to number dictionary
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

tweets_dict = []
with open("fetched_tweets.txt") as f:
    for i in range(66137):
        line = f.readline()
        tweet = json.loads(line)
        fulltext = tweet['text']

        fulltext = re.sub(r'[.,"!]+', '', fulltext)  # removes the characters specified
        fulltext = re.sub(r'^RT[\s]+', '', fulltext)  # removes RT
        fulltext = re.sub(r'https?:\/\/.[\r\n]', '', fulltext)  # remove link
        fulltext = re.sub(r'[:]+', '', fulltext)
        new_line = ''
        for i in fulltext.split():  # remove @ and #words, punctuataion
            if not i.startswith('@') and not i.startswith('#') and i \
                    not in string.punctuation:
                new_line += i + ' '
        fulltext = new_line.lower()

        words = fulltext.split()
        result = [] #Holds single tweet split as per imdb numbers
        for word in words:
            if word in word_index:
                result.append(word_index[word])
        tweets_dict.append(result)

#tweet_dict holds the tweets encoded into numbers

train_dataset_data =[]
train_dataset_labels = []
with open('cleantextlabels7.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        data_text = rows[0]
        data_text = data_text.split()
        res = []
        for word in data_text:
            if word in word_index:
                res.append(word_index[word])
        train_dataset_data.append(res)
        train_dataset_labels.append((int)(rows[1]))


train_data = keras.preprocessing.sequence.pad_sequences(train_dataset_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

tweets_dict = keras.preprocessing.sequence.pad_sequences(tweets_dict,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

vocab_size = len(word_index)

model = keras.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(6, activation=tf.nn.relu))
model.add(keras.layers.Dense(3, activation=tf.nn.softmax))


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:5000]
partial_x_train = train_data[5000:]

y_val = train_dataset_labels[:5000]
partial_y_train = train_dataset_labels[5000:]

model.fit(np.array(partial_x_train), np.array(partial_y_train), epochs=40, batch_size=512, validation_data=(np.array(x_val), np.array(y_val)),verbose=1)

res = model.predict_classes(tweets_dict)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

countzero = 0
countone = 0
counttwo = 0

for i in range(len(res)):
    if res[i] == 0:
        countzero += 1
    elif res[i] == 1:
        countone += 1
    elif res[i] == 2:
        counttwo += 1

print()
print("Zeroes : " + str(countzero))
print("Ones : " + str(countone))
print("Twos : " + str(counttwo))
print()

print(res[:100])


for i in range(len(res)):
    with open("test_results.txt", "a") as results:
        results.write(str(res[i]))
        results.write("\n")