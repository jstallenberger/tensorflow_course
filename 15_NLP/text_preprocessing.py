from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Just a simple test
sentences = [
    "I like eggs and ham.",
    "I love chocolate and bunnies.",
    "I hate onions."
]

MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

print(sequences)

# How to get the word index mapping?
print(tokenizer.word_index)

# use the defaults
data = pad_sequences(sequences)
print(data)

MAX_SEQUENCE_LENGTH = 5
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(data)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
print(data)

# too much padding
data = pad_sequences(sequences, maxlen=6)
print(data)

# truncation
data = pad_sequences(sequences, maxlen=4)
print(data)

data = pad_sequences(sequences, maxlen=4, truncating='post')
print(data)
