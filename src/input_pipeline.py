import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

class Input_Pipeline():

    def __init__(self):
        self.tokenizer_layer = tf.keras.layers.TextVectorization(output_sequence_length=250)
        self.start_token = 1
        self.end_token = 0
        self.vocab_size=10000
        self.tokenizer = None

    def write_vocab_file(filepath, vocab):
        with open(filepath, 'w') as f:
            for token in vocab:
                print(token, file=f)

    def train_tokenizer(self, data):
        plain_text = data.map(lambda text, sentiment: text)

        bert_tokenizer_params = dict(lower_case=True)
        reserved_tokens = ["[START]", "[END]"]

        bert_vocab_args = dict(
            # The target vocabulary size
            vocab_size=self.vocab_size,
            # Reserved tokens that must be included in the vocabulary
            reserved_tokens=reserved_tokens,
            # Arguments for `text.BertTokenizer`
            bert_tokenizer_params=bert_tokenizer_params,
            # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
            learn_params={},
        )
        vocab = bert_vocab.bert_vocab_from_dataset(
            plain_text,
            **bert_vocab_args
        )
        with open('vocab.txt', 'w') as f:
            for token in vocab:
                print(token, file=f)
        self.tokenizer = text.BertTokenizer('vocab.txt', **bert_tokenizer_params, token_out_type=tf.int64)

    def prepare_data(self, data):
        # tokenizing the text
        data = data.map(self.tokenize_data)
        # standard pipeline
        data = data.cache().shuffle(1000).batch(10).prefetch(20)

        return data


    def tokenize_data(self, text, label):
        # text = tf.expand_dims(text, -1)
        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_text = tokenized_text.merge_dims(-2, -1).to_tensor()
        tokenized_text = tf.squeeze(tokenized_text)
        tokenized_text = tf.expand_dims(tokenized_text, -1)
        return tokenized_text, label


    def prepare_data_GAN(self, data):
        # tokenizing the text
        data = data.map(self.tokenize_data)

        # adding start and end to every sentence
        data = data.map(lambda embedding, sentiment: (tf.concat((tf.constant(self.start_token, dtype=tf.int64,
                                                                             shape=(1,1)), embedding,
                                                                 tf.constant(self.end_token, dtype=tf.int64,
                                                                             shape=(1,1))), axis=0), sentiment))

        data = data.map(lambda embedding, sentiment: (tf.cast(embedding, tf.float32), sentiment))

        # adding noise as input for the GAN
        data = data.map(lambda embedding, sentiment: (embedding, sentiment, tf.random.uniform(shape=[100])))

        data = data.map(
            lambda embedding, sentiment, noise: (embedding, tf.roll(embedding, shift=-1, axis=0), sentiment, noise))


        # standard pipeline
        data = data.cache().shuffle(1000).padded_batch(3).prefetch(20)

        return data
