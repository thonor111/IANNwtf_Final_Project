import tensorflow as tf
import tensorflow_text as text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from os.path import exists


class InputPipeline:

    def __init__(self):
        self.start_token = 1
        self.end_token = 0
        self.vocab_size = 9872
        self.tokenizer = None
        self.batch_size = 2
        self.maximal_sentence_length = 3576
        self.sentence_length = 40

    def train_tokenizer(self, data):
        plain_text = data.map(lambda sentence, sentiment: sentence)
        plain_text = plain_text.map(lambda sentence: tf.strings.regex_replace(sentence, "[^a-z A-Z]", ""))
        plain_text = plain_text.padded_batch(1000).prefetch(2)

        bert_tokenizer_params = dict(lower_case=True)

        if not exists('vocab.txt'):
            print('training')
            reserved_tokens = ["[END]", "[START]"]
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
            with open('vocab.txt', 'w', encoding='utf-8') as f:
                for token in vocab:
                    print(token, file=f)
            print("finished training")
        self.tokenizer = text.BertTokenizer('vocab.txt', **bert_tokenizer_params, token_out_type=tf.int64)

    def tokenize_data(self, sentence, label):
        tokenized_text = self.tokenizer.tokenize(sentence)
        tokenized_text = tokenized_text.merge_dims(-2, -1).to_tensor()
        tokenized_text = tf.squeeze(tokenized_text)
        tokenized_text = tf.expand_dims(tokenized_text, -1)
        return tokenized_text, label

    # This was used to shorten the sentences
    # def shorten(self, sentence, sentiment):
    #     sentence = tf.slice(sentence, [0, 0], [self.sentence_length, 1])
    #     return sentence, sentiment

    def prepare_data(self, data):
        # remove parts that are not words
        data = data.map(lambda sentence, sentiment: (tf.strings.regex_replace(sentence, "[^a-z A-Z]", ""), sentiment))
        # tokenizing the text
        data = data.map(self.tokenize_data)

        # This was used to shorten the sentences
        # # adding end to the end of every sentence to be able to achieve a minimum sentence length
        # data = data.map(lambda embedding, sentiment: (tf.concat((embedding,
        #                                                          tf.constant(self.end_token, dtype=tf.int64,
        #                                                                      shape=(self.sentence_length, 1))), axis=0),
        #                                               sentiment))
        #
        # # shortening all sentences to the same length
        # data = data.map(lambda sentence, sentiment: self.shorten(sentence, sentiment))

        # adding start and end to every sentence
        data = data.map(lambda embedding, sentiment: (tf.concat((tf.constant(self.start_token, dtype=tf.int64,
                                                                             shape=(1, 1)), embedding,
                                                                 tf.constant(self.end_token, dtype=tf.int64,
                                                                             shape=(1, 1))), axis=0), sentiment))

        data = data.map(lambda embedding, sentiment: (tf.cast(embedding, tf.float32), sentiment))

        # adding noise as input for the GAN
        data = data.map(lambda embedding, sentiment: (embedding, sentiment, tf.random.uniform(shape=[100])))

        # adding the target for the VAE
        data = data.map(
            lambda embedding, sentiment, noise: (embedding, tf.roll(embedding, shift=-1, axis=0), sentiment, noise))

        # standard pipeline
        data = data.cache().shuffle(1000)
        data = data.padded_batch(self.batch_size)
        data = data.prefetch(20)

        return data
