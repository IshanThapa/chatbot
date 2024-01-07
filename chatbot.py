import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

# ... (Your data preprocessing code remains the same) ...
#importing data-

lines = open('movie_lines.tsv',encoding = 'utf-8', errors = 'ignore').read().split('\n')

conversations = open('movie_conversations.tsv',encoding = 'utf-8', errors = 'ignore').read().split('\n')

id2line = {}

with open('movie_lines.tsv', 'r') as file:
    for line in file:
        _line = line.strip().split('\t')
        if len(_line) >= 5:
            id2line[_line[0]] = _line[4]

conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.strip().split('\t')  # Split the conversation using tab as the delimiter
    if len(_conversation) >= 2:  # Ensure the conversation has at least two fields
        conv_id = _conversation[0]  # Assuming the ID is in the first field
        last_field = _conversation[-1]  # Retrieve the last field
        conversations_ids.append((conv_id, last_field))  # Append a tuple with ID and last field




# Define the model inputs
def model_inputs():
    inputs = Input(shape=(None,))
    targets = Input(shape=(None,))
    lr = Input(shape=(), dtype=tf.float32)
    keep_prob = Input(shape=(), dtype=tf.float32)
    return inputs, targets, lr, keep_prob

# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets

# Encoder RNN
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = Bidirectional(LSTM(rnn_size, return_sequences=True, dropout=keep_prob))(rnn_inputs)
    encoder_cell = Bidirectional(LSTM(rnn_size, return_sequences=True, dropout=keep_prob))(lstm)
    return encoder_cell

# Decoder Training Set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.raw_ops.Empty(shape=(0, 0, 0), dtype=tf.float32), tf.raw_ops.Empty(shape=(0, 0, 0), dtype=tf.float32), tf.raw_ops.Empty(shape=(0, 0, 0), dtype=tf.float32), tf.raw_ops.Empty(shape=(0, 0, 0), dtype=tf.float32)
    training_decoder_function = tf.compat.v1.raw_ops.AttentionDecoderFnTrain(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name="attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.compat.v1.raw_ops.DynamicRnnDecoder(decoder_cell,
                                                                                                                training_decoder_function,
                                                                                                                decoder_embedded_input,
                                                                                                                sequence_length,
                                                                                                                scope=decoding_scope)
    decoder_output_dropout = Dropout(keep_prob)(decoder_output)
    return output_function(decoder_output_dropout)

# Decoder Test Set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.raw_ops.Empty(shape=(0, 0, 0), dtype=tf.float32), tf.raw_ops.Empty(shape=(0, 0, 0), dtype=tf.float32), tf.raw_ops.Empty(shape=(0, 0, 0), dtype=tf.float32), tf.raw_ops.Empty(shape=(0, 0, 0), dtype=tf.float32)
    test_decoder_function = tf.compat.v1.raw_ops.AttentionDecoderFnInference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name="attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.compat.v1.raw_ops.DynamicRnnDecoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope=decoding_scope)
    return test_predictions

# Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.compat.v1.variable_scope("decoding") as decoding_scope:
        lstm = LSTM(rnn_size, return_sequences=True)
        lstm_dropout = Dropout(keep_prob)(lstm)
        decoder_cell = LSTM(rnn_size, return_sequences=True)
        weights = tf.initializers.TruncatedNormal(stddev=0.1)
        biases = tf.initializers.Zeros()
        output_function = Dense(num_words, None, name="dense_layer", kernel_initializer=weights, bias_initializer=biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions

# Seq2Seq Model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = Embedding(answers_num_words + 1, encoder_embedding_size)(inputs)
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = Embedding(questions_num_words + 1, decoder_embedding_size)(targets)
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions

# ... (The training and testing code remains the same) ...
