import os
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input
import tensorflow as tf

path_to_data = 'Data'
X = []
Y = []
START = '<'
END = '>'
input_characters = set()
target_characters = set()

batch_size = 209  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 836  # Number of samples to train on.


def read_all_data(path):
    for filename in os.listdir(path):
        file = path+'/'+filename
        file_content = ''
        with open(file, mode='r', encoding='utf-8') as f:
            file_content = f.read()
        offset = file_content.find('- -')
        file_content = file_content[offset:]
        conversations = file_content.split('\n')
        previous_line = ''
        for line in conversations:
            if line.find('- - ') != -1:
                X.append(START + line[4:] + END)
                previous_line = line
            else:
                if previous_line.find('  - ') != -1:
                    X.append(START + X[-1] + END)
                    Y.append(START + line[4:] + END)
                else:
                    Y.append(START + line[4:] + END)
                    previous_line = line


read_all_data(path_to_data)

# Vectorize the data.
input_texts = X
target_texts = Y

for line in X:
    for char in line:
        input_characters.add(char)
for line in Y:
    for char in line:
        target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')

for i, input_text in enumerate(input_texts):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.

# Restore the model and construct the encoder and decoder.
model = load_model('lstm_trained.h5')
model.summary()

encoder_inputs = model.input[0]   # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


# Decodes an input sequence.  Future work should support beam search.
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['<']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


def talk():
    print("You are connected with Chatbot!")
    line = input()
    while line != 'exit':
        line = START + line + END
        encoded_sequence = np.zeros((len(line), max_encoder_seq_length, num_encoder_tokens))
        for i, input_text in enumerate(line):
            for t, char in enumerate(line):
                encoded_sequence[i, t, input_token_index[char]] = 1.

        print(decode_sequence(encoded_sequence))
        line = input()

talk()
