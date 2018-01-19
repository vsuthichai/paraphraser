from keras.backend import shape
from keras.models import Model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Flatten, Masking
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
import argparse
import numpy as np
from six.moves import xrange
from load_word_embeddings import load_keras_embedding

embeddings_filename = '/media/sdb/datasets/glove.6B/glove.6B.300d.pickle'

def lstm_keras(args):
    encoder_inputs = Input(batch_shape=(None, args.num_encoder_tokens))
    masked_encoder_inputs = Masking(mask_value=-1.)(encoder_inputs)
    encoder_embeddings, vocab_size, embedding_size = load_keras_embedding(embeddings_filename)
    x = encoder_embeddings(masked_encoder_inputs)
    encoder_outputs, hidden_state, cell_state =  LSTM(args.hidden_size, return_state=True)(x)
    encoder_states = [hidden_state, cell_state]

    decoder_inputs = Input(batch_shape=(None, args.num_decoder_tokens))
    masked_decoder_inputs = Masking(mask_value=-1.)(decoder_inputs)
    decoder_embeddings, _, _ = load_keras_embedding(embeddings_filename)
    x = decoder_embeddings(masked_decoder_inputs)
    decoder_outputs, _, _ = LSTM(args.hidden_size, return_sequences=True, return_state=True)(x, initial_state=encoder_states)
    decoder_outputs = TimeDistributed(Dense(1, activation='softmax'))(decoder_outputs)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, help="Log directory to store tensorboard summary and model checkpoints")
    parser.add_argument('--epochs', type=int, default=2000, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size")
    parser.add_argument('--hidden_size', type=int, default=128, help="Hidden dimension size")
    parser.add_argument('--num_encoder_tokens', type=int, default=30, help="Max number of encoder sequence tokens")
    parser.add_argument('--num_decoder_tokens', type=int, default=30, help="Max number of decoder sequence tokens")

    return parser.parse_args()

def main():
    args = parse_arguments()
    model = lstm_keras(args)
    model.summary()

    encoder_input_data = pad_sequences(np.array([[1,2,3,4], [5,6,7,8]]), maxlen=args.num_encoder_tokens, padding='post', value=-1.)
    decoder_input_data = pad_sequences(np.array([[400001, 1,3,2,5,4], [400001, 6,5,8,7]]), maxlen=args.num_decoder_tokens, padding='post', value=-1.)
    decoder_target_data = pad_sequences(np.array([[1,3,2,5,4, 400002], [6,5,8,7,400002]]), maxlen=args.num_decoder_tokens, padding='post', value=-1.)
    decoder_target_data = np.expand_dims(decoder_target_data, -1)

    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs=args.epochs)

if __name__ == '__main__':
    main()


