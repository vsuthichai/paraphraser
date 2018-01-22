from keras.optimizers import Adam
from keras.backend import shape, gradients
from keras.models import Model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Flatten, Masking, Bidirectional, Concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
import argparse
import numpy as np
from six.moves import xrange
from load_sent_embeddings import load_sentence_embeddings
from load_para_nmt_5m import generate_batch, keras_generate_batch
from math import ceil
from dataset_generator import ParaphraseDataset

#embeddings_filename = '/media/sdb/datasets/glove.6B/glove.6B.300d.pickle'

def lstm_keras(args, embeddings, mask_val):
    vocab_size, embedding_size = embeddings.shape

    # Encoder
    encoder_inputs = Input(batch_shape=(None, args.max_encoder_tokens))
    masked_encoder_inputs = Masking(mask_value=mask_val)(encoder_inputs)

    encoder_embeddings = Embedding(vocab_size, embedding_size, weights=[embeddings], trainable=True)
    x = encoder_embeddings(masked_encoder_inputs)

    encoder_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(args.hidden_size, return_state=True, recurrent_dropout=args.dropout))(x)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(batch_shape=(None, args.max_decoder_tokens))
    masked_decoder_inputs = Masking(mask_value=mask_val)(decoder_inputs)

    #decoder_embeddings = Embedding(vocab_size, embedding_size, weights=[embeddings], trainable=True)
    x = encoder_embeddings(masked_decoder_inputs)

    decoder_lstm, _, _ = LSTM(args.hidden_size * 2, return_sequences=True, return_state=True, recurrent_dropout=args.dropout)(x, initial_state=encoder_states)
    decoder_outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder_lstm)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    #optimizer = Adam(lr=1e-3, clipnorm=5.)
    optimizer = Adam(lr=1e-3)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
    return model

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, help="Log directory to store tensorboard summary and model checkpoints")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=16, help="Mini batch size")
    parser.add_argument('--hidden_size', type=int, default=64, help="Hidden dimension size")
    parser.add_argument('--dropout', type=float, default=0.2, help="Hidden dimension size")
    parser.add_argument('--max_encoder_tokens', type=int, default=30, help="Max number of encoder sequence tokens")
    parser.add_argument('--max_decoder_tokens', type=int, default=30, help="Max number of decoder sequence tokens")

    return parser.parse_args()

def main():
    args = parse_arguments()
    word_to_id, idx_to_word, embeddings, start_id, end_id, unk_id = load_sentence_embeddings()
    mask_id = 5800
    vocab_size, embedding_size = embeddings.shape
    model = lstm_keras(args, embeddings, mask_id)
    model.summary()

    dataset_generator = ParaphraseDataset('/home/victor/datasets/para-nmt-5m-processed/para-nmt-5m-processed.txt', embeddings, word_to_id)
    generator = dataset_generator.generate_batch(args.batch_size, start_id, end_id, unk_id, mask_id, args.max_encoder_tokens, args.max_decoder_tokens)
    steps_per_epoch = dataset_generator.dataset_size / args.batch_size

    #model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=args.epochs, verbose=1, workers=10, max_queue_size=128)
    model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=args.epochs, verbose=1)

    model.save('s2s.h5')

if __name__ == '__main__':
    main()

