from keras.optimizers import Adam
from keras.backend import shape, gradients
from keras.models import Model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Flatten, Masking
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
import argparse
import numpy as np
from six.moves import xrange
from load_sent_embeddings import load_sentence_embeddings
from load_para_nmt_5m import generate_batch

#embeddings_filename = '/media/sdb/datasets/glove.6B/glove.6B.300d.pickle'

def lstm_keras(args, embeddings, mask_val):
    vocab_size, embedding_size = embeddings.shape
    # Encoder
    encoder_inputs = Input(batch_shape=(None, args.max_encoder_tokens))
    masked_encoder_inputs = Masking(mask_value=mask_val)(encoder_inputs)

    encoder_embeddings = Embedding(vocab_size, embedding_size, weights=[embeddings], trainable=True)
    x = encoder_embeddings(masked_encoder_inputs)

    encoder_outputs, hidden_state, cell_state = LSTM(args.hidden_size, return_state=True)(x)
    encoder_states = [hidden_state, cell_state]

    # Decoder
    decoder_inputs = Input(batch_shape=(None, args.max_decoder_tokens))
    masked_decoder_inputs = Masking(mask_value=mask_val)(decoder_inputs)

    decoder_embeddings = Embedding(vocab_size, embedding_size, weights=[embeddings], trainable=True)
    x = decoder_embeddings(masked_decoder_inputs)

    decoder_lstm, _, _ = LSTM(args.hidden_size, return_sequences=True, return_state=True)(x, initial_state=encoder_states)
    decoder_outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder_lstm)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    optimizer = Adam(lr=1e-3, clipnorm=5.)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    return model

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, help="Log directory to store tensorboard summary and model checkpoints")
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size")
    parser.add_argument('--hidden_size', type=int, default=128, help="Hidden dimension size")
    parser.add_argument('--max_encoder_tokens', type=int, default=30, help="Max number of encoder sequence tokens")
    parser.add_argument('--max_decoder_tokens', type=int, default=30, help="Max number of decoder sequence tokens")

    return parser.parse_args()

def main():
    args = parse_arguments()
    word_to_id, idx_to_word, embeddings, start_id, end_id, unk_id = load_sentence_embeddings()
    vocab_size, embedding_size = embeddings.shape
    model = lstm_keras(args, embeddings, end_id)
    model.summary()

    for epoch in xrange(args.epochs):
        for source_ids, ref_ids, target_ids, source_len, ref_len, target_len in generate_batch(2, word_to_id, start_id, end_id, unk_id):
            #print(source_ids)
            #print(ref_ids)
            #print(target_ids)
            #print(source_len)
            #print(ref_len)
            #print(target_len)

            encoder_input_data = np.array(pad_sequences(source_ids, maxlen=args.max_encoder_tokens, padding='post', value=end_id))
            decoder_input_data = np.array(pad_sequences(ref_ids, maxlen=args.max_decoder_tokens, padding='post', value=end_id))
            decoder_target_data = np.array(pad_sequences(target_ids, maxlen=args.max_decoder_tokens, padding='post', value=end_id))

            one_hot_decoder_target_data = np.zeros((2, args.max_decoder_tokens, vocab_size))
            for i in xrange(decoder_target_data.shape[0]):
                for j in xrange(args.max_decoder_tokens):
                    one_hot_decoder_target_data[i, j, decoder_target_data[i][j]] = 1

            model.fit([encoder_input_data, decoder_input_data], one_hot_decoder_target_data, epochs=1)

if __name__ == '__main__':
    main()


