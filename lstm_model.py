import tensorflow as tf
import datetime as dt
import sys
import numpy as np
from tensorflow.python.layers import core as layers_core


def lstm_model(args, np_embeddings, mode='train'):
    vocab_size, hidden_size = np_embeddings.shape

    # Embeddings
    with tf.variable_scope('embeddings'):
        encoder_embeddings = tf.get_variable(name="encoder_embeddings", shape=np_embeddings.shape, initializer=tf.constant_initializer(np_embeddings), trainable=True)
        decoder_embeddings = tf.get_variable(name="decoder_embeddings", shape=np_embeddings.shape, initializer=tf.constant_initializer(np_embeddings), trainable=True)

    # Define placeholders
    with tf.variable_scope('placeholders'):
        seq_source_ids = tf.placeholder(tf.int32, shape=(None, args.max_seq_length), name="source_ids")
        seq_source_lengths = tf.placeholder(tf.int32, [None], name="sequence_source_lengths")
        seq_reference_ids = tf.placeholder(tf.int32, shape=(None, args.max_seq_length), name="reference_ids")
        seq_reference_lengths = tf.placeholder(tf.int32, [None], name="sequence_reference_lengths")
        seq_output_ids = tf.placeholder(tf.int32, shape=(None, args.max_seq_length), name="output_ids")

    batch_size = tf.cast(tf.shape(seq_source_ids)[0], tf.float32)

    # Encoder embeddings
    with tf.variable_scope('encoder_embedding'):
        encoder_embedding = tf.nn.embedding_lookup(encoder_embeddings, seq_source_ids, name="encoder_embedding")

    # Encoder
    with tf.variable_scope('encoder'):
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_size)
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_embedding, sequence_length=seq_source_lengths, dtype=tf.float32)

    # Decoder embeddings
    with tf.variable_scope('decoder_embedding'):
        decoder_embedding = tf.nn.embedding_lookup(decoder_embeddings, seq_reference_ids, name="decoder_embedding")

    # Train
    if mode == 'train':
        # Decoder
        with tf.variable_scope('decoder'):
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_size)
            projection_layer = layers_core.Dense(vocab_size, use_bias=False)
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedding, seq_reference_lengths)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_states)
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, swap_memory=True)
            logits = projection_layer(final_outputs.rnn_output)
            sample_id = final_outputs.sample_id

        with tf.variable_scope('train_loss'):
            max_output_len = tf.shape(logits)[1]
            pad = tf.fill((tf.shape(seq_output_ids)[0], args.max_seq_length), -1)
            boolean_mask = tf.not_equal(seq_output_ids, pad)
            mask = tf.cast(boolean_mask, tf.float32)[:, :max_output_len]
            labels = tf.reshape(seq_output_ids[:, :max_output_len], shape=(-1, 1))
            crossent = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, vocab_size), logits=logits)
            #crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            train_loss = (tf.reduce_sum(crossent * mask) / batch_size)
    # Test
    elif mode == 'test':
        with tf.variable_scope('infer'):
            start_tokens = tf.fill([batch_size], vocab_size - 2)
            end_token = vocab_size - 1
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(args.hidden_size)
            projection_layer = layers_core.Dense(vocab_size, use_bias=False)
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder, start_tokens, end_token)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)
            maximum_iterations = tf.round(tf.reduce_max(seq_source_lengths) * 2)
            outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations, swap_memory=True) 
            logits = outputs.rnn_output
            sample_id = outputs.sample_id


    return {
        'seq_source_ids': seq_source_ids,
        'seq_source_lengths': seq_source_lengths,
        'seq_reference_ids': seq_reference_ids,
        'seq_reference_lengths': seq_reference_lengths,
        'seq_output_ids': seq_output_ids,
        'embedding_source': encoder_embedding,
        'encoder_states': encoder_states,
        'loss': train_loss,
        'logits': logits,
        'sample_id': sample_id,
        'labels': labels,
    }

