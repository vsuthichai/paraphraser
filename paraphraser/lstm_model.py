import tensorflow as tf
import datetime as dt
import sys
import numpy as np
from tensorflow.python.layers import core as layers_core
from sample_embedding_helper import MySampleEmbeddingHelper

#def lstm_model(args, np_embeddings, start_id, end_id, mask_id, mode):
def lstm_model(sess, mode, cell_hidden_size, np_embeddings, start_id, end_id, mask_id):
    vocab_size, hidden_size = np_embeddings.shape

    # Embeddings
    with tf.variable_scope('embeddings'):
        encoder_embeddings = tf.get_variable(name="encoder_embeddings", shape=np_embeddings.shape, initializer=tf.constant_initializer(np_embeddings), trainable=True)
        decoder_embeddings = tf.get_variable(name="decoder_embeddings", shape=np_embeddings.shape, initializer=tf.constant_initializer(np_embeddings), trainable=True)
        #embeddings = tf.get_variable(name="embeddings", shape=np_embeddings.shape, initializer=tf.constant_initializer(np_embeddings), trainable=True)

    # Define placeholders
    with tf.variable_scope('placeholders'):
        lr = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        seq_source_ids = tf.placeholder(tf.int32, shape=(None, None), name="source_ids")
        seq_source_lengths = tf.placeholder(tf.int32, [None], name="sequence_source_lengths")
        keep_prob = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")
        # 0: greedy, 1: sampling, 2: beam
        sampling_temperature = tf.placeholder_with_default(0.5, shape=(), name="sampling_temperature")
        decoder_technique = tf.placeholder_with_default(1, shape=(), name="decoder_technique")
        #beam_width = tf.placeholder_with_default(5, shape=(), name="beam_width")
        dummy = tf.add(sampling_temperature, 1, name="dummy")

        if mode in set(['train', 'dev', 'test']):
            seq_reference_ids = tf.placeholder(tf.int32, shape=(None, None), name="reference_ids")
            seq_reference_lengths = tf.placeholder(tf.int32, [None], name="sequence_reference_lengths")
            paddings = tf.constant([[0, 0], [0, 1]])
            seq_output_ids = tf.pad(seq_reference_ids[:, 1:], paddings, mode="CONSTANT", name="seq_output_ids", constant_values=mask_id)
        else:
            seq_reference_ids = None
            seq_reference_lengths = None
            seq_output_ids = None

    #batch_size = tf.cast(tf.shape(seq_source_ids)[0], tf.float32)
    batch_size = tf.shape(seq_source_ids)[0]

    # Encoder
    #with tf.variable_scope('encoder'):
    encoder_embedding = tf.nn.embedding_lookup(encoder_embeddings, seq_source_ids, name="encoder_embedding")
    encoder_fw_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(cell_hidden_size), input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    encoder_bw_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(cell_hidden_size), input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    encoder_outputs, encoder_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_fw_cell, 
                                                                      cell_bw=encoder_bw_cell,
                                                                      inputs=encoder_embedding, 
                                                                      sequence_length=seq_source_lengths, 
                                                                      dtype=tf.float32)
    concat_encoder_outputs = tf.concat(encoder_outputs, 2)
    encoder_fw_state, encoder_bw_state = encoder_states
    encoder_state_c = tf.concat((encoder_fw_state.c, encoder_bw_state.c), axis=1, name="encoder_state_c")
    encoder_state_h = tf.concat((encoder_fw_state.h, encoder_bw_state.h), axis=1, name="encoder_state_h")
    joined_encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state_c, encoder_state_h)

    fc_layer = layers_core.Dense(vocab_size, use_bias=False)
    attention = tf.contrib.seq2seq.BahdanauAttention(num_units=cell_hidden_size, memory=concat_encoder_outputs)
    decoder_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(cell_hidden_size * 2), input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention, attention_layer_size=cell_hidden_size)
    zero_state = attn_cell.zero_state(batch_size, tf.float32)
    decoder_initial_state = zero_state.clone(cell_state=joined_encoder_state)

    ''' Beam search 
    tiled_joined_encoder_state = tf.contrib.seq2seq.tile_batch(joined_encoder_state, multiplier=beam_width)
    tiled_concat_encoder_outputs = tf.contrib.seq2seq.tile_batch(concat_encoder_outputs, multiplier=beam_width)
    beam_attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units=hidden_size,
        memory=tiled_concat_encoder_outputs)
    #decoder_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_size * 2), input_keep_prob=1.0, output_keep_prob=1.0)
    beam_attn_wrapper = tf.contrib.seq2seq.AttentionWrapper(
        cell=tf.nn.rnn_cell.BasicLSTMCell(hidden_size * 2),
        attention_mechanism=beam_attention_mechanism, 
        attention_layer_size=hidden_size)
    '''

    # Train, dev, test
    if mode in set(['train', 'dev', 'test']):
        # Decoder
        decoder_embedding = tf.nn.embedding_lookup(decoder_embeddings, seq_reference_ids, name="decoder_embedding")
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_embedding, seq_reference_lengths)
        decoder = tf.contrib.seq2seq.BasicDecoder(attn_cell, helper, decoder_initial_state, fc_layer)
        final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, swap_memory=True)
        logits = final_outputs.rnn_output
        predictions = final_outputs.sample_id

        with tf.variable_scope('train_loss'):
            max_output_len = tf.shape(logits)[1]
            seq_output_ids = seq_output_ids[:, :max_output_len]
            pad = tf.fill((tf.shape(seq_output_ids)[0], max_output_len), -1) #mask_id
            boolean_mask = tf.not_equal(seq_output_ids, pad)
            mask = tf.cast(boolean_mask, tf.float32)
            labels = tf.reshape(seq_output_ids, shape=(-1, 1))
            crossent = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, vocab_size), logits=logits)
            loss = (tf.reduce_sum(crossent * mask) / tf.cast(batch_size, tf.float32))

        with tf.variable_scope('summaries'):
            tf.summary.scalar("batch_loss", loss)
            summaries = tf.summary.merge_all()

        train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    # Test
    elif mode == 'infer':
        loss = None
        train_step = None
        labels = None
        summaries = None
        start_tokens = tf.fill([batch_size], start_id)

        # Beach search decoder
        '''
        beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=beam_attn_wrapper,
            embedding=decoder_embeddings,
            start_tokens=start_tokens,
            end_token=end_id,
            initial_state=beam_attn_wrapper.zero_state(batch_size * beam_width, tf.float32).clone(cell_state=tiled_joined_encoder_state),
            beam_width=beam_width.eval(),
            output_layer=fc_layer,
            length_penalty_weight=0.0)
        '''

        # Distribution sampling
        #sample_helper = MySampleEmbeddingHelper(decoder_embeddings, start_tokens, end_id, softmax_temperature=sampling_temperature)
        sample_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(decoder_embeddings, start_tokens, end_id, softmax_temperature=sampling_temperature)
        sample_decoder = tf.contrib.seq2seq.BasicDecoder(attn_cell, sample_helper, decoder_initial_state, output_layer=fc_layer)

        # Greedy argmax decoder
        greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_tokens, end_id)
        # applied per timestep
        greedy_decoder = tf.contrib.seq2seq.BasicDecoder(attn_cell, greedy_helper, decoder_initial_state, output_layer=fc_layer)

        # Decode!
        greedy_outputs, greedy_final_state, greedy_fsl = tf.contrib.seq2seq.dynamic_decode(
            greedy_decoder,
            #maximum_iterations=maximum_iterations,
            swap_memory=True)
        greedy_logits = greedy_outputs.rnn_output
        greedy_predictions = tf.identity(greedy_outputs.sample_id, name="greedy_predictions")

        sample_outputs, sample_final_state, sample_fsl = tf.contrib.seq2seq.dynamic_decode(
            sample_decoder,
            swap_memory=True)
        sample_logits = sample_outputs.rnn_output
        sample_predictions = tf.identity(sample_outputs.sample_id, name="sample_predictions")

        '''
        beam_search_outputs, beam_search_final_state, beam_search_fsl = tf.contrib.seq2seq.dynamic_decode(
            beam_search_decoder,
            swap_memory=True)
        beam_search_logits = tf.no_op()
        beam_search_predictions = tf.identity(beam_search_outputs.predicted_ids, name="beam_search_predictions")
        print(beam_search_predictions)
        '''
        z,y,a = tf.case(
            pred_fn_pairs={
                tf.equal(sampling_temperature, tf.constant(0.0)): lambda: (greedy_predictions, greedy_fsl, greedy_logits),
                tf.equal(sampling_temperature, tf.constant(1.0)): lambda: (sample_predictions, sample_fsl, sample_logits),
            },
            default = lambda: (sample_predictions, sample_fsl, sample_logits),
            exclusive=True
        )

        predictions, final_sequence_lengths, logits = tf.case(
            pred_fn_pairs={
                tf.equal(decoder_technique, tf.constant(0)): lambda: (greedy_predictions, greedy_fsl, greedy_logits),
                tf.equal(decoder_technique, tf.constant(1)): lambda: (sample_predictions, sample_fsl, sample_logits),
                #tf.equal(decoder_technique, tf.constant(2)): lambda: (beam_search_predictions, beam_search_fsl)
            },
            exclusive=True)

        predictions = tf.identity(predictions, name='predictions')
        final_sequence_lengths = tf.identity(final_sequence_lengths, name='final_sequence_lengths')
        logits = tf.identity(logits, name='logits')

    return {
        'lr': lr,
        'keep_prob': keep_prob,
        'decoder_technique': decoder_technique,
        'sampling_temperature': sampling_temperature,
        #'beam_width': beam_width,
        'seq_source_ids': seq_source_ids,
        'seq_source_lengths': seq_source_lengths,
        'seq_reference_ids': seq_reference_ids,
        'seq_reference_lengths': seq_reference_lengths,
        #'final_state': final_state,
        'final_sequence_lengths': final_sequence_lengths,
        'embedding_source': encoder_embedding,
        'encoder_states': encoder_states,
        'loss': loss,
        'predictions': predictions,
        'labels': labels,
        'summaries': summaries,
        'train_step': train_step,
        'dummy': dummy
    }

