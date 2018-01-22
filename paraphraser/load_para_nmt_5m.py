from nlp_pipeline import nlp_pipeline
from math import ceil

def generate_batch(batch_size, word_to_id, start_id, end_id, unk_id):
    i = 0
    batch_source_ids = []
    batch_ref_ids = []
    batch_target_ids = []
    batch_source_lengths = []
    batch_ref_lengths = []
    batch_target_lengths = []

    with open('/media/sdb/datasets/para-nmt-5m-processed/para-nmt-5m-processed.txt', 'r') as f:
        for line in f:
            source, ref = line.split('\t')

            source_ids = nlp_pipeline(source, word_to_id, unk_id)
            ref_ids = nlp_pipeline(ref, word_to_id, unk_id)

            ref_ids = [start_id] + ref_ids + [end_id]
            target_ids = ref_ids[1:]

            source_len = len(source_ids)
            ref_len = len(ref_ids)
            target_len = len(target_ids)

            batch_source_ids.append(source_ids)
            batch_ref_ids.append(ref_ids)
            batch_target_ids.append(target_ids)
            batch_source_lengths.append(source_len)
            batch_ref_lengths.append(ref_len)
            batch_target_lengths.append(target_len)

            i += 1
            if i % batch_size == 0:
                yield batch_source_ids, batch_ref_ids, batch_target_ids, batch_source_lengths, batch_ref_lengths, batch_target_lengths
                batch_source_ids = []
                batch_ref_ids = []
                batch_target_ids = []
                batch_source_lengths = []
                batch_ref_lengths = []
                batch_target_lengths = []

    '''
    num_batches = ceil(len(batch_source_ids) / batch_size)

    for i in xrange(num_batches):
        yield (batch_source_ids[i, i+batch_size], 
              batch_ref_ids[i, i+batch_size], 
              batch_target_ids[i, i+batch_size], 
              batch_source_lengths[i, i+batch_size], 
              batch_ref_lengths[i, i+batch_size], 
              batch_target_lengths[i, i+batch_size])    
    '''

def keras_generate_batch(batch_size, word_to_id, start_id, end_id, unk_id):
    generator = generate_batch(batch_size, word_to_id, start_id, end_id, unk_id)

    while 1:
        source_ids, ref_ids, target_ids, source_len, ref_len, target_len = next(generator)

        encoder_input_data = np.array(pad_sequences(source_ids, maxlen=args.max_encoder_tokens, padding='post', value=mask_id))
        decoder_input_data = np.array(pad_sequences(ref_ids, maxlen=args.max_decoder_tokens, padding='post', value=mask_id))
        decoder_target_data = np.array(pad_sequences(target_ids, maxlen=args.max_decoder_tokens, padding='post', value=mask_id))

        one_hot_decoder_target_data = np.zeros((args.batch_size, args.max_decoder_tokens, vocab_size))
        for i in xrange(decoder_target_data.shape[0]):
            for j in xrange(args.max_decoder_tokens):
                one_hot_decoder_target_data[i, j, decoder_target_data[i][j]] = 1

        yield ([source_ids, ref_ids], one_hot_decoder_target_data)

if __name__ == '__main__':
    from load_sent_embeddings import load_sentence_embeddings
    word_to_id, idx_to_word, _, start_id, end_id, unk_id  = load_sentence_embeddings()
    print(start_id, end_id, unk_id)
    for source_ids, ref_ids, target_ids, source_lengths, ref_lengths, target_lengths in generate_batch(2, word_to_id, start_id, end_id, unk_id):
        print(source_ids)
        print(ref_ids)
        print(target_ids)
        print(source_lengths)
        print(ref_lengths)
        print(target_lengths)

