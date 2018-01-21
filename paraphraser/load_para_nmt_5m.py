from nlp_pipeline import nlp_pipeline

def tokens_to_id(tokens, word_to_id, unk_id): #, src_or_ref=, unk_id, start_id, end_id):
    return [ word_to_id.get(token, unk_id) for token in tokens ] 

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

            source_tokens = nlp_pipeline(source.strip())
            ref_tokens = nlp_pipeline(ref.strip())

            source_ids = tokens_to_id(source_tokens, word_to_id, unk_id)
            target_ids = tokens_to_id(ref_tokens, word_to_id, unk_id) + [end_id]
            ref_ids = [start_id] + target_ids

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

if __name__ == '__main__':
    from load_sent_embeddings import load_sentence_embeddings
    word_to_id, idx_to_word, _, start_id, end_id, unk_id  = load_sentence_embeddings()
    #print(start_id, end_id, unk_id)
    for source_ids, ref_ids, target_ids, source_lengths, ref_lengths, target_lengths in generate_batch(2, word_to_id, 20, start_id, end_id, unk_id):
        print(source_ids)
        print(ref_ids)
        print(target_ids)
        print(source_lengths)
        print(ref_lengths)
        print(target_lengths)

