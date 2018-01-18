
def generate_batch(batch_size, max_seq_length, vocab_to_id, embeddings):
    vocab_size, hidden_size = embeddings.shape
    start_token_id = vocab_size - 2
    end_token_id = vocab_size - 1

    s1_words = ['she', 'eat', 'pizza', 'at', 'the', 'restaurant']
    s1_source = [ vocab_to_id[word] for word in s1_words ]
    #s1_source = [4, 39, 9, 10, 15, 3, 1, 4903, 43893]
    s1_ref_words = ['she', 'loves', 'eating', 'pizza', 'at', 'the', 'diner']
    s1_ref = [ vocab_to_id[word] for word in s1_ref_words ]
    s1_ref.insert(0, start_token_id)
    # s1_ref = [start_token_id, 23, 4390, 392, 34893, 343]
    s1_out = s1_ref[1:]
    s1_out.append(end_token_id)

    s1_source_length = len(s1_source)
    s1_source = s1_source + ((max_seq_length - s1_source_length) * [-1])
    s1_ref_length = len(s1_ref)
    s1_ref = s1_ref + ((max_seq_length - s1_ref_length) * [-1])
    s1_out = s1_out + ((max_seq_length - s1_ref_length) * [-1])

    s2_words = ['wow', 'the', 'plane', 'flew', 'to', 'europe']
    s2_source = [ vocab_to_id[word] for word in s2_words ]
    #s2_source = [4, 39, 9, 10, 15, 3, 1, 4903, 43893]
    s2_ref_words = ['the', 'airplane', 'left', 'the', 'airport', 'on', 'route', 'for', 'europe']
    s2_ref = [ vocab_to_id[word] for word in s2_ref_words ]
    s2_ref.insert(0, start_token_id)
    # s2_ref = [start_token_id, 23, 4390, 392, 34893, 343]
    s2_out = s2_ref[1:]
    s2_out.append(end_token_id)

    s2_source_length = len(s2_source)
    s2_source = s2_source + ((max_seq_length - s2_source_length) * [-1])
    s2_ref_length = len(s2_ref)
    s2_ref = s2_ref + ((max_seq_length - s2_ref_length) * [-1])
    s2_out = s2_out + ((max_seq_length - s2_ref_length) * [-1])

    yield ([s1_source, s2_source], [s1_ref, s2_ref], [s1_out, s2_out], [s1_source_length, s2_source_length], [s1_ref_length, s2_ref_length])
