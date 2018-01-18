

def generate_batch(dataset_path, batch_size, max_seq_length, vocab_to_id, embeddings):
    with open(dataset_path, "r") as f:
        for line in f:
            tokens = line.split(' ')


