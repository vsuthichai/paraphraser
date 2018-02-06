import logging
import tensorflow as tf

def summarize_scalar(writer, tag, value, step):
    """Prepare data to be written to protobuf event file.  This is later
    read into tensorboard for visualization.

    Args:
        writer: summary writer
        tag: identifier name of the the data in question
        value: the value the data takes on
        step: global step during training
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(summary, step)


def debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab):
    """Debug dataset batch samples to ensure they take on intended avlues"""
    logging.info("==============================================================")
    logging.info("SOURCE!")
    #logging.info(seq_source_ids)
    for source_ids in seq_source_ids:
        logging.info(' '.join([id_to_vocab[source_id] for source_id in source_ids]))
    logging.info(seq_source_len)
    logging.info("REFERENCE!")
    #logging.info(seq_ref_ids)
    for i in seq_ref_ids:
        logging.info(' '.join([id_to_vocab[label] for label in i if label != -1]))
    logging.info(seq_ref_len)
    logging.info("==============================================================")

def dataset_config():
    """Dataset configuration.  Dataset files are grouped by sentences of maximum
    length for train, dev, and test.  """

    dataset = [
        { 
            'maxlen': 5,
            'train': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.train.5',
            'dev': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.dev.5',
            'test': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.test.5' 
        },
        { 
            'maxlen': 10,
            'train': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.train.10',
            'dev': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.dev.10',
            'test': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.test.10' 
        },
        { 
            'maxlen': 20,
            'train': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.train.20',
            'dev': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.dev.20',
            'test': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.test.20' 
        },
        { 
            'maxlen': 30,
            'train': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.train.30',
            'dev': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.dev.30',
            'test': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.test.30' 
        },
        { 
            'maxlen': 40,
            'train': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.train.40',
            'dev': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.dev.40',
            'test': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.test.40' 
        },
        { 
            'maxlen': 50,
            'train': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.train.50',
            'dev': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.dev.50',
            'test': '/media/sdb/datasets/aggregate_paraphrase_corpus_0/dataset.test.50' 
        }
    ]

    return dataset

