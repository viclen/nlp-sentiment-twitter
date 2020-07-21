import official.nlp.modeling.networks
import official.nlp.modeling.models
import official.nlp.modeling.losses
import official.nlp.data.classifier_data_lib
import official.nlp.bert.tokenization
import official.nlp.bert.run_classifier
import official.nlp.bert.configs
import official.nlp.bert.bert_models
import official.nlp.optimization
from official.nlp import bert
from official import nlp
from official.modeling import tf_utils
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import json

gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"

tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
    do_lower_case=True
)


def encode_sentence(s, tokenizer):
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(glue_dict, tokenizer):
    num_examples = len(glue_dict["sentence1"])

    sentence1 = tf.ragged.constant([
        encode_sentence(s, tokenizer) for s in np.array(glue_dict["sentence1"])
    ])
    sentence2 = tf.ragged.constant([
        encode_sentence(s, tokenizer) for s in np.array(glue_dict["sentence2"])
    ])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat(
        [type_cls, type_s1, type_s2], axis=-1
    ).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    }

    return inputs


# Set up tokenizer to generate Tensorflow dataset
tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
    do_lower_case=True
)

my_examples = bert_encode(
    glue_dict={
        'sentence1': [
            'The rain in Spain falls mainly on the plain.',
            'Look I fine tuned BERT.'
        ],
        'sentence2': [
            'It mostly rains on the flat lands of Spain.',
            'Is it working? This does not match.'
        ]
    },
    tokenizer=tokenizer
)

export_dir = './saved_model'
reloaded = tf.saved_model.load(export_dir)
result = reloaded([
    my_examples['input_word_ids'],
    my_examples['input_mask'],
    my_examples['input_type_ids']
], training=False)

bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config, num_labels=2)
bert_classifier(my_examples, training=False)

print(result)
