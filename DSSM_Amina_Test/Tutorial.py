from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import math
import numpy as np
import os

import cntk as C
# import cntk.tests.test_utils
# cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

location = os.path.normpath('data/DSSM')
data = {
  'train': { 'file': 'train.pair.tok.ctf' },
  'val':{ 'file': 'valid.pair.tok.ctf' },
  'query': { 'file': 'vocab_Q.wl' },
  'answer': { 'file': 'vocab_A.wl' }
}

import requests

def download(url, filename):
    """ utility function to download a file """
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        for data in response.iter_content():
            handle.write(data)

if not os.path.exists(location):
    os.mkdir(location)

for item in data.values():
    path = os.path.normpath(os.path.join(location, item['file']))

    if os.path.exists(path):
        print("Reusing locally cached:", path)

    else:
        print("Starting download:", item['file'])
        url = "http://www.cntk.ai/jup/dat/DSSM/%s.csv"%(item['file'])
        print(url)
        download(url, path)
        print("Download completed")
    item['file'] = path

    # Define the vocabulary size (QRY-stands for question and ANS stands for answer)
    QRY_SIZE = 1204
    ANS_SIZE = 1019


    def create_reader(path, is_training):
        return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
            query=C.io.StreamDef(field='S0', shape=QRY_SIZE, is_sparse=True),
            answer=C.io.StreamDef(field='S1', shape=ANS_SIZE, is_sparse=True)
        )), randomize=is_training, max_sweeps=C.io.INFINITELY_REPEAT if is_training else 1)

train_file = data['train']['file']
print(train_file)

if os.path.exists(train_file):
    train_source = create_reader(train_file, is_training=True)
else:
    raise ValueError("Cannot locate file {0} in current directory {1}".format(train_file, os.getcwd()))

validation_file = data['val']['file']
print(validation_file)
if os.path.exists(validation_file):
    val_source = create_reader(validation_file, is_training=False)
else:
    raise ValueError("Cannot locate file {0} in current directory {1}".format(validation_file, os.getcwd()))

# Create the containers for input feature (x) and the label (y)
qry = C.sequence.input_variable(QRY_SIZE)
ans = C.sequence.input_variable(ANS_SIZE)
# Create the containers for input feature (x) and the label (y)
axis_qry = C.Axis.new_unique_dynamic_axis('axis_qry')
qry = C.sequence.input_variable(QRY_SIZE, sequence_axis=axis_qry)

axis_ans = C.Axis.new_unique_dynamic_axis('axis_ans')
ans = C.sequence.input_variable(ANS_SIZE, sequence_axis=axis_ans)


EMB_DIM   = 25 # Embedding dimension
HIDDEN_DIM = 50 # LSTM dimension
DSSM_DIM = 25 # Dense layer dimension
NEGATIVE_SAMPLES = 5
DROPOUT_RATIO = 0.2


def create_model(qry, ans):
    with C.layers.default_options(initial_state=0.1):
        qry_vector = C.layers.Sequential([
            C.layers.Embedding(EMB_DIM, name='embed'),
            C.layers.Recurrence(C.layers.LSTM(HIDDEN_DIM), go_backwards=False),
            C.sequence.last,
            C.layers.Dense(DSSM_DIM, activation=C.relu, name='q_proj'),
            C.layers.Dropout(DROPOUT_RATIO, name='dropout qdo1'),
            C.layers.Dense(DSSM_DIM, activation=C.tanh, name='q_enc')
        ])

        ans_vector = C.layers.Sequential([
            C.layers.Embedding(EMB_DIM, name='embed'),
            C.layers.Recurrence(C.layers.LSTM(HIDDEN_DIM), go_backwards=False),
            C.sequence.last,
            C.layers.Dense(DSSM_DIM, activation=C.relu, name='a_proj'),
            C.layers.Dropout(DROPOUT_RATIO, name='dropout ado1'),
            C.layers.Dense(DSSM_DIM, activation=C.tanh, name='a_enc')
        ])

    return {
        'query_vector': qry_vector(qry),
        'answer_vector': ans_vector(ans)
    }

# Create the model and store reference in `network` dictionary
network = create_model(qry, ans)

network['query'], network['axis_qry'] = qry, axis_qry
network['answer'], network['axis_ans'] = ans, axis_ans


def create_loss(vector_a, vector_b):
    qry_ans_similarity = C.cosine_distance_with_negative_samples(vector_a, \
                                                                 vector_b, \
                                                                 shift=1, \
                                                                 num_negative_samples=5)
    return 1 - qry_ans_similarity


# Model parameters
MAX_EPOCHS = 5
EPOCH_SIZE = 10000
MINIBATCH_SIZE = 50

# Create trainer
def create_trainer(reader, network):

    # Setup the progress updater
    progress_writer = C.logging.ProgressPrinter(tag='Training', num_epochs=MAX_EPOCHS)

    # Set learning parameters
    lr_per_sample     = [0.0015625]*20 + \
                        [0.00046875]*20 + \
                        [0.00015625]*20 + \
                        [0.000046875]*10 + \
                        [0.000015625]
    lr_schedule       = C.learning_parameter_schedule_per_sample(lr_per_sample, \
                                                 epoch_size=EPOCH_SIZE)
    mms               = [0]*20 + [0.9200444146293233]*20 + [0.9591894571091382]
    mm_schedule       = C.learners.momentum_schedule(mms, \
                                                     epoch_size=EPOCH_SIZE, \
                                                     minibatch_size=MINIBATCH_SIZE)
    l2_reg_weight     = 0.0002

    model = C.combine(network['query_vector'], network['answer_vector'])

    #Notify the network that the two dynamic axes are indeed same
    query_reconciled = C.reconcile_dynamic_axes(network['query_vector'], network['answer_vector'])

    network['loss'] = create_loss(query_reconciled, network['answer_vector'])
    network['error'] = None

    print('Using momentum sgd with no l2')
    dssm_learner = C.learners.momentum_sgd(model.parameters, lr_schedule, mm_schedule)

    network['learner'] = dssm_learner

    print('Using local learner')
    # Create trainer
    return C.Trainer(model, (network['loss'], network['error']), network['learner'], progress_writer)

# Instantiate the trainer
trainer = create_trainer(train_source, network)


# Train
def do_train(network, trainer, train_source):
    # define mapping from intput streams to network inputs
    input_map = {
        network['query']: train_source.streams.query,
        network['answer']: train_source.streams.answer
        }

    t = 0
    for epoch in range(MAX_EPOCHS):         # loop over epochs
        epoch_end = (epoch+1) * EPOCH_SIZE
        while t < epoch_end:                # loop over minibatches on the epoch
            data = train_source.next_minibatch(MINIBATCH_SIZE, input_map= input_map)  # fetch minibatch
            trainer.train_minibatch(data)               # update model with it
            t += MINIBATCH_SIZE

        trainer.summarize_training_progress()

do_train(network, trainer, train_source)