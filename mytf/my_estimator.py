#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the my dataset."""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import tensorflow as tf

import my_data

BATCH_SIZE=100
ITERATIONS=10000
HIDDEN_UNITS=[10,10,10]

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size')
parser.add_argument('--train_steps', default=ITERATIONS, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = my_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    print(my_feature_columns)

    # cost = tf.summary.scalar("cost", cost)
    # accuracy = tf.summary.scalar("accuracy", accuracy)
    # train_summary_op = tf.summary.merge([cost, accuracy])
    # train_writer = tf.summary.FileWriter('/tmp/train',
    #                                      graph=tf.get_default_graph())

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # hidden layers of 10 nodes each.
        hidden_units=HIDDEN_UNITS,
        # The model must choose between 5 classes.
        n_classes=5,
        model_dir='/tmp/train229',  config=tf.contrib.learn.RunConfig(
        save_checkpoints_steps=150,
        save_checkpoints_secs=None,
        save_summary_steps=100,
    ))

    # Train the Model.
    classifier.train(
        input_fn=lambda:my_data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:my_data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = [0, 3, 2]
    predict_x = {
        'Age': [20, 33, 53],
        'Code1': [8, 7, 5],
        'Code2': [1, 10, 5],
    }

    predictions = classifier.predict(
        input_fn=lambda:my_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    print(zip(predictions, expected))

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(my_data.OUTPUT[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
