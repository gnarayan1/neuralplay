import argparse
import tensorflow as tf

import my_data
import my_estimator
import sys

BATCH_SIZE=100
ITERATIONS=10000
HIDDEN_UNITS=[10,10,10]

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size')
parser.add_argument('--train_steps', default=ITERATIONS, type=int,
                    help='number of training steps')



def main(argv):
    args = parser.parse_args(argv[1:])

    my_feature_columns = []
    for key in my_data.CSV_COLUMN_NAMES[:-1]:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    print(my_feature_columns)

    classifier = tf.estimator.DNNClassifier(model_dir='/tmp/train229', warm_start_from='/tmp/train229', hidden_units=my_estimator.HIDDEN_UNITS,
        feature_columns=my_feature_columns,  n_classes=5)

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