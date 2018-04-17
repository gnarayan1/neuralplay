import argparse
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=20, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


feature_names = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth'
]

SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def read_my_csv(filename_queue):

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [tf.constant([], dtype=tf.float32), tf.constant([], dtype=tf.float32),
                       tf.constant([], dtype=tf.float32), tf.constant([], dtype=tf.float32),
                       tf.constant([], dtype=tf.int32)]
    col1, col2, col3, col4, label = tf.decode_csv(
        value, record_defaults=record_defaults)
    features = tf.stack([col1, col2, col3, col4])
    return features, tf.stack([label])


def my_input_fn1(filenames, perform_shuffle=False, repeat_count=1):
   def decode_csv(line):
       parsed_line = tf.decode_csv(line,  record_defaults = [[0.], [0.], [0.], [0.], [0]])
       label = parsed_line[-1:] # Last element is the label
       del parsed_line[-1] # Delete last element
       features = parsed_line # Everything (but last element) are the features
       x=zip(feature_names, features)
       print(list(x))
       d = dict(list(x)), label
       return d

   dataset = (tf.data.TextLineDataset(filenames) # Read text file
       .skip(1) # Skip header row
       .map(decode_csv)) # Transform each elem by applying decode_csv fn
   if perform_shuffle:
       # Randomizes input using a window of 256 elements (read into memory)
       dataset = dataset.shuffle(buffer_size=256)
   dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
   dataset = dataset.batch(32)  # Batch size to use
   iterator = dataset.make_one_shot_iterator()
   batch_features, batch_labels = iterator.get_next()
   return batch_features, batch_labels


def my_input_fn(filenames, is_shuffle=False, repeat_count=1):
    dataset = tf.data.TextLineDataset(filenames).skip(1)  # filename is a list

    def parser(record):
        keys_to_features = {
            'label': tf.FixedLenFeature((), dtype=tf.int64),
            'features': tf.FixedLenFeature(shape=(4,), dtype=tf.float32),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        my_features = {}
        for idx, names in enumerate(feature_names):
            my_features[names] = parsed['features'][idx]
        return my_features, parsed['label']

    dataset = dataset.map(parser)
    if is_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.batch(32)
    dataset = dataset.repeat(repeat_count)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    print(features, labels)
    return features, labels





def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example, label = read_my_csv(filename_queue)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 30
  capacity = min_after_dequeue + 3 * batch_size
  # example_batch, label_batch = tf.train.shuffle_batch(
  #     [example, label], batch_size=batch_size, capacity=capacity,
  #     min_after_dequeue=min_after_dequeue)
  # return example_batch, label_batch
  return tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)



def main(argv):
    args = parser.parse_args(argv[1:])
    batch_size=args.batch_size

    # train_filename_queue = tf.train.string_input_producer(
    #     ["/Users/geeth/Downloads/iris_training.csv"], num_epochs=None, shuffle=True)
    # (train_x, train_y) = read_my_csv(train_filename_queue)
    #
    # test_filename_queue = tf.train.string_input_producer(
    #     ["/Users/geeth/Downloads/iris_test.csv"], num_epochs=None, shuffle=True)
    # (test_x, test_y) = read_my_csv(test_filename_queue)

    # Feature columns describe how to use the input.
    feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3)

    # Train the Model.
    classifier.train(input_fn=lambda: my_input_fn1(filenames=["/Users/geeth/Downloads/iris_training.csv"], perform_shuffle=False, repeat_count=1))
    # classifier.train(input_fn=lambda: my_input_fn(filenames="/Users/geeth/Downloads/iris_training.csv", is_shuffle=False, repeat_count=1))

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda: my_input_fn(filenames=["/Users/geeth/Downloads/iris_test.csv"], is_shuffle=True,
                                     repeat_count=100))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(predict_x,
                                                 labels=None,
                                                 batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)