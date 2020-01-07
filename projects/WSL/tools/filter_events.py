# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import tensorflow as tf
import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--event", help="event file", required=True)

    return parser.parse_args()


def main(args):
    out_path = os.path.join(os.path.dirname(args.event), "filtered_events")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    tf.compat.v1.disable_eager_execution()
    # writer = tf.summary.create_file_writer(out_path)
    writer = tf.compat.v1.summary.FileWriter(out_path)

    total = None
    # Pre-compute total number of events (takes lot of time)
    # total = 0
    # for event in tf.train.summary_iterator(args.event):
    #     total += 1

    # Working with Summaries and Events: https://stackoverflow.com/a/45023082/1705970
    # https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/core/util/event.proto
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto
    for event in tqdm.tqdm(tf.compat.v1.train.summary_iterator(args.event), total=total):
        event_type = event.WhichOneof("what")
        if event_type != "summary":
            writer.add_event(event)
        else:
            wall_time = event.wall_time
            step = event.step

            # possible types: simple_value, image, histo, audio
            filtered_values = [
                value for value in event.summary.value if value.HasField("simple_value")
            ]
            summary = tf.compat.v1.Summary(value=filtered_values)

            filtered_event = tf.compat.v1.summary.Event(
                summary=summary, wall_time=wall_time, step=step
            )
            writer.add_event(filtered_event)
    writer.close()
    return 0


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
