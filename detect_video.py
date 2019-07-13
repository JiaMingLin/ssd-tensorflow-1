#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   05.02.2018
#-------------------------------------------------------------------------------
# This file is part of SSD-TensorFlow.
#
# SSD-TensorFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SSD-TensorFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SSD-Tensorflow.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------

import tensorflow as tf
import argparse
import pickle
import numpy as np
import sys
import cv2
import os
import imageio

from ssdutils import get_anchors_for_preset, decode_boxes, suppress_overlaps
from utils import draw_box
from tqdm import tqdm

if sys.version_info[0] < 3:
    print("This is a Python 3 program. Use Python 3 or higher.")
    sys.exit(1)

def run(files, img_input_tensor, result_tensor, batch_size = 32):
    output_imgs = []

    for i in range(0, len(files), batch_size):
        batch_names = files[i:i+batch_size]
        batch_imgs = []
        batch = []
        for f in batch_names:
            img = cv2.imread(f)
            batch_imgs.append(img)
            img = cv2.resize(img, (300, 300))
            batch.append(img)

        batch = np.array(batch)
        feed = {img_input: batch}
        enc_boxes = sess.run(result, feed_dict=feed)

        for i in range(len(batch_names)):
            boxes = decode_boxes(enc_boxes[i], anchors, 0.5, lid2name, None)
            boxes = suppress_overlaps(boxes)[:200]
            name = os.path.basename(batch_names[i])

            for box in boxes:
                draw_box(batch_imgs[i], box[1], colors[box[1].label])
            output_imgs.append(batch_imgs[i])

                #with open(os.path.join(args.output_dir, name+'.txt'), 'w') as f:
                #    for box in boxes:
                #        draw_box(batch_imgs[i], box[1], colors[box[1].label])

                        #box_data = '{} {} {} {} {} {}\n'.format(box[1].label,
                        #    box[1].labelid, box[1].center.x, box[1].center.y,
                        #    box[1].size.w, box[1].size.h)
                        #f.write(box_data)
                
                #cv2.imwrite(os.path.join(args.output_dir, name),
                #            batch_imgs[i])
    return output_imgs

def save_to_video(frame_list, output_path):
    writer = imageio.get_writer(output_path, fps=25)
    for frame in frame_list:
        writer.append_data(frame)
    writer.close()


#-------------------------------------------------------------------------------
# Start the show
#-------------------------------------------------------------------------------
def main():
    #---------------------------------------------------------------------------
    # Parse the commandline
    #---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='SSD inference for video')
    parser.add_argument('--model', default='model300.pb',
                        help='model file')
    parser.add_argument('--training-data', default='training-data-300.pkl',
                        help='training data')
    parser.add_argument('--output-dir', default='test-out',
                        help='output directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    args = parser.parse_args()

    #---------------------------------------------------------------------------
    # Print parameters
    #---------------------------------------------------------------------------
    print('[i] Model:         ', args.model)
    print('[i] Training data: ', args.training_data)
    print('[i] Output dir:    ', args.output_dir)
    print('[i] Batch size:    ', args.batch_size)

    #---------------------------------------------------------------------------
    # Create the output directory
    #---------------------------------------------------------------------------
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #---------------------------------------------------------------------------
    # Load the graph and the training data
    #---------------------------------------------------------------------------
    graph_def = tf.GraphDef()
    with open(args.model, 'rb') as f:
        serialized = f.read()
        graph_def.ParseFromString(serialized)

    with open(args.training_data, 'rb') as f:
        data = pickle.load(f)
        preset = data['preset']
        colors = data['colors']
        lid2name = data['lid2name']
        anchors = get_anchors_for_preset(preset)

    with tf.Session() as sess:
        tf.import_graph_def(graph_def, name='detector')
        img_input = sess.graph.get_tensor_by_name('detector/image_input:0')
        result = sess.graph.get_tensor_by_name('detector/result/result:0')

    #---------------------------------------------------------------------------
    # Run Detection
    #---------------------------------------------------------------------------

    video_root = '../data/ucf24'
    test_output = 'detect_output'
    with open(os.path.join(video_root, 'splitfiles/testlist01.txt')) as fin:
        video_paths = [os.path.join(video_root, 'rgb-images', line[:-1]) for line in fin.readlines()]

    for video in tqdm(video_paths, total = len(video_paths)):
        # video: frame folder
        frames = os.listdir(video)
        frame_paths = [os.path.join(video, frame), for frame in frames]
        detected_frames = run(frame_paths, img_input, result, batch_size = args.batch_size)

        video_name = video.split('/')[:-2]
        video_name[-1] = video_name[-1] + '.mp4'
        video_save_path = os.path.join(video_root, test_output, *video_name)
        save_to_video(detected_frames, video_save_path)

if __name__ == '__main__':
    main()
