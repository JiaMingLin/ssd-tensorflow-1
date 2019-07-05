#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   30.08.2017
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

import lxml.etree
import random
import math
import cv2
import os

import numpy as np

from utils import Label, Box, Sample, Size
from utils import rgb2bgr, abs2prop
from glob import glob
from tqdm import tqdm

#-------------------------------------------------------------------------------
# Labels
#-------------------------------------------------------------------------------
actions = ['Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 
'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 
'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 
'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 
'WalkingWithDog']

label_defs = [
    Label('Basketball',         rgb2bgr((0,     0,   0))),
    Label('BasketballDunk',     rgb2bgr((111,  74,   0))),
    Label('Biking',             rgb2bgr(( 81,   0,  81))),
    Label('CliffDiving',        rgb2bgr((128,  64, 128))),
    Label('CricketBowling',     rgb2bgr((244,  35, 232))),
    Label('Diving',             rgb2bgr((230, 150, 140))),
    Label('Fencing',            rgb2bgr(( 70,  70,  70))),
    Label('FloorGymnastics',    rgb2bgr((102, 102, 156))),
    Label('GolfSwing',          rgb2bgr((190, 153, 153))),
    Label('HorseRiding',        rgb2bgr((150, 120,  90))),
    Label('IceDancing',         rgb2bgr((153, 153, 153))),
    Label('LongJump',           rgb2bgr((250, 170,  30))),
    Label('PoleVault',          rgb2bgr((220, 220,   0))),
    Label('RopeClimbing',       rgb2bgr((107, 142,  35))),
    Label('SalsaSpin',          rgb2bgr(( 52, 151,  52))),
    Label('SkateBoarding',      rgb2bgr(( 70, 130, 180))),
    Label('Skiing',             rgb2bgr((220,  20,  60))),
    Label('Skijet',             rgb2bgr((  0,   0, 142))),
    Label('SoccerJuggling',     rgb2bgr((  0,   0, 230))),
    Label('Surfing',            rgb2bgr((119,  11,  32))),
    Label('TennisSwing',        rgb2bgr((0,  11,  32))),
    Label('TrampolineJumping',  rgb2bgr((119,  0,  32))),
    Label('VolleyballSpiking',  rgb2bgr((11,  156,  5))),
    Label('WalkingWithDog',     rgb2bgr((81,  81,  0))),

    ]

#-------------------------------------------------------------------------------
class UCF24Source:
    #---------------------------------------------------------------------------
    def __init__(self):
        self.num_classes   = len(label_defs)
        self.colors        = {l.name: l.color for l in label_defs}
        self.lid2name      = {i: l.name for i, l in enumerate(label_defs)}
        self.lname2id      = {l.name: i for i, l in enumerate(label_defs)}
        self.num_train     = 0
        self.num_valid     = 0
        self.num_test      = 0
        self.train_samples = []
        self.valid_samples = []
        self.test_samples  = []

    #---------------------------------------------------------------------------
    def __build_annotation_list(self, root):
        """
        Build a list of samples for the VOC dataset (either trainval or test)
        """
        annot_files = []
        annot_root  = os.path.join(root, 'labels')
        for act in actions:
            # list video
            for video in os.listdir(os.path.join(annot_root, act)):
                video_label_list = os.listdir(os.path.join(annot_root, act, video))
                video_label_list = [os.path.join(annot_root, act, video, label_file) for label_file in video_label_list]
                annot_files += video_label_list
        return annot_files

    #---------------------------------------------------------------------------
    def __build_sample_list(self, root, annot_files):
        """
        Build a list of samples for the VOC dataset (either trainval or test)
        """
        image_root  = os.path.join(root, 'rgb-images')
        samples     = []
        #-----------------------------------------------------------------------
        # Process each annotated sample
        #-----------------------------------------------------------------------
        for fn in tqdm(annot_files, desc='ucf_24_frame', unit='samples'):
            act = fn.split('/')[4]
            video = fn.split('/')[5]
            frame_id = fn.split('/')[-1][:-4]
            image_path = os.path.join(image_root, act, video, '{}.jpg'.format(frame_id))

            #---------------------------------------------------------------
            # Get the file dimensions
            #---------------------------------------------------------------
            if not os.path.exists(image_path):
                continue

            img     = cv2.imread(image_path)
            imgsize = Size(img.shape[1], img.shape[0])

            #---------------------------------------------------------------
            # Get boxes for all the objects
            #---------------------------------------------------------------
            boxes = []
            with open(fn, 'r') as fin:
                objects = fin.readlines()
            for line in objects:
                line = line[:-1]
                #-----------------------------------------------------------
                # Get the properties of the box and convert them to the
                # proportional terms
                #-----------------------------------------------------------
                obj = line.split(' ')
                label = int(obj[0])-1
                xmin = int(float(obj[1]))
                ymin = int(float(obj[2]))
                xmax = int(float(obj[3]))
                ymax = int(float(obj[4]))

                center, size = abs2prop(xmin, xmax, ymin, ymax, imgsize)
                box = Box(self.lid2name[label], label, center, size)
                boxes.append(box)
            if not boxes:
                continue
            sample = Sample(image_path, boxes, imgsize)
            samples.append(sample)

        return samples

    def __filter_sample_set(self, total_samples, filter_list):

        result = []
        for sample in total_samples:
            act = sample.filename.split('/')[4]
            video = sample.filename.split('/')[5]
            v_str = '{}/{}'.format(act, video)
            if v_str in filter_list:
                result.append(sample)
        return result


    #---------------------------------------------------------------------------
    def load_trainval_data(self, data_dir, valid_fraction):
        """
        Load the training and validation data
        :param data_dir:       the directory where the dataset's file are stored   ../data/ucf24/
                               rgb-images
                               labels
                               splitfiles
        :param valid_fraction: what franction of the dataset should be used
                               as a validation sample
        """

        #-----------------------------------------------------------------------
        # Process the samples defined in the relevant file lists
        #-----------------------------------------------------------------------
        train_annot = []
        train_samples = []
        train_annot = self.__build_annotation_list(data_dir)[:1000]
        train_samples = self.__build_sample_list(data_dir, train_annot)

        #-----------------------------------------------------------------------
        # Split train and validation
        #-----------------------------------------------------------------------
        # read the train and val split
        splitfiles = os.path.join(data_dir, 'splitfiles')
        with open(os.path.join(splitfiles, 'trainlist01.txt')) as fin:
            train_file_list = [line[:-1] for line in fin.readlines()]

        with open(os.path.join(splitfiles, 'testlist01.txt')) as fin:
            test_file_list = [line[:-1] for line in fin.readlines()]
        self.valid_samples = self.__filter_sample_set(train_samples, test_file_list)
        self.train_samples = self.__filter_sample_set(train_samples, train_file_list)

        #-----------------------------------------------------------------------
        # Final set up and sanity check
        #-----------------------------------------------------------------------
        if len(self.train_samples) == 0:
            raise RuntimeError('No training samples found in ' + data_dir)

        if valid_fraction > 0:
            if len(self.valid_samples) == 0:
                raise RuntimeError('No validation samples found in ' + data_dir)

        self.num_train = len(self.train_samples)
        self.num_valid = len(self.valid_samples)

    #---------------------------------------------------------------------------
    def load_test_data(self, data_dir):
        """
        Load the test data
        :param data_dir: the directory where the dataset's file are stored
        """
        root = data_dir + '/test/VOCdevkit/VOC2012'
        annot = self.__build_annotation_list(root, 'test')
        self.test_samples  = self.__build_sample_list(root, annot,
                                                      'test_VOC2012')

        if len(self.test_samples) == 0:
            raise RuntimeError('No testing samples found in ' + data_dir)

        self.num_test  = len(self.test_samples)

#-------------------------------------------------------------------------------
def get_source():
    return UCF24Source()
