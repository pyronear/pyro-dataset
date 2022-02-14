# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import unittest
from pyrodataset.parse_annotations import AnnotationParser, splitStates, pickFrames
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from functools import partial


class TestFunctions(unittest.TestCase):
    """
    Test auxiliary functions used by AnnotationParser
    """
    def test_splitStates1(self):
        "Split states with start and endpoints"
        a = pd.DataFrame({'fire': [0, 1, 1, 1],
                          'clf_confidence': [1, 0, 1, 1],
                          'loc_confidence': [0, 0, 0, 0],
                          'splitEnd': None,
                          'frame': [0, 100, 200, 500]})

        states_a = pd.DataFrame({'fire': [0, 1, 1],
                                 'clf_confidence': [1, 0, 1],
                                 'loc_confidence': [0, 0, 0],
                                 'stateStart': [0, 100, 200],
                                 'stateEnd': [99.0, 199.0, 500.0]})

        pd.testing.assert_frame_equal(splitStates(a), states_a)

    def test_splitStates2(self):
        "Split states without endpoint"
        b = pd.DataFrame({'fire': [0, 1, 1, 1],
                          'clf_confidence': [1, 0, 1, 0],
                          'loc_confidence': [0, 0, 0, 0],
                          'splitEnd': 700,
                          'frame': [0, 100, 200, 500]})

        states_b = pd.DataFrame({'fire': [0, 1, 1, 1],
                                 'clf_confidence': [1, 0, 1, 0],
                                 'loc_confidence': [0, 0, 0, 0],
                                 'stateStart': [0, 100, 200, 500],
                                 'stateEnd': [99.0, 199.0, 499.0, 700.0]})

        pd.testing.assert_frame_equal(splitStates(b), states_b)

    def test_pickFrames(self):
        states_a = pd.DataFrame({'fire': [0, 1, 1],
                                 'clf_confidence': [1, 0, 1],
                                 'loc_confidence': [0, 0, 0],
                                 'stateStart': [0, 100, 200],
                                 'stateEnd': [99.0, 199.0, 500.0]})

        frames = pd.DataFrame([[0, 49, 99], [100, 149, 199], [200, 350, 500]]).astype('int32')
        x = states_a.apply(partial(pickFrames, nFrames=3, random=False), axis=1)
        pd.testing.assert_frame_equal(x, frames)


def setupTester(cls):
    """
    Setup tester for AnnotationParser. Also used by test_checkAnnotation
    Download movies if needed and create 2 instances of AnnotationParser
    from the json files in this directory
    """
    parent = Path(__file__).parent
    movies_dir = parent / 'movies'
    inputJson = parent / 'test_3_videos.json'
    inputJson_only_exploitable = parent / 'test_3_videos_only_exploitable.json'
    inputJson_only_non_exploitable = parent / 'test_only_non_exploitable.json'
    # TODO: maybe better to use a tmp directory, but couldn't make it work
    cls.inputdir = movies_dir
    if not movies_dir.exists():
        cls.inputdir.mkdir()
        import urllib
        import yaml
        import pafy
        yamlFile = "https://gist.githubusercontent.com/blenzi/d01fb4bf68256ed05ecbb11df226d0f2/raw"

        with urllib.request.urlopen(yamlFile) as yF:
            URLs = yaml.safe_load(yF)
        for dest, url in URLs.items():
            vid = pafy.new(url)
            stream = vid.getbest()
            print(f'Downloading {stream.get_filesize()/1e6:.2f} MB')
            # youtube-dl behind pafy does not support Pathlib objects, str is okay
            stream.download(filepath=str(cls.inputdir / dest))

    Parser = AnnotationParser
    cls.parser = Parser(inputJson, inputdir=cls.inputdir)
    cls.parser_only_exploitable = Parser(inputJson_only_exploitable, inputdir=cls.inputdir)
    cls.parser_only_non_exploitable = Parser(inputJson_only_non_exploitable, inputdir=cls.inputdir)

class TestAnnotationParser(unittest.TestCase):
    """
    Test parseAnnotations
    """
    @classmethod
    def setUpClass(cls):
        "Setup once for all tests"
        setupTester(cls)

    def test_columns_are_right(self):
        for col_name in self.parser.labels['aname']:
            if col_name != 'spatial':  # spatial is expected to be dropped during the parsing
                self.assertIn(col_name, self.parser.keypoints.columns)

        # When all video are exploitable, is the column correctly created ?
        for col_name in self.parser_only_exploitable.labels['aname']:
            if col_name != 'spatial':  # spatial is expected to be dropped during the parsing
                self.assertIn(col_name, self.parser_only_exploitable.keypoints.columns)

    def test_files(self):
        files = '10.mp4', '19_seq0_591.mp4', '19_seq598_608.mp4'
        np.testing.assert_array_equal(self.parser.files.fname, files)

    def test_keypoints(self):
        ref_keypoints = pd.DataFrame({
            'fname': ['10.mp4'] * 8 + ['19_seq0_591.mp4'],
            'fire': ['1', '1', '1', '1', '0', '0', '1', '1', '1'],
            'sequence': ['0', '0', '1', '1', '2', '2', '3', '3', '0'],
            'clf_confidence': ['1', '1', '1', '1', '1', '1', '1', '0', '0'],
            'loc_confidence': ['2', '2', '0', '0', '0', '0', '2', '0', '2'],
            'x': [598.974, 609.231, 873.846, 869.744, 724.103, 543.59, 939.487, 957.949, 568.205],
            'y': [467.692, 463.59, 500.513, 506.667, 449.231, 418.462, 244.103, 237.949, 358.974],
            't': [1.32, 2.826, 15.564, 18.637, 19.779, 20.057, 28.191, 38.907, 2.261],
            'frame': [33.0, 71.0, 389.0, 466.0, 494.0, 501.0, 705.0, 973.0, 57.0]})
        keypoints = self.parser.keypoints[ref_keypoints.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(ref_keypoints, keypoints)

    def test_states(self):
        pass

    def test_writeCsv(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.parser.writeCsv(tmpdirname)
            tmpdir = Path(tmpdirname)
            basename = Path(self.parser.fname).name
            keypointFile = (tmpdir / basename).with_suffix('.keypoints.csv')
            self.assertTrue(keypointFile.exists())
            statesFile = (tmpdir / basename).with_suffix('.states.csv')
            self.assertTrue(statesFile.exists())
            # TODO: test reading csv and comparing with original

    def test_writeFrames(self):
        "Test writing frames and check if directory containg only the images and csv file"
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.parser.writeFrames(tmpdirname, nFrames=2, random=False)
            tmpdir = Path(tmpdirname)
            csvFile = tmpdir / 'test_3_videos.labels.csv'
            self.assertTrue(csvFile.exists())
            labels = pd.read_csv(csvFile)
            self.assertEqual(len(labels), 10)  # number of frames
            for file in labels.imgFile:
                assert (tmpdir / file).exists(), f'{file} not found'
            self.assertEqual(len(list(tmpdir.iterdir())), 11)  # csv + 10 images


if __name__ == '__main__':
    unittest.main()
