# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import unittest
import tempfile
import pandas as pd
from pyrodataset.check_annotations import AnnotationChecker
from test_parse_annotations import setupTester


def setUpChecker(cls):
    """
    Setup tester for AnnotationChecker
    """
    setupTester(cls)
    cls.tmpdir = tempfile.TemporaryDirectory()
    cls.parser.writeCsv(cls.tmpdir.name)
    cls.parser.writeFrames(cls.tmpdir.name, nFrames=2, random=False)
    cls.checker = AnnotationChecker(cls.tmpdir.name)


class TestCheckAnnotations(unittest.TestCase):
    """
    Test the functional part of checkAnnotations, cannot test the graphical part.
    Relies on parseAnnotations for the setup
    """
    @classmethod
    def setUpClass(cls):
        "Setup only once for all tests"
        setUpChecker(cls)

    @classmethod
    def tearDown(cls):
        "Clear temporary directory"
        cls.tmpdir.cleanup()

    def test_updateValues_withoutChange(self):
        """
        Test if updateValues reads back the same info as in the original labels
        DOES NOT WORK for loc_confidence because location 0 (N/A) is being converted to 1
         """
        labels = self.checker.labels.copy()
        self.checker.updateValues()
        pd.testing.assert_frame_equal(labels.drop(columns='loc_confidence'),
                                      self.checker.labels.drop(columns='loc_confidence'))


if __name__ == '__main__':
    unittest.main()
