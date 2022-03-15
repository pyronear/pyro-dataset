import unittest
import os
import pickle
from pyrodataset.video_splitter import VideoSplitter


def setupTester(cls):
    """
    Prepare tester class for VideoSplitter
    """
    import urllib.request
    import yaml
    # Test parameters
    url = 'https://gist.githubusercontent.com/blenzi/82746e11119cb88a67603944869e29e2/raw'
    cls.ref = eval(urllib.request.urlopen(url).read())

    # Stream
    if not os.path.exists(cls.ref['fname']):
        import pafy
        vid = pafy.new(cls.ref['url'])
        stream = vid.getbest()
        print(f'Downloading {stream.get_filesize()/1e6:.2f} MB')
        stream.download(filepath=cls.ref['fname'])
    cls.fname = cls.ref['fname']

    # Ref captions
    yamlFile = "https://gist.github.com/blenzi/02027e8973d79cd89bc601b119d2a190/raw"
    with urllib.request.urlopen(yamlFile) as yF:
        cls.captions = yaml.safe_load(yF)


class VideoTester(unittest.TestCase):
    """
    Test VideoSplitter
    """
    @classmethod
    def setUpClass(cls):
        "Setup only once for all tests"
        setupTester(cls)
        cls.splitter = VideoSplitter(cls.fname)
        cls.testFindSequences = False  # skip finding sequences (takes about 30s)

    def a_test_loadFrame(self):  # call it a_ as they are executed in alphabetical order
        frame = self.splitter.loadFrame(self.ref['extract']['frame'])
        self.assertEqual(len(frame.shape), 3)

    def test_analyseFrame(self):
        # TODO: compare caption and coordinates with expected values (modulo OCR problems)
        frame_index = self.ref['extract']['frame']
        self.splitter.processFrame(frame_index)
        self.assertIn(frame_index, self.splitter.captions)
        self.assertIn(frame_index, self.splitter.coordinates)

    def test_findSequences(self):
        "Test frame range in sequences (ignore exact coordinates)"
        if not self.testFindSequences:
            return
        self.maxDiff = None
        self.splitter.findSequences()
        seqs = self.splitter.sequences
        inv_seqs = dict(map(reversed, seqs.items()))  # invert keys and values
        self.assertEqual(inv_seqs.keys(), self.ref['sequences'].keys())

    def test_writeSequences(self):
        "Test writing movie sequences"
        if not self.testFindSequences:
            return
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.splitter.writeSequences(tmpdirname, min_frames=0)
            basename, ext = os.path.splitext(os.path.basename(self.splitter.fname))
            for fmin, fmax in self.splitter.sequences.values():
                fname = os.path.join(tmpdirname, f'{basename}_seq{fmin}_{fmax}{ext}')
                self.assertTrue(os.path.exists(fname))

    def test_writeInfo(self):
        "Test writing dictionaries with captions, sequences, ..."
        import tempfile
        basename, ext = os.path.splitext(os.path.basename(self.splitter.fname))
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.splitter.writeInfo(tmpdirname)
            names = 'captions', 'coordinates', 'seqID', 'sequences'
            dicts = [getattr(self.splitter, name) for name in names]
            self.assertTrue(any(dicts))
            for name, d in zip(names, dicts):
                fname = os.path.join(tmpdirname, f'{basename}_{name}.pickle')
                with open(fname, 'rb') as pickleFile:
                    dSaved = pickle.load(pickleFile)
                    self.assertEqual(d, dSaved)


class VideoTesterWithCaptions(VideoTester):
    """
    Test VideoSplitter with captions loaded externally
    """
    @classmethod
    def setUpClass(cls):
        "Setup only once for all tests"
        setupTester(cls)
        cls.splitter = VideoSplitter(cls.fname, cls.captions)
        cls.testFindSequences = True

    def test_loadCaptions(self):
        "Test loadCaption from pickle file"
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdirname:
            basename, ext = os.path.splitext(os.path.basename(self.splitter.fname))
            fname = os.path.join(tmpdirname, f'{basename}_captions.pickle')
            with open(fname, 'wb') as pickleFile:
                pickle.dump(self.captions, pickleFile)

            # Load from fname
            self.splitter.loadCaptions(fname)
            self.assertEqual(self.captions, self.splitter.captions)

            # Load from directory name
            self.splitter.loadCaptions(tmpdirname)
            self.assertEqual(self.captions, self.splitter.captions)
