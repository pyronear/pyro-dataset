import unittest
from pyrodataset.splitter_utils import extract_coordinates, extract_timestamp


class UtilsTester(unittest.TestCase):
    """
    Test caption manipulation
    """
    def setUp(self):
        import urllib
        import imutils
        url = 'https://gist.github.com/blenzi/1cf8d14fd01494f7d9c0e34714f35c29/raw'
        self.ref = eval(urllib.request.urlopen(url).read())
        self.img = imutils.url_to_image(self.ref['url'])

    def test_extract_coordinates(self):
        caption = self.ref['caption']
        coordinates = self.ref['coordinates']
        self.assertEqual(extract_coordinates(caption), coordinates)

    def test_extract_timestamp(self):
        caption = self.ref['caption']
        timestamp = self.ref['timestamp']
        self.assertEqual(extract_timestamp(caption), timestamp)

    def test_shape(self):
        self.assertEqual(self.ref['shape'], self.img.shape)


if __name__ == '__main__':
    unittest.main()
