# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import pickle
from bisect import bisect_left, bisect_right
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2
from pyrodataset.splitter_utils import frame_to_string, prepareOCR, extract_coordinates


class VideoSplitter:
    """
    To split a video into sequences of frames with the same camera settings (coordinates x,y,z, location name).

    The coordinates are extracted by applying OCR to the bottom of the frames.
    Since OCR is very slow, in order to avoid analysing all frames the splitting is done using binary search
    and assuming the same settings/coordinates only appear in one continuous sequence.

    Args:
        fname: video file name
        captions: dict or file containing captions
        acceptCloseMatches: accept locations with similar names to try to bypass OCR problems
        max_frames: int, default: 200. Maximum number of frames to analyse
        img_preprocessing (optional): function used to prepare image for OCR
        frame_to_string (optional): function used to extract caption from frame
        extract_coordinates (optional): function used to extract coordinates from caption
    """
    def __init__(self, fname, captions=None,
                 acceptCloseMatches=True, max_frames=200,
                 frame_to_string=frame_to_string,
                 img_preprocessing=prepareOCR,
                 extract_coordinates=extract_coordinates):
        self.fname = fname
        self.acceptCloseMatches = acceptCloseMatches
        self.max_frames = max_frames
        self.img_preprocessing = img_preprocessing
        self.frame_to_string = frame_to_string
        self.extract_coordinates = extract_coordinates

        self.video = cv2.VideoCapture(fname)
        self.Nframes = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        assert self.Nframes > 0, f'Invalid video file {fname}'
        self.fps = self.video.get(cv2.CAP_PROP_FPS)

        self.captions = {}  # (frame_index, caption)
        self.coordinates = {}  # (frame_index, camera position in x,y,z) N.B.: strings, not float
        # (coordinates, first frame where coordinates appeared):
        # represents a sorted list of frames, used for defining the start and end of sequences by bisection
        self.seqID = {}
        self.sequences = {}
        if captions is not None:
            self.loadCaptions(captions)

    def loadCaptions(self, captions):
        """
        Load captions from the given dictionary, file or path and process them
        in order to extract coordinates and seqID.
        In case of a path, look for <path>/<fname>_captions.pickle
        """
        if not isinstance(captions, dict):
            if os.path.isdir(captions):
                basename, ext = os.path.splitext(os.path.basename(self.fname))
                captions = os.path.join(captions, f'{basename}_captions.pickle')
            with open(captions, 'rb') as capFile:
                captions = pickle.load(capFile)
        self.captions = captions
        for frame_index in captions:
            self.processFrame(frame_index)

    def loadFrame(self, frame_index):
        """
        Load and return the frame with the given index or None if reading fails
        """
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = self.video.read()
        return frame if success else None

    def processFrame(self, frame_index, ignore_caption=False):
        """
        Extract and process the caption from the given frame, storing it in
        self.caption if not present, the coordinates in self.coordinates,
        and check if the frame belongs to a sequence previously identified
        or define a new entry in self.seqID

        Args:
            frame_index: the frame number
            ignore_caption: bool (default: False). Extract the caption even if it is
                            present in self.captions
        """
        if ignore_caption or frame_index not in self.captions:
            if len(self.captions) >= self.max_frames:
                raise RuntimeError(f'Maximum number of frames ({self.max_frames}) analysed')
            frame = self.loadFrame(frame_index)
            preprocessed = self.img_preprocessing(frame)
            caption = self.frame_to_string(preprocessed)
            self.captions[frame_index] = caption
        else:
            caption = self.captions[frame_index]
        try:
            pm = self.seqID if self.acceptCloseMatches else []
            coordinates = self.extract_coordinates(caption, possible_matches=pm)
        except ValueError:
            coordinates = ('EXTRACTION FAILED', frame_index, caption)

        self.coordinates[frame_index] = coordinates
        # If first time the coordinates appeared, add a new item to seqID
        if coordinates not in self.seqID:
            self.seqID[coordinates] = frame_index

    def getCoordinates(self, frame_index):
        """
        Return the coordinates of the camera for the frame with the given index.

        """
        if frame_index not in self.coordinates:
            self.processFrame(frame_index)
        return self.coordinates[frame_index]

    def printCaptions(self):
        "Print captions"
        print('Frame \t Caption')
        for i in sorted(self.captions.items()):
            print(f'{i[0]} \t {i[1]}')

    def __getitem__(self, item):
        """
        Return the first frame that revealed the sequence to which the given frame belongs.
        This method is called by bisect to find the boundaries of the sequence
        """
        return self.seqID[self.getCoordinates(item)]

    def findSequences(self):
        """
        Fill dictionary self.sequences with coordinates, (first frame, last frame) for each sequence
        """
        if not self.coordinates:
            # fill coordinates with first and last values if empty
            self[0], self[self.Nframes - 1]
        while True:
            # Coordinates not yet analysed
            missing = dict((frame, coord) for (frame, coord) in self.coordinates.items()
                           if coord not in self.sequences)
            if not missing:
                return
            # Find the start and end of each sequence corresponding to each set of coordinates
            for (frame, coord) in missing.items():
                self.sequences[coord] = bisect_left(self, self[frame]), bisect_right(self, self[frame]) - 1

    def printSequences(self):
        """
        Print frame range and coordinates of each sequence
        """
        if not self.sequences:
            return
        print('Frames \t Coordinates')
        for v, k in sorted((v, k) for k, v in self.sequences.items()):
            print(f'{v} \t {k}')

    def writeSequences(self, outputdir, min_frames=10):
        """
        Write each sequence to outputdir as <fname>_seq<fmin>_<fmax>.<ext>
        where fmin and fmax are the number of first and last frame

        Args:
            outputdir: str, output directory
            min_frames: int, default: 10. Minimum number of frames with same settings to consider a sequence
        """
        self.findSequences()
        if not os.path.isdir(outputdir):
            os.mkdir(outputdir)
        valid_sequences = filter(lambda x: x[1] - x[0] >= min_frames, self.sequences.values())
        for (fmin, fmax) in sorted(valid_sequences):
            basename, ext = os.path.splitext(os.path.basename(self.fname))
            fname = os.path.join(outputdir, f'{basename}_seq{fmin}_{fmax}{ext}')
            ffmpeg_extract_subclip(self.fname, fmin / self.fps, fmax / self.fps, fname)

    def writeInfo(self, outputdir):
        """
        Write dictionaries in outputdir/fname_<dictName>.pickle where
        dictName = captions, coordinates, seqID, sequences

        Args:
            outputdir: str, output directory
        """
        if not os.path.isdir(outputdir):
            os.mkdir(outputdir)
        basename, ext = os.path.splitext(os.path.basename(self.fname))
        dicts = {'captions': self.captions, 'coordinates': self.coordinates,
                 'seqID': self.seqID, 'sequences': self.sequences}
        for k, v in dicts.items():
            fname = os.path.join(outputdir, f'{basename}_{k}.pickle')
            with open(fname, 'wb') as pickleFile:
                pickle.dump(v, pickleFile)

    def __len__(self):
        return self.Nframes


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Split videos in sequences')
    parser.add_argument('filenames', help='Video filenames', nargs='+')
    parser.add_argument('--outputdir', required=True,
                        help='Output directory for writing sequences and info')
    parser.add_argument('--captions', default=None,
                        help='Pickle file or directory containing captions (optional)')
    parser.add_argument('--no-print', help='Do not print sequences', action='store_true')
    parser.add_argument('--min-frames', help='Minimum frames to write a sequence (default: 10)',
                        default=10, type=int)
    parser.add_argument('--max-frames', help='Maximum frames to process (default: 200)',
                        default=200, type=int)
    args = parser.parse_args()

    for fname in args.filenames:
        vs = VideoSplitter(fname, captions=args.captions, max_frames=args.max_frames)
        vs.writeSequences(args.outputdir, min_frames=args.min_frames)
        if not args.no_print:
            vs.printSequences()
        vs.writeInfo(args.outputdir)
