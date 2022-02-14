# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import json
import numpy as np
import pandas as pd
import cv2
import os
from functools import partial


__all__ = ['AnnotationParser', 'splitStates', 'pickFrames']


def getFps(fname, inputdir='.'):
    "Return the number of frames per second for the given movie file"
    return cv2.VideoCapture(os.path.join(inputdir, fname)).get(cv2.CAP_PROP_FPS)


def getFileInfo(fname, pattern=r'(?P<fBase>\w+)_seq(?P<splitStart>\d+)_(?P<splitEnd>\d+).(?P<ext>\w+)', inputdir='.'):
    """
    Return a DataFrame with file info from the given series containing fname

    Args:
    - fname: Series
    - pattern: fname pattern to extract fBase, splitStart, splitEnd
    - inputdir: str, where to find the movie files (default: '')

    Returns: DataFrame with columns
    - fBase (fname without _seqX_Y),
    - splitStart and splitEnd (first and last frame for split file)
    - fps (frames per second)
    """
    d = fname.str.extract(pattern).astype({'splitStart': float, 'splitEnd': float})  # to allow NaN
    d['fBase'] = (d.fBase + '.' + d.ext).fillna(fname)
    d['fps'] = d.fBase.apply(partial(getFps, inputdir=inputdir))
    return d[['fBase', 'fps', 'splitStart', 'splitEnd']]


def splitStates(df, stateKeys=['fire', 'clf_confidence', 'loc_confidence']):
    """
    Return a DataFrame with one row per state, containing the first and last frames
    (stateStart and stateEnd) in addition to the columns in the given DataFrame
    (that must include information about the state).
    """
    def sameState(x, y):
        "Return true if rows x and y have the same state"
        return np.all(x[stateKeys] == y[stateKeys])

    # Take stateEnd as the frame of the next row - 1. For the last row it will be NaN
    # If state is the same for the last 2 rows (endpoint of sequence), add 1 to stateEnd (last frame)
    # Otherwise there was no endpoint in the sequence, set it to splitEnd
    # Finally, drop the last row if it remains at NaN (endpoint or no splitEnd defined)
    Next = df.shift(-1)
    Prev = df.shift(1)  # NaN in case of 1 row in df
    states = df.rename(columns={'frame': 'stateStart'}).join(Next.frame.rename('stateEnd') - 1)
    if not sameState(states.iloc[-1], Prev.iloc[-1]):
        states.loc[states.index[-1], 'stateEnd'] = states.splitEnd.iloc[-1]
    else:
        states.loc[states.index[-2], 'stateEnd'] += 1
    states.dropna(subset=['stateEnd'], inplace=True)
    # FIXME: find a way to flag invalid and return useful info
    reject = states.stateStart >= states.stateEnd
    return states[~reject].drop(columns=['splitStart', 'splitEnd'], errors='ignore')


def pickFrames(state, nFrames, random=True, seed=42):
    """
    Return a Series with the list of selected frames for the given state

    Args:
    - state: Series containing stateStart, stateEnd
    - nFrames: number of frames to pick
    - random: bool (default: True). Pick frames randomly or according to np.linspace,
      e.g. first if nFrames = 1, + last if nFrames = 2, + middle if nFrames = 3, etc
    - seed: int, seed for random picking (default: 42)
    """
    np.random.seed(seed)
    if random:
        return pd.Series(np.random.randint(state.stateStart, state.stateEnd, nFrames))
    else:
        return pd.Series(np.linspace(state.stateStart, state.stateEnd, nFrames, dtype=int))


def getFrameLabels(states, nFrames, **kw):
    """
    Given a DataFrame with states, call pickFrames to create a DataFrame with
    nFrames per state containing the state information, filename and
    imgFile (the name of the file to be used when writing an image)

    Args:
    - states: DataFrame containing fBase, stateStart, stateEnd
    - nFrames: int, number of frames per state
    - kw: list of keyword arguments for pickFrames
    """
    fcn = partial(pickFrames, nFrames=nFrames, **kw)
    # DataFrame containing columns (0..nFrames - 1)
    frames = states.apply(fcn, axis=1)
    # Merge states and frames and transform each value of the new columns into a row
    # Drop the new column 'variable' that represents the column name in frames
    df = pd.melt(states.join(frames), id_vars=states.columns,
                 value_vars=range(nFrames), value_name='frame').drop(columns=['variable'])
    # Add image file name
    df['imgFile'] = df.apply(lambda x: os.path.splitext(x.fBase)[0] + f'_frame{x.frame}.png', axis=1)
    return df.sort_values(['fBase', 'frame'])


def writeFrames(labels, inputdir, outputdir):
    """
    Extract frames from <inputfile>/<fBase> and write frames as
    <outputdir>/<fBase>_frame<frame>.png

    Args:
    - labels: DataFrame containing fBase, frame, imgFile
    - inputdir: str, directory containing movie files
    - outputdir: str, output directory. Created if needed
    """
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    for name, group in labels.groupby('fBase'):
        movie = cv2.VideoCapture(os.path.join(inputdir, name))
        for index, row in group.iterrows():
            movie.set(cv2.CAP_PROP_POS_FRAMES, row.frame)
            success, frame = movie.read()
            if success:
                cv2.imwrite(os.path.join(outputdir, row.imgFile), frame)
            else:
                print(f'Could not read frame {row.frame} from {name}')


class AnnotationParser:
    """
    Parse JSON file containing annotations for movies and produce the DataFrames described
    and illustrated below.

    Args:
    - fname: str, json file
    - inputdir: str, path of original (unsplit) movie files (default: '.')
    - defineStates: bool, define states from keypoints (default: True)

    Attributes:
    - labels: description of the information used in the annotations

    - files: list of movie files loaded for annotations. Some information about the files
      like fBase, fps, etc are only filled in case there are annotation points

            fid 	fname 	fBase 	fps 	splitStart 	splitEnd
    0 	1 	10.mp4 	10.mp4 	25.0 	NaN 	NaN
    1 	2 	19_seq0_591.mp4 	19.mp4 	25.0 	0.0 	591.0
    2 	3 	19_seq598_608.mp4 	19.mp4 	25.0 	598.0 	608.0

    - keypoints:
        fname 	fBase 	fps 	splitStart 	splitEnd 	fire 	sequence 	clf_confidence 	loc_confidence 	exploitable x y t frame
    1 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	1 	0 	1 	2 	True 	598.974 	467.692 	1.320 	33.0
    2 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	1 	0 	1 	2 	True 	609.231 	463.590 	2.826 	71.0
    3 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	1 	1 	1 	0 	True 	873.846 	500.513 	15.564 	389.0
    4 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	1 	1 	1 	0 	True 	869.744 	506.667 	18.637 	466.0
    6 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	0 	2 	1 	0 	True 	724.103 	449.231 	19.779 	494.0
    5 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	0 	2 	1 	0 	True 	543.590 	418.462 	20.057 	501.0
    7 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	1 	3 	1 	2 	True 	939.487 	244.103 	28.191 	705.0
    8 	10.mp4 	10.mp4 	25.0 	NaN 	NaN 	1 	3 	0 	0 	True 	957.949 	237.949 	38.907 	973.0
    10 	19_seq0_591.mp4 	19.mp4 	25.0 	0.0 	591.0 	1 	0 	0 	2 	True 	568.205 	358.974 	2.261 	57.0

    - states:
            fname 	fBase 	fps 	fire sequence clf_confidence 	loc_confidence 	exploitable x 	y 	t stateStart stateEnd
    fname 	sequence
    10.mp4 	0 	1 	10.mp4 	10.mp4 	25.0 	1 	0 	1 	2 	True 	598.974 	467.692 	1.320 	33.0 	71.0
    1 	3 	10.mp4 	10.mp4 	25.0 	1 	1 	1 	0 	True 	873.846 	500.513 	15.564 	389.0 	466.0
    2 	6 	10.mp4 	10.mp4 	25.0 	0 	2 	1 	0 	True 	724.103 	449.231 	19.779 	494.0 	501.0
    3 	7 	10.mp4 	10.mp4 	25.0 	1 	3 	1 	2 	True 	939.487 	244.103 	28.191 	705.0 	972.0
    19_seq0_591.mp4 	0 	10 	19_seq0_591.mp4 	19.mp4 	25.0 	1 	0 	0 	2 	True 	568.205 	358.974 	2.261 	57.0 	591.0

    """
    def __init__(self, fname, inputdir='.', defineStates=True):
        self.fname = fname
        self.inputdir = inputdir
        assert os.path.isdir(inputdir), f'Invalid path: {inputdir}'
        with open(fname) as jsonFile:
            info = json.load(jsonFile)

        # Annotation labels
        self.labels = pd.DataFrame(info['attribute'].values())
        self.labels['class'] = info['attribute'].keys()

        # Annotations
        self.annotations = pd.DataFrame(info['metadata'].values()).drop(columns='flg')

        # DataFrame with fid and fname. Add fBase, fps, splitStart, splitEnd
        # only for files with annotations
        self.files = pd.DataFrame(info['file'].values())[['fid', 'fname']]
        fnames = self.files.loc[self.files.fid.isin(self.annotations.vid), 'fname']
        self.files = self.files.join(getFileInfo(fnames, inputdir=self.inputdir))
        for fname in self.files.fBase.dropna():
            assert os.path.isfile(os.path.join(inputdir, fname)), f'File {fname} not found in path {inputdir}'

        # Process and cleanup annotations to extract keypoints
        # - merge with 'files' to get fname, fBase, fps, splitStart, splitEnd
        # - add information from all other DataFrames (in DFS)
        # - drop columns which are not needed after processing
        # - drop lines where both fire and exploitable are NaN (no annotation)
        # - replace exploitable=NaN by True
        # - sort by fname and t
        # - drop non-exploitable
        d = pd.merge(self.annotations, self.files, left_on='vid', right_on='fid')

        def splitKeypointValues(x):
            "Convert annotation info to a Series with the keys and values"
            class_to_aname = dict(zip(self.labels['class'], self.labels['aname']))
            aname_to_class = dict(zip(self.labels['aname'], self.labels['class']))

            # Explicitly converts (no value in JSON) to (NaN in Python)
            # for Exploitable videos. If not, column won't be created
            if aname_to_class['exploitable'] not in x:
                x[aname_to_class['exploitable']] = float('nan')

            return pd.Series(x).rename(index=class_to_aname)

        try:
            DFS = [
                d['av'].apply(splitKeypointValues),  # annotation info
                pd.DataFrame([xy for xy in d.xy.tolist()], columns=['dummy', 'x', 'y']),
                pd.DataFrame([z for z in d.z.tolist()], columns=['t'])
            ]
            d = d.join(DFS).drop(columns=['fid', 'xy', 'av', 'dummy', 'vid', 'z', 'spatial'], errors='ignore')
        except ValueError:
            # Exception is raised if xy does not contain 3 values,
            # which happens in JSON file only contains unexploitables videos
            # Create empty DataFrame with default columns to match standard csv files
            d = pd.DataFrame(columns=['fname', 'fBase', 'fps', 'splitStart', 'splitEnd',
                                      'exploitable', 'fire', 'sequence', 'clf_confidence',
                                      'loc_confidence', 'x', 'y', 't', 'frame'])

        d = d.dropna(how='all', subset=['fire', 'exploitable'])\
             .fillna({'exploitable': True}).sort_values(['fname', 't'])

        # Convert time to frame
        d['frame'] = np.round(d.fps * d.t.fillna(0)) + d.splitStart.fillna(0)

        # Reject invalid keypoints
        # FIXME: frame == splitEnd can remove end of sequence.
        # Should not be a problem but needs testing
        reject = d.eval('exploitable == "0" or frame >= splitEnd') | d.frame.isna()
        self.rejected = d[reject]
        self.keypoints = d[~reject]

        # Define states from keypoints
        if defineStates:
            self.states = self.keypoints.groupby(['fname', 'sequence']).apply(splitStates)

    def writeCsv(self, outputdir):
        """
        Write csv files with keypoints, rejected keypoints and states to the given
        output directory, created if needed
        """
        if not os.path.isdir(outputdir):
            os.mkdir(outputdir)

        basename = os.path.splitext(os.path.basename(self.fname))[0]
        for name in 'keypoints', 'rejected', 'states':
            df = getattr(self, name)
            fname = f'{os.path.join(outputdir, basename)}.{name}.csv'
            print(f'Writing csv file {fname}')
            try:
                df.to_csv(fname, index=False)
            except AttributeError:
                pass  # states not defined

    def writeFrames(self, outputdir, nFrames, random=True, seed=42):
        """
        Write frames from each state as <fBase>_frameX.png as well as a csv file
        <fname>.labels.csv with their info (frame, fname, state)

        Args:
        - outputdir: str, output directory. Created if needed
        - nFrames: int, number of frames per state
        - random: bool (default: True). Pick frames randomly or according to np.linspace,
          e.g. first if nFrames = 1, + last if nFrames = 2, + middle if nFrames = 3, etc
        - seed: int, seed for random picking (default: 42)
        """
        labels = getFrameLabels(self.states, nFrames, random=random, seed=seed)

        # Write labels
        if not os.path.isdir(outputdir):
            os.mkdir(outputdir)

        basename = os.path.splitext(os.path.basename(self.fname))[0]
        fLabels = os.path.join(outputdir, basename) + '.labels.csv'
        print(f'Writing frame labels to {fLabels}')
        labels.to_csv(fLabels, index=False)

        # Write frames
        print(f'Extracting {nFrames} frames per state ({len(labels)} in total) to {outputdir}')
        writeFrames(labels, self.inputdir, outputdir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fname', help='JSON file containing annotations')
    parser.add_argument('-i', '--inputdir', required=True,
                        help='Input directory containing movie files')
    parser.add_argument('-o', '--outputdir', required=True,
                        help='Output directory for writing csv files and images')
    parser.add_argument('-n', '--nFrames', default=0, type=int,
                        help='Number of frames per state to write')
    parser.add_argument('--random', help='Pick frames randomly', action='store_true')
    parser.add_argument('--seed', help='Seed for random picking the frames',
                        default=42, type=int)
    args = parser.parse_args()
    x = AnnotationParser(args.fname, inputdir=args.inputdir)
    x.writeCsv(args.outputdir)
    if args.nFrames > 0:
        x.writeFrames(args.outputdir, args.nFrames, args.random, args.seed)
    #print(x.keypoints)
    #print(x.states)

# TODO:
# - Test for getFrameLabels
# - save rejected states
# - tests for rejections
# - open and keep movie instances ?
