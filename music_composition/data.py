import warnings
warnings.filterwarnings('ignore')
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from typing import Optional
from collections import defaultdict
import note_seq
import pandas as pd
import numpy as np
import pickle
import torch
import pytorch_lightning as pl


vocabulary_csv_file = os.path.join(sys.path[0], '../metadata/vocabulary.csv')
note_sequence_cache_folder = os.path.join(sys.path[0], '../metadata/note_sequence_cache')
if not os.path.exists(note_sequence_cache_folder):
    os.makedirs(note_sequence_cache_folder)


class MusicCompDataModule(pl.LightningDataModule):
    """Music composition data module."""

    def __init__(self, batch_size: Optional[int] = 1,
                    dataset_path: Optional[str] = '/import/c4dm-datasets/GiantMIDI-Piano/midis/'):
        super(MusicCompDataModule, self).__init__()

        self.batch_size = batch_size
        self.dataset_path = dataset_path

    def prepare_data(self):
        """Override prepare_data()
        
        Get midi files in the dataset
        """
        midi_files = os.listdir(self.dataset_path)
        self.midi_files = [os.path.join(self.dataset_path, f) for f in midi_files]

    def setup(self, stage: Optional[str] = None):
        """Override setup()

        Split train/validation/test sets; setup vocabulary
        """
        self.midi_files_train = []
        self.midi_files_validation = []
        self.midi_files_test = []

        # split train/validation/test by 8/1/1
        for i, f in enumerate(self.midi_files):
            if i % 10 == 0:
                self.midi_files_validation.append(f)
            elif i % 10 == 1:
                self.midi_files_test.append(f)
            else:
                self.midi_files_train.append(f)

        # setup vocabulary
        if not os.path.exists(vocabulary_csv_file):
            # setup vocabulary by the training set
            vocabulary = Vocabulary._from_midi_files(self.midi_files_train)
            # save vocabulary to csv file for future use
            if not os.path.exists(os.path.split(vocabulary_csv_file)[0]):
                os.makedirs(os.path.split(vocabulary_csv_file)[0])
            vocabulary.to_csv(vocabulary_csv_file)
        else:
            # read vocabulary from csv file
            vocabulary = Vocabulary._from_csv(vocabulary_csv_file)

    ## in the following, override train/val/test dataloaders
    def train_dataloader(self):
        dataset = MusicCompDataset(self.midi_files_train)
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, drop_last=True)
        return data_loader
        
    def val_dataloader(self):
        dataset = MusicCompDataset(self.midi_files_validation)
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, drop_last=True)
        return data_loader
        
    def test_dataloader(self):
        dataset = MusicCompDataset(self.midi_files_test)
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, drop_last=True)
        return data_loader


class MusicCompDataset(torch.utils.data.Dataset):
    """Music composition dataset"""

    def __init__(self, midi_files: list):
        """Initialise dataset by midi files
        
        Args:
            midi_files: list of midi filenames.
        """
        super(MusicCompDataset, self).__init__()

        self.midi_files = midi_files

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, index):
        # define note sequence cache file
        note_sequence_cache_file = os.path.join(note_sequence_cache_folder, os.path.split(self.midi_files[index])[1][:-3]+'pkl')
        if not os.path.exists(note_sequence_cache_file):
            # calculate note_sequence representation
            note_sequence = note_seq.midi_io.midi_file_to_note_sequence(self.midi_files[index])
            # quantize notes by 10ms
            quantized_note_sequence = note_seq.sequences_lib.quantize_note_sequence_absolute(note_sequence, 100)
            # get performance events
            performance = note_seq.performance_lib.Performance(quantized_note_sequence)
            # get index sequence
            onehot_encoding = note_seq.encoder_decoder.OneHotEventSequenceEncoderDecoder(OneHotEncodingNoteSequence())
            sequence = [Vocabulary.SOS_INDEX] + onehot_encoding.encode(performance)[1] + [Vocabulary.EOS_INDEX]

            # save index sequence for future use
            pickle.dump(sequence, open(note_sequence_cache_file, 'wb'), protocol=2)
        else:
            sequence = pickle.load(open(note_sequence_cache_file, 'rb'))
        
        # trim sequence by MAX_LENGTH
        start = int(np.random.random() * (len(sequence) - Vocabulary.MAX_LENGTH))
        sequence = sequence[start:start+Vocabulary.MAX_LENGTH]
        # to numpy
        sequence = np.array(sequence)
        sequence_pad = np.ones(Vocabulary.MAX_LENGTH, dtype=int) * Vocabulary.PAD_INDEX
        sequence_pad[:len(sequence)] = sequence

        return sequence_pad

class OneHotEncodingNoteSequence(note_seq.encoder_decoder.OneHotEncoding):
    """Define one hot encoding using our vocabulary"""

    def __init__(self):
        """Initialise one hot encoding using our vocabulary"""
        self.vocabulary = Vocabulary._from_csv(vocabulary_csv_file)

    ##############################
    ## Override utility functions
    @property
    def num_classes(self):
        return len(self.vocabulary.word2count)

    def default_event(self):
        return Vocabulary.UNK_INDEX

    def encode_event(self, event):
        word = str(event)
        return self.vocabulary.word2index[word]

    def decode_event(self, index):
        word = self.vocabulary.index2word[index]
        if word[0] == '<':  # is <PAD>, <UNK>, <SOS> or <EOS>
            return None
        else:
            event_type_info, event_value_info = word.split('(')[1].split(')')[0].split(', ')
            event_type = int(event_type_info.split('=')[1])
            event_value = int(event_value_info.split('=')[1])
            performance_event = note_seq.performance_lib.PerformanceEvent(event_type=event_type, event_value=event_value)
            return performance_event


class Vocabulary(object):
    """Note sequence vocabulary"""

    UNK_INDEX = 0
    PAD_INDEX = 1
    SOS_INDEX = 2
    EOS_INDEX = 3
    MAX_LENGTH = 2500

    def __init__(self, word2count: dict, word2index: dict, index2word: dict):
        """Initialise vocabulary by relavant dictionarys.

        Args:
            word2count: word2count dictionary.
            word2index: word2index dictionary.
            index2word: index2word dictionary.
        """
        self.word2count = word2count
        self.word2index = word2index
        self.index2word = index2word

    @classmethod
    def _from_midi_files(cls, midi_files: list):
        """Initialise vocabulary from midi files in a dataset

        Args:
            midi_files: a list of midi files.
        Returns:
            vocabulary: (Vocabulary) calculated vocabulary.
        """
        # setup vocabulary count based on midi files in the dataset
        word2count = defaultdict(int)
        word2count['<UNK>'] = 0
        word2count['<PAD>'] = 0
        word2count['<SOS>'] = 0
        word2count['<EOS>'] = 0
        
        for i, midi_file in enumerate(midi_files):
            print(f'setup vocabulary {i}/{len(midi_files)}', end='\r')
            # get note_sequence representation
            note_sequence = note_seq.midi_io.midi_file_to_note_sequence(midi_file)
            # quantize notes by 10ms
            quantized_note_sequence = note_seq.sequences_lib.quantize_note_sequence_absolute(note_sequence, 100)
            # get performance events
            performance = note_seq.performance_lib.Performance(quantized_note_sequence)

            # update word2count
            for event in performance[:10000]:
                word = str(event)
                word2count[word] += 1
        print()
                
        # setup vocabulary indexes
        word2index = dict([(word, i) for i, word in enumerate(word2count.keys())])
        index2word = dict([(i, word) for word, i in word2index.items()])
        
        return cls(word2count, word2index, index2word)

    @classmethod
    def _from_csv(cls, csv_file: str):
        """Initialise vocabulary from pre-calculated csv file

        Args:
            csv_file: csv file to the vocabulary.
        Returns:
            vocabulary: (Vocabulary) vocabulary read from the csv file.
        """
        # initialise dictionaries
        word2count = defaultdict(int)
        word2index = defaultdict(int)
        index2word = dict()

        # get vocabulary information from csv file
        df = pd.read_csv(csv_file)
        for i, row in df.iterrows():
            word2count[row['word']] = int(row['count'])
            word2index[row['word']] = int(row['index'])
            index2word[int(row['index'])] = row['word']
        
        vocabulary = cls(word2count, word2index, index2word)
        return vocabulary

    @property
    def size(self):
        return len(self.word2count)

    def to_csv(self, csv_file: str):
        """Save vocabulary to a csv file.

        Args:
            csv_file: csv filename to save the vocabulary.
        """
        # save vocabulary information in a pandas dataframe
        rows = []
        for index in range(len(self.word2index)):
            word = self.index2word[index]
            count = self.word2count[word]
            rows.append({'index': index, 'word': word, 'count': count})
        df = pd.DataFrame(rows)

        # save to csv file
        df.to_csv(csv_file, index=False)
