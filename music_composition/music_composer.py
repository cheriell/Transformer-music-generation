import warnings
warnings.filterwarnings('ignore')
import os
import sys
from typing import Optional
import note_seq
import numpy as np
import torch
from models import MusicCompTransformerModel

from data import Vocabulary, OneHotEncodingNoteSequence, vocabulary_csv_file, MusicCompDataModule


class MusicComp(object):

    def __init__(self, model_checkpoint: Optional[str] = os.path.join(sys.path[0], '../model_checkpoint/model_checkpoint.ckpt')):
        """Initialise Music Composer from model checkpoint.

        Args:
            model_checkpoint: filename to the model checkpoint.
        """
        self.vocabulary = Vocabulary._from_csv(vocabulary_csv_file)
        self.model = MusicCompTransformerModel.load_from_checkpoint(model_checkpoint, vocab_size=self.vocabulary.size)
        self.model.eval()
        self.device = 'cpu'

    def compose(self, seed: Optional[str] = None, 
                    seed_length: Optional[int] = 1000,
                    steps: Optional[int] = 1000,
                    gpu: Optional[int] = None,
                    diversity: Optional[int] = 30,
                    output_file: Optional[str] = os.path.join(sys.path[0], '../examples/sample_output.mid')):
        """Generate music by a seed from a midi file.

        Args:
            seed: seed saved in a midi file, if None, start from <SOS>.
            steps: number of steps to compose following the given seed.
            gpu: set GPU id if use gpu to predict, otherwise use CPU.
            diversity: generate music from top k=diversity predictions.
        """
        # setup device
        if gpu is not None:
            self.device = f'cuda:{gpu}'
        self.model.to(self.device)

        # get sequence
        if seed is None:
            sequence = [Vocabulary.SOS_INDEX]
        else:
            print('get seed')
            # calculate note_sequence representation
            note_sequence = note_seq.midi_io.midi_file_to_note_sequence(seed)
            # quantize notes by 10ms
            quantized_note_sequence = note_seq.sequences_lib.quantize_note_sequence_absolute(note_sequence, 100)
            # get performance events
            performance = note_seq.performance_lib.Performance(quantized_note_sequence)
            # get index sequence
            onehot_encoding = note_seq.encoder_decoder.OneHotEventSequenceEncoderDecoder(OneHotEncodingNoteSequence())
            sequence = [Vocabulary.SOS_INDEX] + onehot_encoding.encode(performance)[1]
            # trim sequence, use the begining <seed_length> symbols as seed
            sequence = sequence[:seed_length]
        # to Tensor
        sequence = torch.Tensor(sequence).int().unsqueeze(0).to(self.device)
        sequence_seed = sequence
        
        # compose
        for step in range(steps):
            print(f'step {step}/{steps}', end='\r')
            # trim input_sequence to avoid occupying too much memory
            input_sequence = sequence[:,-Vocabulary.MAX_LENGTH:]

            # get prediction
            mask = self.model.generate_square_subsequent_mask(input_sequence.shape[1]).to(self.device)
            probs = self.model(src=input_sequence, src_mask=mask)
            candidates = probs[0,:,-1].topk(diversity)[1]
            prediction = candidates[int(np.random.random() // (1. / diversity))]

            # update sequence
            sequence = torch.cat([sequence, prediction.unsqueeze(0).unsqueeze(1)], dim=1)
        print()

        # decode and get note sequence
        note_sequence = self.decode(list(sequence[0].cpu().detach().numpy()))
        note_sequence_seed = self.decode(list(sequence_seed[0].cpu().detach().numpy()))

        # save output to midi file
        if not os.path.exists(os.path.split(output_file)[0]):
            os.makedirs(os.path.split(output_file)[0])
        note_seq.sequence_proto_to_midi_file(note_sequence, output_file)
        note_seq.sequence_proto_to_midi_file(note_sequence_seed, output_file+'.seed.mid')

    def decode(self, sequence: list):
        """Decode list of index sequence to note sequence representation.

        Args: sequence of indexes
        """
        onehot_encoding = OneHotEncodingNoteSequence()
        performance = note_seq.performance_lib.Performance(steps_per_second=100)
        for idx in sequence:
            performance_event = onehot_encoding.decode_event(idx)
            if performance_event is not None:
                performance._events.append(performance_event)
        note_sequence = performance.to_sequence()
        return note_sequence
        

music_composer = MusicComp()
data_module = MusicCompDataModule()
data_module.prepare_data()
data_module.setup()
music_composer.compose(seed=data_module.midi_files_test[1], gpu=0)
# music_composer.compose(gpu=0)
