import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import sys
import pytorch_lightning as pl
    
from data import Vocabulary, vocabulary_csv_file, MusicCompDataModule
from models import MusicCompTransformerModel
from music_composer import MusicComp


def train(args):
    """Model training.

    Args:
        args.gpu: set GPU id to for model training.
        args.dataset_path: path to the GiantMIDI-Piano dataset.
    """
    vocabulary = Vocabulary._from_csv(vocabulary_csv_file)
    model = MusicCompTransformerModel(vocab_size=vocabulary.size)
    datamodule = MusicCompDataModule(dataset_path=args.dataset_path)
    trainer = pl.Trainer(gpus=[args.gpu], reload_dataloaders_every_epoch=True)
    trainer.fit(model, datamodule=datamodule)


def compose(args):
    """Generate new midi from a seed.

    Args:
        args.model_checkpoint: path to the pre-trained model checkpoint.
        args.seed_midi: seed saved in a midi file, if None, start from <SOS>.
        args.seed_length: maximum length to the note sequence representation to use as seed.
        args.steps: number of steps to compose following the given seed.
        args.gpu: set GPU id if use gpu to predict, otherwise use CPU.
        args.diversity: generate music from top k=diversity predictions.
        args.output_file: filename to the generated midi output.
    """
    music_composer = MusicComp(model_checkpoint=args.model_checkpoint)
    data_module = MusicCompDataModule()
    data_module.prepare_data()
    data_module.setup()
    music_composer.compose(seed=args.seed_midi, 
                            seed_length=args.seed_length,
                            steps=args.steps,
                            gpu=args.gpu,
                            diversity=args.diversity,
                            output_file=args.output_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', help='Select mode from [train][compose]')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--gpu', type=int, default=None, help='set GPU id to for model training')
    parser_train.add_argument('--dataset_path', type=str, default=os.path.join(sys.path[0], '../GiantMIDI-Piano/midis/'), help='path to the GiantMIDI-Piano dataset')

    parser_compose = subparsers.add_parser('compose')
    parser_compose.add_argument('--model_checkpoint', type=str, default=os.path.join(sys.path[0], '../model_checkpoint/model_checkpoint.ckpt'), help='path to the pre-trained model checkpoint')
    parser_compose.add_argument('--seed_midi', type=str, default=None, help='seed saved in a midi file, if None, start from <SOS>.')
    parser_compose.add_argument('--seed_length', type=int, default=1000, help='maximum length to the note sequence representation to use as seed.')
    parser_compose.add_argument('--steps', type=int, default=1000, help='number of steps to compose following the given seed.')
    parser_compose.add_argument('--diversity', type=int, default=40, help='generate music from top k=diversity predictions.')
    parser_compose.add_argument('--gpu', type=int, default=None, help='set GPU id if use gpu to predict, otherwise use CPU.')
    parser_compose.add_argument('--output_file', type=str, default=os.path.join(sys.path[0], '../examples/sample_output.mid'), help='filename to the generated midi output.')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'compose':
        compose(args)
    else:
        raise ValueError('Please select mode from [train][compose]!')
    