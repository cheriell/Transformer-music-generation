# Transformer-music-generation

ECS7022P Computational Creativity Homework - A simple pytorch implementation of Transformer-based music generator.

## Environment installation

This repository is written in python3.8 and pytorch, please install relavant python packages before running:

    pip install -r requirements.txt

## Data

This project uses the [GiantMIDI-Piano](https://github.com/bytedance/GiantMIDI-Piano) dataset, which contains 10,854 MIDI files from 2,786 composers. A copy of the dataset is submitted together with my code in folder `GiantMIDI-Piano`. The default dataset path to the dataset in `mian.py` is set to be run with my submission. If you download the dataset and save it somewhere else, please remember to specify the dataset path to where you save the dataset when you train the model.

## Model training and music generation

Please use GPU to train the model, simply run:

    python main.py train --gpu <cuda_device_id>

or

    python main.py train --dataset_path <path_to_the_dataset> --gpu <cuda_device_id>

to specify a different path to the dataset.

To use the trained model to generate new music, please run:

    python main.py compose --seed_midi <seed_midi_file> --gpu <cuda_device_id> --output_file <outout_midi_file>

This will use the beginning of your provided midi file as seed and a pretrained model to generate music. You can omit the `gpu` parameter to simply use CPU in generation, although this will be much more slower. `output_file` is where the generated new midi file will be. For more parameters, please refer to:

    python music_composition/main.py compose --help



