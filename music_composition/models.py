import os

from typing import Optional
import math
from functools import reduce
from jiwer import wer
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl

from data import Vocabulary


class MusicCompTransformerModel(pl.LightningModule):

    def __init__(self, vocab_size: int,
                embedding_size: Optional[int] = 128,
                d_model: Optional[int] = 128,
                nhead: Optional[int] = 8,
                num_layers: Optional[int] = 4):
        """Create a Transformer-based music composition model.

        Args:
            vocab_size: vocabulary size of the NoteSeqeunce representation.
            embedding_size: embedding size.
            d_model: the number of expected features in the Transformer input.
            nhead: the number of heads in the multiheadattention models.
            num_layers: the number of sub-encoder-layers in the transformer encoder.
        """
        super(MusicCompTransformerModel, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.output_layer = nn.Linear(in_features=d_model, out_features=vocab_size)

        self.d_model = d_model
        self.configure_loss_function()

    def forward(self, src: torch.Tensor, 
                src_mask: torch.Tensor):
        """Get model output.

        Args:
            src:
            src_mask:
        Returns:
            probs:
        """
        embedding = self.embedding_layer(src) * math.sqrt(self.d_model)  # [batch_size, length, embedding_size]
        embedding = embedding.transpose(0, 1)  # [length, batch_size, embedding_size]
        transformer_output = self.transformer_encoder(embedding, src_mask)  # [length, batch_size, d_model]
        probs = self.output_layer(transformer_output)  # [length, batch_size, vocab_size]
        probs = probs.permute(1, 2, 0)  # [batch_size, vocabulary, length]
        return probs

    ### this function if copied from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def configure_loss_function(self):
        self.loss_fn = nn.NLLLoss(ignore_index=Vocabulary.PAD_INDEX)

    def probs_to_seqeunce(self, probs: torch.Tensor):
        """Convert probs to list of index (for a single batch)

        Args:
            probs: [vocabulary, length] probs from the model output.
        Returns:
            sequence: (np.ndarray) index in 1D array.
        """
        probs = probs.cpu().detach().numpy()  # [vocabulary, length]
        sequence = probs.argmax(axis=0)  # [length]
        return sequence

    ### in the following, override model training functions
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, betas=(0.8, 0.8), eps=1e-4, weight_decay=2e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', filename='{epoch}-{val_loss:.4f}', save_top_k=1)
        earlystop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min')
        return [checkpoint_callback, earlystop_callback]

    def training_step(self, batch, batch_index):
        # get batch data
        sequence = batch
        input_sequence = sequence[:,:-1]
        output_sequence = sequence[:,1:]
        batch_size = input_sequence.shape[0]

        # get model output and loss
        mask = self.generate_square_subsequent_mask(input_sequence.shape[1]).to(self.device)
        probs = self(src=input_sequence, src_mask=mask)
        loss = self.loss_fn(probs, output_sequence)

        # WER
        word_error_rate = []
        for i in range(batch_size):
            # get prediction and target output
            predict_sequence = self.probs_to_seqeunce(probs[i])
            target_sequence = output_sequence[i].cpu().detach().numpy()
            # to strings
            target_string = ' '.join(list(target_sequence[target_sequence != Vocabulary.PAD_INDEX].astype(str)))
            predict_string = ' '.join(list(predict_sequence[predict_sequence != Vocabulary.PAD_INDEX].astype(str)))
            # udpate wer
            word_error_rate.append(wer(target_string, predict_string))
        
        # get logs
        logs = {'epoch': self.current_epoch,
                'loss': loss.item(),
                'wer': np.mean(word_error_rate)}

        return {'loss': loss, 'logs': logs}

    def validation_step(self, batch, batch_index):
        # get batch data
        sequence = batch
        input_sequence = sequence[:,:-1]
        output_sequence = sequence[:,1:]
        batch_size = input_sequence.shape[0]

        # get model output and loss
        mask = self.generate_square_subsequent_mask(input_sequence.shape[1]).to(self.device)
        probs = self(src=input_sequence, src_mask=mask)
        loss = self.loss_fn(probs, output_sequence)

        # WER
        word_error_rate = []
        for i in range(batch_size):
            # get prediction and target output
            predict_sequence = self.probs_to_seqeunce(probs[i])
            target_sequence = output_sequence[i].cpu().detach().numpy()
            # to strings
            target_string = ' '.join(list(target_sequence[target_sequence != Vocabulary.PAD_INDEX].astype(str)))
            predict_string = ' '.join(list(predict_sequence[predict_sequence != Vocabulary.PAD_INDEX].astype(str)))
            # udpate wer
            word_error_rate.append(wer(target_string, predict_string))
        
        # get logs
        logs = {'epoch': self.current_epoch,
                'val_loss': loss.item(),
                'val_wer': np.mean(word_error_rate)}
        self.log('val_loss', loss.item())

        return {'val_loss': loss, 'val_logs': logs}


    def training_epoch_end(self, outputs):
        # save logs
        new_logs = pd.DataFrame([output['logs'] for output in outputs])
        logs_file = f'lightning_logs/version_{self.trainer.logger.version}/training_logs.csv'
        if os.path.exists(logs_file):
            logs = pd.read_csv(logs_file)
            logs = pd.concat([logs, new_logs], ignore_index=True)
        else:
            logs = new_logs
        logs.to_csv(logs_file, index=False)

    def validation_epoch_end(self, outputs):
        # save logs
        new_logs = pd.DataFrame([output['val_logs'] for output in outputs])
        val_logs_file = f'lightning_logs/version_{self.trainer.logger.version}/validation_logs.csv'
        if os.path.exists(val_logs_file):
            logs_valid = pd.read_csv(val_logs_file)
            logs_valid = pd.concat([logs_valid, new_logs], ignore_index=True)
        else:
            logs_valid = new_logs
        logs_valid.to_csv(val_logs_file, index=False)




def train_model(gpus):
    import warnings
    warnings.filterwarnings('ignore')
    from data import Vocabulary, vocabulary_csv_file, MusicCompDataModule

    vocabulary = Vocabulary._from_csv(vocabulary_csv_file)
    model = MusicCompTransformerModel(vocab_size=vocabulary.size)
    datamodule = MusicCompDataModule()
    trainer = pl.Trainer(gpus=gpus, reload_dataloaders_every_epoch=True)
    trainer.fit(model, datamodule=datamodule)

train_model(gpus=[1])