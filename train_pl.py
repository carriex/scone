# -*- coding: utf-8 -*-

"""
PyTorchLighting training script
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pytorch_lightning as pl
import torch
import numpy as np
import random
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import OrderedDict
from model import SeqToSeqModel
from dataloader import AlchemyInstructionDataset, collate_fn
from argparse import ArgumentParser


def seed_everything(seed):
    '''set the seed'''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LearningRateCallBack(pl.Callback):
    '''Call back function for checking learning rate'''
    def on_init_start(self, trainer):
        print('Starting to init trainer!')
    def on_epoch_end(self, trainer, pl_module):
        if trainer.lr_schedulers:
            for scheduler in trainer.lr_schedulers:
                opt = scheduler['scheduler'].optimizer
                param_groups = opt.param_groups
                for param_group in param_groups:
                    print("--------------------------LR: ", param_group.get('lr'),
                          '------------------------------------------------------')


class AlchemySolver(pl.LightningModule):
    def __init__(self, hparams):
        super(AlchemySolver, self).__init__()
        self.hparams = hparams

        # datasets
        self.train_dataset = AlchemyInstructionDataset(self.hparams.train_data)
        self.val_dataset = AlchemyInstructionDataset(self.hparams.val_data,
                                          word_to_idx=self.train_dataset.word_to_idx,
                                          action_to_idx=self.train_dataset.action_to_idx,
                                          action_word_to_idx=self.train_dataset.action_word_to_idx,
                                          state_to_idx=self.train_dataset.state_to_idx,
                                          max_instruction_length=self.train_dataset.max_instruction_length,
                                          max_whole_instruction_length=self.train_dataset.max_whole_instruction_length)

        # model
        self.model = SeqToSeqModel(
            instruction_input_size=len(self.train_dataset.word_to_idx),
            state_input_size=len(self.train_dataset.state_to_idx),
            hidden_size=self.hparams.hidden_size,
            # max_encoder_length=self.train_dataset.max_instruction_length,
            max_encoder_length=self.train_dataset.max_whole_instruction_length,
            max_decoder_length=self.train_dataset.max_action_word_length,
            action_SOS_token=self.train_dataset.action_word_to_idx['_SOS'],
            action_EOS_token=self.train_dataset.action_word_to_idx['_EOS'],
            action_PAD_token=self.train_dataset.action_word_to_idx['_PAD'],
            output_size=self.train_dataset.num_word_actions,
            # max_decoder_length=self.train_dataset.max_action_length,
            # action_SOS_token=self.train_dataset.action_to_idx['_SOS'],
            # action_EOS_token=self.train_dataset.action_to_idx['_EOS'],
            # action_PAD_token=self.train_dataset.action_to_idx['_PAD'],
            # output_size=self.train_dataset.num_actions,
            state_to_idx=self.train_dataset.state_to_idx,
            idx_to_action=self.train_dataset.idx_to_action,
            idx_to_action_word=self.train_dataset.idx_to_action_words,
            vocab = self.train_dataset,
            teacher_forcing_ratio=self.hparams.teacher_forcing_ratio,
            use_attention=self.hparams.use_attention
        )

        # loss function
        self.criterion = torch.nn.NLLLoss(ignore_index=self.train_dataset.action_word_to_idx['_PAD'])

        # dict to store test results
        self.test_instruction_results = OrderedDict({'id': [],
                                         'final_world_state': []})

        self.test_interaction_results = OrderedDict({'id': [],
                                         'final_world_state': []})


    def _get_accuracy(self, predictions, labels):
        """return instruction level action accuracy"""
        _, predict_labels = predictions.max(1)
        correct_prediction = 0
        labels = labels.cpu().detach().numpy().reshape(-1, self.train_dataset.max_action_length)
        predict_labels = predict_labels.cpu().detach().numpy().reshape(-1, self.train_dataset.max_action_length)
        for si in range(labels.shape[0]):
            prediction = True
            for ti in range(labels.shape[1]):
                if labels[si][ti] == self.train_dataset.action_to_label_idx['_PAD']:
                    correct_prediction += prediction
                    break
                if labels[si][ti] != predict_labels[si][ti]:
                    prediction = False
                else:
                    continue
        return float(correct_prediction) / float(labels.shape[0])

    def _get_world_state_accuracy(self, predictions, labels):
        """return world state accuracy"""
        assert len(predictions) == len(labels), "predictions and labels should have same length"
        num_samples = len(predictions)
        num_correct_prediction = 0
        incorrect_prediction = []
        correct_prediction = []
        for i in range(num_samples):
            if predictions[i] == labels[i]:
                num_correct_prediction += 1
                correct_prediction.append(i)
            else:
                incorrect_prediction.append(i)
        return float(num_correct_prediction) / float(num_samples), incorrect_prediction, correct_prediction

    def _compute_loss(self, decoded_actions, target_actions):
        """return loss value given decoded actions and target actions"""
        num_samples = decoded_actions.size(0) * decoded_actions.size(1)
        target = target_actions.squeeze(-1).view(num_samples)
        prediction = decoded_actions.view((num_samples, -1))
        ### move to gpu ###
        if self.on_gpu:
            prediction = prediction.cuda(target.device.index)
        ######################
        loss = self.criterion(prediction, target)
        return loss


    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        # forward
        decoded_actions, world_states, attn_weights, state_attn_weights = self(batch)

        # compute loss
        loss = self._compute_loss(decoded_actions, batch['action_word_labels'])
        # compute accuracy
        world_state_accuracy, incorrect_prediction, correct_prediction\
            = self._get_world_state_accuracy(world_states, batch["after_env_str"])
        # check sample prediction
        if batch_idx % 100 == 0:
            if len(incorrect_prediction) > 0:
                print("########{} incorrect sample:".format(batch_idx))
                sample_idx = incorrect_prediction[np.random.randint(0, len(incorrect_prediction))]
                self._print_sample(batch, decoded_actions, attn_weights, state_attn_weights, sample_idx)
            if len(correct_prediction) > 0:
                print("########{} correct sample:".format(batch_idx))
                sample_idx = correct_prediction[np.random.randint(0, len(correct_prediction))]
                self._print_sample(batch, decoded_actions, attn_weights, state_attn_weights, sample_idx)
        # return values
        tqdm_dict = {
            'training_loss': loss,
            'training_accuracy': world_state_accuracy
        }
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output


    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset,
                                               batch_size=self.hparams.train_batch_size,
                                               shuffle=True,
                                               collate_fn=collate_fn,
                                               num_workers=self.hparams.num_worker)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]

    def _print_sample(self, batch, decoded_actions, attn_weights, state_attn_weights, sample_idx):
        """print out samples and predictions"""
        _, predict_labels = decoded_actions.max(2)
        print(">>>Idx {} Before env: ".format(batch['identifier'][sample_idx]), batch['before_env_str'][sample_idx])
        print("Instruction:",
              [self.train_dataset.idx_to_word[idx] for idx in batch['whole_instruction'].cpu().numpy()[sample_idx]])
        print("Action words:",
        [self.train_dataset.idx_to_action_words[idx] for idx in batch['action_words'].cpu().numpy()[sample_idx]])
        print("Predicted actions:",
              [self.train_dataset.idx_to_action_words[idx] for idx in predict_labels.cpu().numpy()[sample_idx]])
        if state_attn_weights is not None:
            state_attn_weights = state_attn_weights[-1]
            state_weight_idx = torch.argsort(state_attn_weights[sample_idx], descending=True).cpu().numpy()
            print("State weights rank:", state_weight_idx+1)
            print("State weights:", state_attn_weights[sample_idx].squeeze()[state_weight_idx].cpu().detach().numpy())
        if attn_weights is not None:
            attn_weights = attn_weights[-1]
            mask_idx = batch['whole_instruction_mask'][sample_idx]
            attn_weight = attn_weights[sample_idx].squeeze()
            weights_idx = torch.argsort(attn_weight[~mask_idx], descending=True).cpu().numpy()
            words_idx = batch['whole_instruction'].cpu().numpy()[sample_idx][weights_idx]
            print("Attention words:",
                  [self.train_dataset.idx_to_word[idx] for idx in words_idx])
            print("Attention weights:",
                  attn_weight[weights_idx].detach().cpu().numpy())

    def validation_step(self, batch, batch_idx):
        # forward
        decoded_actions, world_states, attn_weights, state_attn_weights = self.model.predict_instruction(batch)
        # compute loss
        loss = self._compute_loss(decoded_actions, batch['action_word_labels'])
        # compute accuracy
        world_state_accuracy, incorrect_prediction, correct_prediction \
            = self._get_world_state_accuracy(world_states, batch["after_env_str"])
        # check sample prediction
        if batch_idx == 0:
            if len(incorrect_prediction) > 0:
                print("########{} incorrect sample:".format(batch_idx))
                sample_idx = incorrect_prediction[np.random.randint(0, len(incorrect_prediction))]
                self._print_sample(batch, decoded_actions, attn_weights, state_attn_weights, sample_idx)
            if len(correct_prediction) > 0:
                print("########{} correct sample:".format(batch_idx))
                sample_idx = correct_prediction[np.random.randint(0, len(correct_prediction))]
                self._print_sample(batch, decoded_actions, attn_weights, state_attn_weights, sample_idx)
        # return values
        tqdm_dict = {
            'val_loss': loss,
            'val_acc': torch.tensor(world_state_accuracy)
        }
        output = OrderedDict({
            'val_loss': loss,
            'val_acc': torch.tensor(world_state_accuracy),
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output

    def validation_epoch_end(self, outputs):
        # compute average loss and accuracy
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': avg_loss,
                     'val_acc': avg_acc}
        return {'val_loss': avg_loss,
                'val_acc': avg_acc,
                'log': tqdm_dict,
                'progress_bar': tqdm_dict}

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset,
                                           batch_size=self.hparams.val_batch_size,
                                           shuffle=False,
                                           collate_fn=collate_fn,
                                           num_workers=self.hparams.num_worker)

    def test_step(self, batch, batch_idx, dataset_idx):
        # instruction prediction
        if dataset_idx == 0:
            decoded_actions, world_states, attn_weights, state_attn_weights = self.model.predict_instruction(batch)
            for idx in range(len(batch)):
                try:
                    self._print_sample(batch, decoded_actions, attn_weights, state_attn_weights, idx)
                except IndexError:
                    print(idx, " index error")
            self.test_instruction_results['id'] += batch['identifier']
            self.test_instruction_results['final_world_state'] += world_states

        # interaction prediction
        else:
            decoded_actions, world_states = self.model.predict_interaction(batch)
            self.test_interaction_results['id'] += batch['identifier']
            self.test_interaction_results['final_world_state'] += world_states
        ########### dumb output #########
        output = OrderedDict({
            'test_loss': 0,
            'progress_bar': {},
            'log': {}
        })
        ##################################
        return output

    def _get_test_accuracy(self, prediction_csv_file, label_csv_file):
        pred = pd.read_csv(prediction_csv_file, index_col="id")
        labels = pd.read_csv(label_csv_file, index_col="id")
        pred.columns = ["predicted"]
        labels.columns = ["actual"]
        data = labels.join(pred)
        return accuracy_score(data.actual, data.predicted)

    def test_epoch_end(self, outputs):
        # save dev result to csv
        # instruction
        df_instruction = pd.DataFrame.from_dict(self.test_instruction_results)
        instruction_pred_name = '{}-instruction.csv'.format(self.hparams.test_result)
        df_instruction.to_csv(instruction_pred_name, index=False)
        print("Test instruction output saved at {}".format(instruction_pred_name))
        instruction_acc = self._get_test_accuracy(instruction_pred_name, self.hparams.test_instruction_label)
        # instruction_acc = 0.0
        # interaction
        df_interaction = pd.DataFrame.from_dict(self.test_interaction_results)
        interaction_pred_name = '{}-interaction.csv'.format(self.hparams.test_result)
        df_interaction.to_csv(interaction_pred_name, index=False)
        print("Test interaction output saved at {}".format(interaction_pred_name))
        interaction_acc = self._get_test_accuracy(interaction_pred_name, self.hparams.test_interaction_label)
        # interaction_acc = 0.0

        ###########  output ##############
        output = OrderedDict({
            'instruction_acc': instruction_acc,
            'interaction_acc': interaction_acc,
            'progress_bar': {},
            'log': {
                'instruction_acc': instruction_acc,
                'interaction_acc': interaction_acc,
            }
        })
        ##################################
        return output


    def test_dataloader(self):
        self.test_instruction_dataset = AlchemyInstructionDataset(self.hparams.test_data,
                                          word_to_idx=self.train_dataset.word_to_idx,
                                          action_to_idx=self.train_dataset.action_to_idx,
                                          action_word_to_idx=self.train_dataset.action_word_to_idx,
                                          state_to_idx=self.train_dataset.state_to_idx,
                                          max_instruction_length=self.train_dataset.max_instruction_length,
                                          max_whole_instruction_length=self.train_dataset.max_whole_instruction_length,
                                          max_beaker_state_length=self.train_dataset.max_beaker_state_length)


        self.test_interaction_dataset = AlchemyInstructionDataset(self.hparams.test_data,
                                          is_interaction_dataset=True,
                                          word_to_idx=self.train_dataset.word_to_idx,
                                          action_to_idx=self.train_dataset.action_to_idx,
                                          action_word_to_idx=self.train_dataset.action_word_to_idx,
                                          state_to_idx=self.train_dataset.state_to_idx,
                                          max_instruction_length=self.train_dataset.max_instruction_length,
                                          max_whole_instruction_length=self.train_dataset.max_whole_instruction_length,
                                          max_beaker_state_length=self.train_dataset.max_beaker_state_length)


        return [torch.utils.data.DataLoader(dataset=self.test_instruction_dataset,
                                           batch_size=self.hparams.test_batch_size,
                                           shuffle=False,
                                           collate_fn=collate_fn,
                                           num_workers=self.hparams.num_worker),

                torch.utils.data.DataLoader(dataset=self.test_interaction_dataset,
                                            batch_size=self.hparams.test_batch_size,
                                            shuffle=False,
                                            collate_fn=collate_fn,
                                            num_workers=self.hparams.num_worker)
                                     ]

def main():
    # A few command line arguments
    argparse = ArgumentParser()
    argparse.add_argument("--train_data", dest="train_data", default='data/train_sequences.json')
    argparse.add_argument("--val_data", dest="val_data", default='data/dev_sequences.json')
    argparse.add_argument("--test_data", dest="test_data", default='/home/ec2-user/nlp/scone/data/alchemy/test_sequences.json')
    argparse.add_argument("--test_interaction_label", dest="test_interaction_label",
                          default='/home/ec2-user/nlp/scone/data/alchemy/test_interaction_y.csv')
    argparse.add_argument("--test_instruction_label", dest="test_instruction_label",
                          default='/home/ec2-user/nlp/scone/data/alchemy/test_instruction_y.csv')
    argparse.add_argument("--test_result", dest="test_result",
                          default='results/test_pred')
    argparse.add_argument("--train_batch_size",
                          help="training batch size",
                          dest="train_batch_size", type=int, default=16)
    argparse.add_argument("--test_batch_size",
                          help="testing batch size",
                          dest="test_batch_size", type=int, default=16)
    argparse.add_argument("--val_batch_size",
                          help="validation batch size",
                          dest="val_batch_size", type=int, default=16)
    argparse.add_argument("--num_worker",
                          help="number of worker for dataloader",
                          dest="num_worker", type=int, default=4)
    argparse.add_argument("--hidden_size",
                          help="size of the hidden layer of RNN",
                          dest="hidden_size", type=int, default=100)
    argparse.add_argument("--num_layers",
                          help="number of layers of the RNN",
                          dest="num_layers", type=int, default=1)
    argparse.add_argument("--teacher_forcing_ratio",
                          help="teacher forcing ratio",
                          dest="teacher_forcing_ratio", type=float, default=1)
    argparse.add_argument("--num_epoches",
                          help="number of training epoches",
                          dest="num_epoches", type=int, default=100)
    argparse.add_argument("--lr",
                          help="initial learning rate",
                          dest="lr", type=float, default=1e-3)
    argparse.add_argument("--load_from",
                          help="path to pretrained model checkpoint",
                          dest="load_from")
    argparse.add_argument("--run_test",
                          help="test on pretrained model",
                          dest="run_test",
                          action="store_true",
                          default=False)
    argparse.add_argument("--unit_test",
                          help="run fast dev",
                          dest="unit_test",
                          action="store_true",
                          default=False)
    argparse.add_argument("--use_attention",
                          help="use attention for decoder",
                          dest="use_attention",
                          action="store_true",
                          default=True)
    argparse.add_argument("-p", "--percent_check",
                          help="use only a certain percentage of data to run",
                          dest="percent_check",
                          type=float, default=0.0)

    seed_everything(88888)
    args = argparse.parse_args()

    if args.load_from is not None:
        # load from a previous checkpoint
        solver = AlchemySolver.load_from_checkpoint(
            checkpoint_path=args.load_from
        )
        solver.hparams = args
    else:
        solver = AlchemySolver(hparams=args)

    # turn off logging / checkpointing if running test
    use_logger = False if args.unit_test or args.run_test else True
    use_checkpoint_callback = False if args.unit_test or args.run_test else True

    trainer = pl.Trainer(gpus=1,
                         deterministic=True,
                         max_epochs=args.num_epoches,
                         check_val_every_n_epoch=1,
                         logger=use_logger,
                         checkpoint_callback=use_checkpoint_callback,
                         overfit_pct=args.percent_check,
                         callbacks=[LearningRateCallBack()]
                         )

    if not args.run_test:
        trainer.fit(solver)

    if not args.unit_test:
        trainer.test(solver)

if __name__ == "__main__":
    main()


