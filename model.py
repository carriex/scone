"""
A Sequence to Sequence model to translate instructions to actions in the Alchemy environment
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from alchemy_world_state import AlchemyWorldState


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim):
        """
        Bilinear Attention Layer
        """
        super(Attention, self).__init__()
        # BiLinear?
        self.attn = nn.Linear(query_dim, context_dim)

    def forward(self, query, context_vecs, context_mask=None):
        """
        Returns context vector and attention weights
        """
        # first attention head
        attention = self.attn(query)
        attention = torch.bmm(attention, context_vecs.transpose(1, 2))
        if context_mask is not None:
            attention.data.masked_fill_(context_mask, -float('inf'))
        attn_weights = F.softmax(attention, dim=2)
        # apply attention
        # (b x 1 x c)
        context_vector = torch.bmm(attn_weights,
                                   context_vecs)

        return context_vector, attn_weights


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 use_bidirectional=True):
        """
        Bi-directional LSTM Encoder for instruction
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)

        self.rnn = nn.LSTM(input_size=hidden_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True,
                           bidirectional=use_bidirectional)

        self.num_direction = 2 if use_bidirectional else 1

    def forward(self, input, hidden, gpu_index):
        # shape: batch x seq x feature
        input = input.view(-1, 1)
        ### move tensor to GPU ###
        if torch.cuda.is_available():
            input = input.cuda(gpu_index)
        ###########################
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self, batch_size, gpu_idx):
        hidden_state = torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_size)
        cell_state = torch.zeros(self.num_layers * self.num_direction, batch_size, self.hidden_size)
        ### move tensor to GPU ###
        if torch.cuda.is_available():
            hidden_state = hidden_state.cuda(gpu_idx)
            cell_state = cell_state.cuda(gpu_idx)
        ###########################
        return (hidden_state, cell_state)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size,
                 output_size,
                 max_encoder_length,
                 num_layers=2,
                 use_attention=False,
                 dropout_p=0.2,
                 attention_dropout=0.5):
        """
        Two layer LSTM with attention for action decoding
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size * 2 # as encoder is bi-directional
        self.max_encoder_length = max_encoder_length
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_size, hidden_size*2)
        self.use_attention = use_attention

        # -------------- attention -------------- #
        if self.use_attention:
            # self.attention = Attention(query_dim=self.hidden_size,
            #                            context_dim=self.hidden_size)
            # self.state_attention = Attention(query_dim=self.hidden_size,
            #                            context_dim=hidden_size)
            # Use multi-head attention
            self.encoder_attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4,
                                                           dropout=attention_dropout)
            self.state_attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=2,
                                                         dropout=attention_dropout)
            # self.init_state_attention = Attention(query_dim=self.hidden_size,
            #                            context_dim=hidden_size)
            # TODO: update shape
            # self.attn_combine = nn.Linear((self.hidden_size * 3) + self.hidden_size, self.hidden_size)
            self.attn_combine = nn.Linear((self.hidden_size * 2) + self.hidden_size, self.hidden_size)
        # --------------------------------------- #
        self.rnn = nn.LSTM(input_size=self.hidden_size,
                           hidden_size=self.hidden_size,
                           num_layers=num_layers, batch_first=True, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, state, init_state, encoder_states, instruction_mask, context_vector):
        # shape: batch x seq x feature
        input = input.view(-1, 1)
        embedded = self.embedding(input)
        encoder_attn_weights = None
        state_attn_weights = None

        # -------------- attention -------------- #
        if self.use_attention:
            hidden_state = hidden[0].transpose(0, 1)
            # context, attn_weights = self.attention(query=hidden_state,
            #                        context_vecs=encoder_states, context_mask=instruction_mask)
            # state, state_attn_weights = self.state_attention(query=hidden_state,
            #                        context_vecs=state)
            # Use mutlihead attention
            encoder_attn_outputs, encoder_attn_weights = self.encoder_attention(query=hidden_state.transpose(0, 1),
                                                                               key=encoder_states.transpose(0, 1),
                                                                               value=encoder_states.transpose(0, 1),
                                                                               key_padding_mask=instruction_mask.squeeze())
            # (batch, hidden_size)
            state_attn_outputs, state_attn_weights = self.state_attention(query=hidden_state.transpose(0, 1),
                                                                         key=state.transpose(0, 1),
                                                                         value=state.transpose(0, 1))
            hidden_state = torch.cat((hidden_state.transpose(0, 1),
                                      encoder_attn_outputs, state_attn_outputs), dim=2)
            hidden_state = nn.ReLU()(self.attn_combine(hidden_state))
            hidden = (hidden_state.contiguous(), hidden[1])
        # --------------------------------------- #
        output, hidden = self.rnn(embedded, hidden)
        output = self.softmax(self.out(output.squeeze()))
        return output, hidden, encoder_attn_weights, context_vector, state_attn_weights

    def initHidden(self, batch_size, gpu_idx):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        ### move tensor to GPU ###
        if torch.cuda.is_available():
            hidden_state = hidden_state.cuda(gpu_idx)
            cell_state = cell_state.cuda(gpu_idx)
        ###########################
        # random initialization
        return (hidden_state, cell_state)


class SeqToSeqModel(nn.Module):
    """
    A Sequence to Sequence model which predicts a sequence of output tokens given a sequence of input
    """
    def __init__(self, instruction_input_size,
                 state_input_size,
                 hidden_size,
                 output_size,
                 max_encoder_length,
                 max_decoder_length,
                 action_SOS_token,
                 action_EOS_token,
                 action_PAD_token,
                 state_to_idx,
                 idx_to_action,
                 idx_to_action_word,
                 vocab,
                 use_attention=False,
                 teacher_forcing_ratio=1):

        super(SeqToSeqModel, self).__init__()

        # -------------- param -------------- #
        self.max_encoder_length = max_encoder_length
        self.max_decoder_length = max_decoder_length
        self.action_SOS_token = action_SOS_token
        self.action_EOS_token = action_EOS_token
        self.action_PAD_token = action_PAD_token
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.state_to_idx = state_to_idx
        self.idx_to_action = idx_to_action
        self.idx_to_action_word = idx_to_action_word
        self.vocab = vocab

        # -------------- network -------------- #
        self.instruction_encoder = EncoderRNN(instruction_input_size, hidden_size)
        self.state_encoder = EncoderRNN(state_input_size, hidden_size * 2, use_bidirectional=False)
        self.state_beaker_encoder = EncoderRNN(state_input_size, hidden_size, use_bidirectional=False)
        self.action_decoder = DecoderRNN(hidden_size=hidden_size,
                                         output_size=output_size,
                                         max_encoder_length=self.max_encoder_length,
                                         num_layers=20,
                                         use_attention=use_attention)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gpu_index = None


    def _encode_instructions(self, input_seq):
        """
        Encodes a batch of input sequence and returns the outputs for each encoder input and the last hidden state
        """
        batch_size = input_seq.size(0)
        # init encoder_hidden
        # shape: batch_size x 1 x hidden_size
        instruction_encoder_hidden = self.instruction_encoder.initHidden(batch_size, self.gpu_index)
        # shape: batch_size x seq_len x hidden_size
        instruction_encoder_states = torch.zeros(batch_size,
                                                  self.max_encoder_length,
                                                  self.instruction_encoder.hidden_size * self.instruction_encoder.num_direction)
        for idx in range(input_seq.size(1)):
            instruction_encoder_output, instruction_encoder_hidden = self.instruction_encoder(
                input_seq[:, idx], instruction_encoder_hidden, self.gpu_index
            )
            instruction_encoder_states[:, idx, :] = torch.cat((instruction_encoder_hidden[0][0],
                                                               instruction_encoder_hidden[0][1]), dim=1)

        return instruction_encoder_states, instruction_encoder_hidden

    def _encode_world_state(self, world_states_strs):
        """
        Encode a batch of one world state string to embeddings
        Inputs:
            world_states_strs(String): Batch of String representing a batch of world state
        Output:
            embedding(Tensor): embedding of the world states; shape: batch_size x state encoder hidden size
        """
        batch_size = len(world_states_strs)
        state_encoder_hidden = self.state_encoder.initHidden(batch_size, self.gpu_index)
        batch_state = torch.stack([self.vocab.encode_and_pad_state(world_states_strs[idx]) for idx in range(batch_size)])
        ### move tensor to GPU ###
        if torch.cuda.is_available():
            batch_state = batch_state.cuda(self.gpu_index)
        ###########################
        for idx in range(self.vocab.max_state_length):
            state_encoder_output, state_encoder_hidden = self.state_encoder(
                batch_state[:, idx], state_encoder_hidden, self.gpu_index
            )
        return state_encoder_hidden[0].transpose(0, 1)

    def _encode_beaker_state(self, world_states_strs):
        """
        Encode a batch of beaker state strings
        Inputs:
            world_states_strs(String): Batch of List of String representing beaker state
        Outputs:
            embeddings(Tensor): embedding of the beaker states; shape: batch_size x num_beakers x state encoder hidden size
        """
        batch_size = len(world_states_strs)
        batch_states = [self.vocab.preprocess_world_state_str(state_str) for state_str in world_states_strs]
        beaker_state_encoding = []
        for i in range(self.vocab.num_beakers):
            state_encoder_hidden = self.state_beaker_encoder.initHidden(batch_size, self.gpu_index)
            batch_state = torch.stack(
                [self.vocab.encode_and_pad_beaker_state(batch_states[idx][i]) for idx in range(batch_size)])
            ### move tensor to GPU ###
            if torch.cuda.is_available():
                batch_state = batch_state.cuda(self.gpu_index)
            ###########################
            for idx in range(self.vocab.max_beaker_state_length):
                state_encoder_output, state_encoder_hidden = self.state_beaker_encoder(
                    batch_state[:, idx], state_encoder_hidden, self.gpu_index
                )
            beaker_state_encoding.append(state_encoder_hidden[0])
        return torch.cat(beaker_state_encoding, dim=0).transpose(0, 1)

    # def _encode_beakers_state(self, world_states_strs):
    #     """
    #     Encode a batch of beaker state strings
    #     Inputs:
    #         world_states_strs(String): Batch of List of String representing beaker state
    #     Outputs:
    #         embeddings(Tensor): embedding of the beaker states; shape: batch_size x state encoder hidden size
    #     """
    #     batch_size = len(world_states_strs)
    #     batch_states = [self.vocab.preprocess_world_state_str(state_str) for state_str in world_states_strs]
    #     # join beaker states into one string
    #     # (batch_size, )
    #     batch_states = " ".join(batch_states)
    #     beaker_state_encoding = []
    #     state_encoder_hidden = self.state_beaker_encoder.initHidden(batch_size, self.gpu_index)
    #     batch_state = torch.stack(
    #         [self.vocab.encode_and_pad_beaker_state(batch_states[idx]) for idx in range(batch_size)])
    #     ### move tensor to GPU ###
    #     if torch.cuda.is_available():
    #         batch_state = batch_state.cuda(self.gpu_index)
    #     ###########################
    #     for idx in range(self.vocab.max_beaker_state_length):
    #         state_encoder_output, state_encoder_hidden = self.state_beaker_encoder(
    #             batch_state[idx], state_encoder_hidden
    #         )
    #     beaker_state_encoding.append(state_encoder_hidden[0])
    #     return torch.cat(beaker_state_encoding, dim=0).transpose(0, 1)

    def _decode(self, instruction_encoder_hidden,
                       encoder_states,
                       instruction_mask,
                       world_states,
                       batch_state_encoding,
                       batch_init_state_encoding,
                       actions=None,
                       actions_str=None):
        """
        Decoder
        """
        use_teacher_forcing = False

        # only use teacher forcing during training
        if actions is not None:
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        batch_size = len(world_states)
        # shape: batch_size x 1 x hidden_size
        decoder_input = torch.tensor([[self.action_SOS_token] * batch_size]).view(batch_size, 1)
        # shape: batch_size x seq_len x output_size
        decoder_outputs = torch.zeros((batch_size, self.max_decoder_length, self.output_size))
        # shape: 1 x batch_size x embedding shape
        decoder_hidden_states = torch.zeros((self.action_decoder.num_layers,
                                             batch_size, self.action_decoder.hidden_size))
        decoder_cell_states = torch.zeros((self.action_decoder.num_layers,
                                           batch_size, self.action_decoder.hidden_size))

        context_vector = torch.zeros((batch_size, 1, self.instruction_encoder.hidden_size * 2))

        for i in range(self.action_decoder.num_layers):
            decoder_hidden_states[i] = torch.cat((instruction_encoder_hidden[0][0],
                                                  instruction_encoder_hidden[0][1]), dim=1).view(1, batch_size, -1)
            decoder_cell_states[i] = torch.cat((instruction_encoder_hidden[1][0],
                                                instruction_encoder_hidden[1][1]), dim=1).view(1, batch_size, -1)

        ### move tensor to GPU ###
        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda(self.gpu_index)
            decoder_cell_states = decoder_cell_states.cuda(self.gpu_index)
            decoder_hidden_states = decoder_hidden_states.cuda(self.gpu_index)
            context_vector = context_vector.cuda(self.gpu_index)
        ###########################

        single_action = [[] for _ in range(batch_size)]
        EOS = [False for _ in range(batch_size)]
        instruction_mask = instruction_mask.reshape(batch_size, 1, -1)
        seq_state_attn_weights = []
        seq_attn_weights = []


        for idx in range(self.max_decoder_length):
            decoder_output, (decoder_hidden_states, decoder_cell_states), attn_weights, \
            context_vector, state_attn_weights = self.action_decoder(
                input=decoder_input,
                hidden=(decoder_hidden_states, decoder_cell_states),
                state=batch_state_encoding,
                init_state=batch_init_state_encoding,
                encoder_states=encoder_states,
                instruction_mask=instruction_mask,
                context_vector=context_vector
            )
            decoder_outputs[:, idx, :] = decoder_output
            topv, topi = decoder_output.topk(k=1, dim=1)
            predicted_action = topi.squeeze().detach().cpu().numpy()
            # -------------- update world_state -------------- #
            for i, world_state in enumerate(world_states):
                action_word = self.idx_to_action_word[predicted_action[i]]
                single_action[i].append(action_word)
                # ------------------------------------------------ #
                if len(single_action[i]) == 4 and not EOS[i]:
                    if single_action[i][0] != '_EOS' and single_action[i][0] != 'SEP':
                        try:
                            world_states[i] = world_state.execute_seq([single_action[i][:3]])
                        except Exception:
                            # ignore wrong actions
                            pass
                    else:
                        EOS[i] = True
                    single_action[i] = []
            # -------------- update input for next -------------- #
            if use_teacher_forcing:
                decoder_input = actions[:, idx]
            else:
                decoder_input = topi

            if self.action_decoder.num_layers > 1 and attn_weights is not None and state_attn_weights is not None:
                attn_weights = attn_weights[:, 0:1, :]
                seq_attn_weights.append(attn_weights)
                state_attn_weights = state_attn_weights[:, 0:1, :]
                seq_state_attn_weights.append(state_attn_weights)

        return decoder_outputs, [str(state) for state in world_states], torch.stack(seq_attn_weights).transpose(0, 1), \
               torch.stack(seq_state_attn_weights).transpose(0, 1)

    def forward(self, batch_inputs):
        """
        Given a batch of input sequence, output the probability distribution of output
        sequence and predicted world states
        """
        # instruction = batch_inputs['instruction']
        instruction = batch_inputs['whole_instruction']
        # instruction_mask = batch_inputs['instruction_mask']
        instruction_mask = batch_inputs['whole_instruction_mask']
        # actions = batch_inputs['actions']
        action_words = batch_inputs['action_words']
        actions_str = batch_inputs['actions_str']
        before_env_str = batch_inputs['before_env_str']
        initial_env_str = batch_inputs['initial_env_str']
        batch_size = instruction.size(0)


        if torch.cuda.is_available():
            self.gpu_index = instruction.device.index

        # -------------- encoder -------------- #
        instruction_encoder_outputs, instruction_encoder_hidden = self._encode_instructions(instruction)

        # -------------- state encoding -------------- #
        world_states = [None] * batch_size
        for i, env in enumerate(before_env_str):
            world_states[i] = AlchemyWorldState(str(env))
        batch_state_encoding = self._encode_world_state([str(world_state) for world_state in world_states])
        # batch_state_encoding = self._encode_beaker_state([str(world_state) for world_state in world_states])
        batch_init_state_encoding = self._encode_beaker_state([str(world_state) for world_state in initial_env_str])

        ### move tensor to GPU ###
        if torch.cuda.is_available():
            batch_state_encoding = batch_state_encoding.cuda(self.gpu_index)
            batch_init_state_encoding = batch_init_state_encoding.cuda(self.gpu_index)
            instruction_encoder_outputs = instruction_encoder_outputs.cuda(self.gpu_index)
        ###########################

        # -------------- decoder -------------- #
        decoder_outputs, world_states_str, attn_weights, state_attn_weights = self._decode(
                                       instruction_encoder_hidden=instruction_encoder_hidden,
                                       encoder_states=instruction_encoder_outputs,
                                       instruction_mask=instruction_mask,
                                       world_states=world_states,
                                       batch_state_encoding=batch_state_encoding,
                                       batch_init_state_encoding=batch_init_state_encoding,
                                       actions=action_words, # for single token
                                       actions_str=actions_str)

        return decoder_outputs, world_states_str, attn_weights, state_attn_weights


    def predict_instruction(self, batch_instruction):
        """
        Given a batch of instruction example, returns a predicted action and final world state
        """
        return self(batch_instruction)

    def predict_interaction(self, interaction_example):
        """
        Given a batch of interaction example, returns a predicted action and final world state
        """
        whole_instruction = interaction_example['whole_instruction'] # batch_size x num_instruction x seq_len
        before_env_str = [before_env[0] for before_env in interaction_example['before_env_str']] # batch_size
        initial_env_str = interaction_example['initial_env_str']
        whole_instruction_mask = interaction_example['whole_instruction_mask']
        num_instruction = whole_instruction.size(1)

        if torch.cuda.is_available():
            self.gpu_index = whole_instruction.device.index

        # -------------- predict instructions -------------- #
        for i in range(num_instruction):
            instruction_example = {
                'whole_instruction': whole_instruction[:, i, :],
                'before_env_str': before_env_str,
                'whole_instruction_mask': whole_instruction_mask[:, i, :],
                'initial_env_str': initial_env_str
            }
            decoder_outputs, world_states_str, attn_weights, state_attn_weights = self.predict_instruction(instruction_example)
            before_env_str = world_states_str

        return decoder_outputs, world_states_str








