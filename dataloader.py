"""
Dataset creation for both instruction dataset and interaction dataset
"""
import torch
import json
from alchemy_fsa import AlchemyFSA, COLORS, EMPTY_SYMBOL, NUM_BEAKER, STARTER_STATE, NUM_INSTRUCTION_PER_INTERACTION
from alchemy_world_state import AlchemyWorldState

def collate_fn(data):
    """
    Function to collect list of samples into batches based on type:
    Input: data(dict): key: feature_name, value: list of features
    Returns: data(dict): key: feature_name, value: batch of features
    """
    keys = data[0].keys()
    res = {}
    for key in keys:
        if type(data[0][key]) == torch.Tensor:
            res[key] = torch.stack([d[key] for d in data], dim=0)
        else:
            res[key] = [d[key] for d in data]

    return res


class AlchemyInstructionDataset(torch.utils.data.Dataset):
    """
    Pytorch dataset for Alchemy instructions
    """
    def __init__(self, file_name, is_interaction_dataset=False,
                                  word_to_idx=None,
                                  action_word_to_idx=None,
                                  action_to_idx=None,
                                  state_to_idx=None,
                                  max_instruction_length=None,
                                  max_whole_instruction_length=None,
                                  max_beaker_state_length=None):

        self.is_interaction_dataset = is_interaction_dataset
        self.max_instruction_length = 0
        self.max_action_length = 0
        self.max_action_word_length = 0
        self.max_state_length = 0
        self.max_whole_instruction_length = 0
        self.max_beaker_state_length = 0
        # self.word_to_idx = {}
        self.word_to_idx = {'SEP_PREV':0, 'SEP_CURR':1}
        self.action_to_idx = {}
        self.action_word_to_idx = {'SEP': 0}
        self.state_to_idx = {}
        self.num_actions = 0
        self.num_word_actions = 0
        self.vocab_size = len(self.word_to_idx)
        self.num_beakers = 7
        self.supplementary_tokens = ['_UNK', '_EOS', '_SOS', '_PAD']

        # create encoding dicts and list of samples
        self.interactions_examples, self.instruction_examples = self._load_data_and_word_encoding(file_name)
        # update preset encoding dicts if passed-in
        if word_to_idx is not None:
            self.word_to_idx = word_to_idx
        if action_to_idx is not None:
            self.action_to_idx = action_to_idx
        else:
            self._create_action_encoding()
        if action_word_to_idx is not None:
            self.action_word_to_idx = action_word_to_idx
        else:
            self._create_action_word_encoding()
        if state_to_idx is not None:
            self.state_to_idx = state_to_idx
        else:
            self._create_state_encoding()

        if max_instruction_length is not None:
            self.max_instruction_length = max_instruction_length

        if max_whole_instruction_length is not None:
            self.max_whole_instruction_length = max_whole_instruction_length

        if max_beaker_state_length is not None:
            self.max_beaker_state_length = max_beaker_state_length

        self.num_word_actions = len(self.action_word_to_idx)
        self.idx_to_action = self._idx_to_action_list(self.action_to_idx)
        self.idx_to_action_words = {idx:word for (word, idx) in self.action_word_to_idx.items()}
        self.idx_to_word = {idx:word for (word, idx) in self.word_to_idx.items()}
        self.action_to_label_idx = self.action_to_idx.copy()


    def __len__(self):
        if self.is_interaction_dataset:
            return len(self.interactions_examples)
        else:
            return len(self.instruction_examples)

    def _get_instruction_item(self, idx):
        '''
                Return an instruction example
                Inputs:
                        idx: idx of current sample
                Outputs:
                        instruction(Array): instruction encoded by word_to_idx
                        actions(Array): action encoded by action_to_idx
                        before_env(String): world state string before the action
                        after_env(Sting): world state string after the action
                '''
        identifier, instruction_str, actions_str, action_words, \
        before_env_str, after_env_str, initial_env_str, init_state, \
        whole_instruction, all_instructions, all_states = self.instruction_examples[idx]

        actions_str = self._preprocess_actions_str(actions_str)
        # encode instruction and action
        instruction, instruction_mask = self._encode_and_pad_instruction(instruction_str)
        # actions = self._encode_and_pad_action(actions_str)
        action_words = self._encode_and_pad_action_word(action_words)
        # action_labels = self._encode_and_pad_action_labels(actions_str)
        # turn states into matrix
        # init_states = torch.stack([self.encode_and_pad_beaker_state(state) for state in init_states])
        init_state = self.encode_beaker_state_into_matrix(init_state)
        whole_instruction, whole_instruction_mask = self._encode_and_pad_whole_instruction(whole_instruction)
        # encode all instructions
        all_instructions, all_instructions_mask = self._encode_and_pad_instruction_in_interaction(all_instructions)
        # encode all states
        all_states = self.encode_five_beaker_states_into_matrix(all_states)
        return {
            'identifier': identifier,
            'instruction': instruction,
            'instruction_mask': instruction_mask,
            'instruction_str': instruction_str,
            # 'actions': actions,
            'action_words': action_words,
            'action_word_labels': action_words,
            # 'action_labels': action_labels,
            'actions_str': actions_str,
            'before_env_str': before_env_str,
            'after_env_str': after_env_str,
            'initial_env_str': initial_env_str,
            'init_state': init_state,
            'whole_instruction': whole_instruction,
            'whole_instruction_mask': whole_instruction_mask,
            'up_to_this_instructions': all_instructions,
            'up_to_this_instructions_mask': all_instructions_mask,
            'up_to_this_states': all_states
        }

    def _get_interaction_item(self, idx):
        '''
        Return list of instruction examples (interaction)
        TODO: align interaction item / instruction item
        '''
        identifier = self.interactions_examples[idx]['identifier']
        instruction_examples = self.interactions_examples[idx]['instructions']
        instructions = []
        actions = []
        action_words = []
        action_labels = []
        before_env_strs = []
        after_env_strs = []
        actions_strs = []
        instruction_strs = []
        instruction_mask = []
        init_states = []
        whole_instructions = []
        whole_instruction_masks = []

        for example in instruction_examples:
            instruction_identifier, instruction_str, actions_str, action_word, \
            before_env_str, after_env_str, initial_env_str, init_state, whole_instruction, \
            all_instructions, all_states = example  # TODO

            actions_str = self._preprocess_actions_str(actions_str)
            instruction, masked = self._encode_and_pad_instruction(instruction_str)
            instructions.append(instruction)
            instruction_mask.append(masked)
            actions.append(self._encode_and_pad_action(actions_str))
            action_words.append(self._encode_and_pad_action_word(action_word))
            action_labels.append(self._encode_and_pad_action_labels(actions_str))
            instruction_strs.append(instruction_str)
            actions_strs.append(actions_str)
            before_env_strs.append(before_env_str)
            after_env_strs.append(after_env_str)
            init_states.append(torch.stack([self.encode_and_pad_beaker_state(state) for state in init_state]))
            whole_instruction, whole_instruction_mask = self._encode_and_pad_whole_instruction(whole_instruction)
            whole_instructions.append(whole_instruction)
            whole_instruction_masks.append(whole_instruction_mask)

        return {
            'identifier': identifier,
            'initial_env_str': self.interactions_examples[idx]['initial_env_str'],
            'instruction': torch.stack(instructions),
            'instruction_mask': torch.stack(instruction_mask),
            'instruction_str': instruction_strs,
            'actions': torch.stack(actions),
            'action_labels': torch.stack(action_labels),
            'action_words': torch.stack(action_words),
            'action_word_labels': torch.stack(action_words),
            'actions_str': actions_strs,
            'before_env_str': before_env_strs,
            'after_env_str': after_env_strs,
            'init_states': init_states,
            'whole_instruction': torch.stack(whole_instructions),
            'whole_instruction_mask': torch.stack(whole_instruction_masks)
        }

    def __getitem__(self, idx):
        if self.is_interaction_dataset:
            return self._get_interaction_item(idx)
        else:
            return self._get_instruction_item(idx)


    def _add_supplementary_tokens(self, token_to_idx):
        for token in self.supplementary_tokens:
            if token not in token_to_idx:
                token_to_idx[token] = len(token_to_idx)

    def _dict_to_list(self, dict):
        list = [None] * len(dict)
        for key, value in dict.items():
            list[value] = key
        return list

    def _idx_to_action_list(self, action_to_idx):
        idx_to_action = {}
        for action, idx in action_to_idx.items():
            if action in self.supplementary_tokens:
                idx_to_action[idx] = action
            else:
                idx_to_action[idx] = action.split()
        return idx_to_action

    def _preprocess_actions_str(self, actions_str):
        processed_actions_str = []
        for i, actions in enumerate(actions_str):
            if 'pop' in actions:
                actions += ' _NONE'
            processed_actions_str.append(actions)
            # Add a separating token between actions
            if i < len(actions_str) - 1:
                processed_actions_str.append('SEP')
            #########################################
        return processed_actions_str

    def preprocess_world_state_str(self, world_state_str):
        states = []
        for state in world_state_str.split(" "):
            states.append(state)
        return states

    def _encode_state(self, state_string):
        indexes = []
        for c in state_string:
            indexes.append(self.state_to_idx[c])
        return torch.tensor(indexes).view(-1, 1)

    def _pad_sequence(self, sequence, length, pad_token, pad_start=False):
        pad_length = max(0, length - len(sequence))
        if pad_start:
            return [pad_token] * pad_length + sequence[:length]
        else:
            return sequence[:length] + [pad_token] * pad_length

    def _encode_and_pad_action(self, actions):
        '''returns tensor of encoded action'''
        indexes = []
        for action in actions:
            indexes.append(self.action_to_idx[action])
        indexes.append(self.action_to_idx['_EOS'])
        indexes = self._pad_sequence(indexes, self.max_action_length, self.action_to_idx['_PAD'],
                                     pad_start=False)
        return torch.tensor(indexes)

    def _encode_and_pad_action_word(self, actions):
        indexes = []
        for action in actions:
            indexes.append(self.action_word_to_idx[action])
        indexes.extend([self.action_word_to_idx['_EOS'], self.action_word_to_idx['_NONE'], self.action_word_to_idx['_NONE']])
        indexes = self._pad_sequence(indexes, self.max_action_word_length, self.action_word_to_idx['_PAD'],
                                     pad_start=False)
        return torch.tensor(indexes)

    def _encode_and_pad_action_labels(self, actions):
        indexes = []
        for action in actions:
            indexes.append(self.action_to_label_idx[action])
        indexes.append(self.action_to_label_idx['_EOS'])
        indexes = self._pad_sequence(indexes, self.max_action_length, self.action_to_label_idx['_PAD'],
                                     pad_start=False)
        return torch.tensor(indexes)

    def encode_and_pad_state(self, world_state_str):
        world_state_str = world_state_str.replace(" ", "")
        indexes = []
        for state in world_state_str:
            indexes.append(self.state_to_idx[state])
        indexes.append(self.state_to_idx['_EOS'])
        indexes = self._pad_sequence(indexes, self.max_state_length, self.state_to_idx['_PAD'],
                                     pad_start=False)
        return torch.tensor(indexes)

    def encode_and_pad_beaker_state(self, world_state_str):
        world_state_str = world_state_str.replace(" ", "")
        indexes = []
        for state in world_state_str:
            indexes.append(self.state_to_idx[state])
        indexes.append(self.state_to_idx['_EOS'])
        indexes = self._pad_sequence(indexes, self.max_beaker_state_length, self.state_to_idx['_PAD'],
                                     pad_start=False)
        return torch.tensor(indexes)

    def encode_beaker_state_into_matrix(self, world_state_str):
        '''returns num_beakers x num_colors matrix representing a world state'''
        num_colors = len(COLORS)
        # initialize as 0
        matrix = torch.zeros((self.num_beakers, num_colors))
        for i, state_str in enumerate(world_state_str):
            state = state_str.split(':')[-1]
            for color in state:
                if color == '_':
                    continue
                matrix[i][COLORS.index(color)] += 1
        return matrix

    def encode_five_beaker_states_into_matrix(self, list_of_states):
        num_colors = len(COLORS)
        matrix = torch.zeros((NUM_INSTRUCTION_PER_INTERACTION, self.num_beakers, num_colors))
        for i, state in enumerate(list_of_states):
            matrix[i] = self.encode_beaker_state_into_matrix(state)
        return matrix


    def _encode_and_pad_instruction(self, instruction):
        '''returns tensor of encoded instruction and corresponding masking array'''
        indexes = []
        for word in instruction.split():
            if word in self.word_to_idx:
                indexes.append(self.word_to_idx[word])
            else:
                indexes.append(self.word_to_idx['_UNK'])
        indexes.append(self.word_to_idx['_EOS'])
        masked_true = torch.tensor([False for _ in indexes])
        masked_false = torch.tensor([True for _ in range(self.max_instruction_length - len(indexes))])
        masked_array = torch.cat((masked_true, masked_false), dim=0)
        indexes = self._pad_sequence(indexes, self.max_instruction_length, self.word_to_idx['_PAD'])
        return torch.tensor(indexes), masked_array

    def _encode_and_pad_instruction_in_interaction(self, instructions):
        '''
        return tensor of encoded instructions (shape: 5 x max_len) and corresponding masking array
        '''
        encoded_instructions = torch.ones((NUM_INSTRUCTION_PER_INTERACTION, self.max_instruction_length)) \
                               * self.word_to_idx['_PAD']
        masking_array = torch.ones((NUM_INSTRUCTION_PER_INTERACTION, self.max_instruction_length))
        for i, instruction in enumerate(instructions):
            encoded_instruction, mask = self._encode_and_pad_instruction(instruction)
            encoded_instructions[i] = encoded_instruction
            masking_array[i] = mask
        return encoded_instructions, masking_array

    def _encode_and_pad_whole_instruction(self, instruction):
        indexes = []
        for word in instruction.split():
            if word in self.word_to_idx:
                indexes.append(self.word_to_idx[word])
            else:
                indexes.append(self.word_to_idx['_UNK'])
        indexes.append(self.word_to_idx['_EOS'])
        masked_true = torch.tensor([False for _ in indexes])
        masked_false = torch.tensor([True for _ in range(self.max_whole_instruction_length - len(indexes))])
        masked_array = torch.cat((masked_true, masked_false), dim=0)
        indexes = self._pad_sequence(indexes, self.max_whole_instruction_length, self.word_to_idx['_PAD'])
        return torch.tensor(indexes), masked_array

    def _create_state_encoding(self):
        # add colors
        for color in COLORS:
            self.state_to_idx[color] = len(self.state_to_idx)
        # add empty symbol
        self.state_to_idx[EMPTY_SYMBOL] = len(self.state_to_idx)
        # add beaker id
        for id in range(NUM_BEAKER):
            self.state_to_idx[str(id + 1)] = len(self.state_to_idx)
        # add colon
        self.state_to_idx[':'] = len(self.state_to_idx)
        # add EOS
        self.state_to_idx['_EOS'] = len(self.state_to_idx)
        # add PAD
        self.state_to_idx['_PAD'] = len(self.state_to_idx)

    def _create_action_encoding(self):
        starter_world_state = AlchemyWorldState(STARTER_STATE)
        fsa = AlchemyFSA(starter_world_state)
        actions = fsa.valid_actions()
        for (i, action) in enumerate(actions):
            action_string = ' '.join(action)
            self.action_to_idx[action_string] = len(self.action_to_idx)
        self._add_supplementary_tokens(self.action_to_idx)
        self.num_actions = len(self.action_to_idx)

    def _create_action_word_encoding(self):
        starter_world_state = AlchemyWorldState(STARTER_STATE)
        fsa = AlchemyFSA(starter_world_state)
        actions = fsa.valid_actions()
        for (i, action) in enumerate(actions):
            for action_word in action:
                if action_word not in self.action_word_to_idx:
                    self.action_word_to_idx[action_word] = len(self.action_word_to_idx)
        self._add_supplementary_tokens(self.action_word_to_idx)

    def _load_data_and_word_encoding(self, filename):
        """Loads the data from the JSON files.
        Inputs:
            filename (str): Filename of a JSON encoded file containing the data.
        Returns:
            examples: [instruction, actions, before_env, after_env]
        """
        instruction_examples = []
        interactions_examples = []
        with open(filename) as json_file:
            data = json.load(json_file)
            for d in data:
                examples = []
                utterances = d['utterances']
                identifier = d['identifier']
                initial_env_str = d['initial_env']
                for i, utterance in enumerate(utterances):
                    if i == 0:
                        before_env = initial_env_str
                    else:
                        before_env = utterances[i-1]['after_env']
                    after_env = utterance['after_env']
                    actions = utterance['actions']
                    action_words = [word for action in self._preprocess_actions_str(actions) for word in action.split()]
                    instruction = utterance['instruction']
                    # separate beaker states
                    states = self.preprocess_world_state_str(before_env)
                    whole_instruction = ''
                    all_instructions = []
                    all_states = []
                    for idx, example in enumerate(examples):
                        all_instructions.append(example[1])
                        all_states.append(states)
                        if idx < len(examples) -1:
                            whole_instruction += example[1] + ' SEP_PREV '
                        else:
                            whole_instruction += example[1] + ' SEP_CURR '
                    all_instructions.append(instruction)
                    all_states.append(states)
                    whole_instruction += instruction
                    if len(actions) > self.max_action_length:
                        self.max_action_length = len(actions)
                    if len(action_words) > self.max_action_word_length:
                        self.max_action_word_length = len(action_words)
                    if len(instruction.split()) > self.max_instruction_length:
                        self.max_instruction_length = len(instruction.split())
                    if len(whole_instruction.split()) > self.max_whole_instruction_length:
                        self.max_whole_instruction_length = len(whole_instruction.split())
                    if len(after_env) > self.max_state_length:
                        self.max_state_length = len(after_env)
                    for state in states:
                        if len(state) > self.max_beaker_state_length:
                            self.max_beaker_state_length = len(state)
                    for word in instruction.split():
                        if word not in self.word_to_idx:
                            self.word_to_idx[word] = len(self.word_to_idx)
                            self.vocab_size += 1
                    examples.append(['{}-{}'.format(identifier, i), instruction, actions, action_words,
                                     before_env, after_env, initial_env_str, states, whole_instruction, all_instructions, all_states])
                interactions_examples.append({
                    'identifier': identifier,
                    'instructions': examples,
                    'initial_env_str': initial_env_str
                })
                instruction_examples.extend(examples)

        self._add_supplementary_tokens(self.word_to_idx)
        self.max_action_length = self.max_action_length + 2
        self.max_instruction_length = self.max_instruction_length + 2
        self.max_whole_instruction_length = self.max_whole_instruction_length + 2
        self.max_action_word_length = self.max_action_word_length + 4
        self.max_state_length = self.max_state_length + 1
        self.max_beaker_state_length = self.max_beaker_state_length + 1

        return interactions_examples, instruction_examples




