import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationMixin

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()

        all_dims = [input_dim] + hidden_dims + [output_dim]

        self.linear_layers = nn.ModuleList()
        for i in range(len(all_dims) - 1):
            self.linear_layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            if i < len(self.linear_layers) - 1:
                x = F.gelu(x)
        return x
    
class InstructTime(GenerationMixin, nn.Module):
    _is_stateful = False
    def __init__(self, base_model: AutoModelForCausalLM, ecgTokenizers, text_embedding: int):
        super().__init__()
        self.base_model = base_model
        self.config = self.base_model.config
        self.generation_config = self.base_model.generation_config
        self.ecgTokenizers = ecgTokenizers
        self.model_dtype = self.base_model.get_input_embeddings().weight.dtype

        embed_vector = torch.empty(0, self.ecgTokenizers[0].hidden_dim)
        for tokenizer in self.ecgTokenizers:
            tokenizer_embed_vector = copy.deepcopy(tokenizer.quantize.embed).transpose(-1, 0)
            embed_vector = torch.cat([embed_vector, tokenizer_embed_vector], dim=0)
        self.embed_layer = nn.Embedding.from_pretrained(embed_vector)
        self.embed_layer.weight.data = self.embed_layer.weight.data.to(self.model_dtype)

        self.text_embedding = text_embedding
        self.embed = self.config.hidden_size
        if self.config.pad_token_id is None:
            self.config.pad_token_id = self.config.eos_token_id

        self.projection_layers = nn.ModuleList()
        for _ in ecgTokenizers:
            mlp = MLP(self.ecgTokenizers[0].hidden_dim, [64, 128, 256, 512], self.embed)
            mlp.apply(self.init_weights_kaiming)
            self.projection_layers.append(mlp.to(self.model_dtype))

        self.offsets = [self.text_embedding]
        for tokenizer in self.ecgTokenizers:
            self.offsets.append(self.offsets[-1] + tokenizer.n_embed)

    @staticmethod
    def init_weights_kaiming(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def forward(self, input_ids=None, **kwargs):
        if input_ids is None:
            return self.base_model(**kwargs)

        text_mask = torch.lt(input_ids, self.text_embedding)
        ecg_mask = ~text_mask

        text_ids = input_ids.clone()
        text_ids[ecg_mask] = self.config.pad_token_id

        text_embeddings = self.base_model.get_input_embeddings()(text_ids)
        text_embeddings.mul_(text_mask.float().unsqueeze(-1))

        ecg_embeddings = torch.zeros_like(text_embeddings)
        for i, _ in enumerate(self.ecgTokenizers):
            tokenizer_mask = (input_ids >= self.offsets[i]) & (input_ids < self.offsets[i + 1])
            tokenizer_ids = input_ids.clone()
            tokenizer_ids[~tokenizer_mask] = 0
            tokenizer_ids[tokenizer_mask] -= self.offsets[i]

            tokenizer_embeddings = self.embed_layer(tokenizer_ids)
            tokenizer_embeddings = self.projection_layers[i](tokenizer_embeddings)
            tokenizer_embeddings = tokenizer_embeddings.to(ecg_embeddings.dtype)
            tokenizer_embeddings.mul_(tokenizer_mask.float().unsqueeze(-1))
            ecg_embeddings.add_(tokenizer_embeddings)

        kwargs["inputs_embeds"] = ecg_embeddings + text_embeddings

        outputs = self.base_model(**kwargs)
        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def _reorder_cache(self, past, beam_idx):
        return self.base_model._reorder_cache(past, beam_idx)

    def can_generate(self):
        return True

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.base_model.set_input_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens):
        return self.base_model.resize_token_embeddings(new_num_tokens)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.base_model.set_output_embeddings(new_embeddings)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        self.config.architectures = ["InstructTime"]
        self.config.save_pretrained(save_directory)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def main_input_name(self):
        return self.base_model.main_input_name

class MultiTokenizer:
    def __init__(self, ecgTokenizers, text_model_dir: str) -> None:
        self.textTokenizer = AutoTokenizer.from_pretrained(text_model_dir)
        if self.textTokenizer.pad_token is None:
            self.textTokenizer.pad_token = self.textTokenizer.eos_token
        new_special_tokens = ["<BET>", "<EET>"]
        self.textTokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
        self.text_vocab_size = len(self.textTokenizer)

        self.ecgTokenizers = ecgTokenizers

        self.pad_token_id = self.textTokenizer.eos_token_id
        self.eos_token_id = self.textTokenizer.eos_token_id

        self.offsets = self._calculate_offsets()

    def _calculate_offsets(self):
        offsets = []
        current_offset = self.text_vocab_size
        for tokenizer in self.ecgTokenizers:
            offsets.append(current_offset)
            current_offset += tokenizer.n_embed
        return offsets

    def vocabSize_all(self):
        return self.text_vocab_size + sum(tokenizer.n_embed for tokenizer in self.ecgTokenizers)

    def encode(self, input, model_id=0):
        if isinstance(input, str):
            return self.textTokenizer(input)["input_ids"]
        elif isinstance(input, torch.Tensor):
            input = input.to('cpu')
            if model_id < len(self.ecgTokenizers):
                tokenizer_index = model_id
                _, _, indices = self.ecgTokenizers[tokenizer_index](input)
                return indices + self.offsets[tokenizer_index]
            else:
                raise ValueError(f"Invalid model_id. Please provide a number between 0 and {len(self.ecgTokenizers)}.")
        else:
            raise ValueError("Unsupported input type. Please provide either a string or a torch.Tensor.")
        
    def decode(self, input, skip_special_tokens=True):        
        return self.textTokenizer.decode(input, skip_special_tokens=skip_special_tokens)
