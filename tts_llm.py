from dataclasses import dataclass
from typing import Optional

import s3tokenizer
import torch
from simpletts.utils.common import (add_sos_eos, combine_shif_left,
                                    make_non_pad_mask, make_pad_mask)
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


def init_model(llm_model_path):

    ckpts = torch.load(llm_model_path, map_location='cpu', weights_only=True)
    rename_state_dict = {}
    for name, value in ckpts.items():
        if 'llm.model.model' in name:
            new_name = name.replace('llm.model.model', 'llm.model')
        elif name == 'llm.model.lm_head.weight':
            new_name = 'llm.lm_head.weight'
        elif 'llm_embedding.weight' == name:
            new_name = 'eossos_task_emb.weight'
        elif 'speech_embedding' in name:
            new_name = name.replace('speech_embedding', 'speech_embed')
        elif 'llm_decoder' in name:
            new_name = name.replace('llm_decoder', 'speech_head')
        else:
            new_name = name
        rename_state_dict[new_name] = value

    qwen_config_json = {
        "architectures": ["Qwen2ForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "hidden_act": "silu",
        "hidden_size": 896,
        "initializer_range": 0.02,
        "intermediate_size": 4864,
        "max_position_embeddings": 32768,
        "max_window_layers": 24,
        "model_type": "qwen2",
        "num_attention_heads": 14,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "sliding_window": 32768,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.1",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 151936,
        "pad_token_id": 151643,  # fix hg bug
        "speech_tokens": 6561,
    }
    llm_config = Qwen2Config(**qwen_config_json)
    llm = AutoModelForCausalLM.from_config(llm_config)
    ttsllm = Cosyvoice2LLM(llm_config, llm)
    llm_config.speech_tokens = 6561
    missing_keys, unexpected_keys = ttsllm.load_state_dict(rename_state_dict)
    for key in missing_keys:
        print("missing tensor: {}".format(key))
    for key in unexpected_keys:
        print("unexpected tensor: {}".format(key))
    return ttsllm


class Cosyvoice2LLM(PreTrainedModel, GenerationMixin):

    supports_gradient_checkpointing = True

    def __init__(
        self,
        config,
        llm: torch.nn.Module,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.llm = llm
        self.sos_eos = 0
        self.task_id = 1
        self.fill_token = 2
        self.eossos_task_emb = torch.nn.Embedding(2, config.hidden_size)

        # 3: sos_eos, task_id, fill_token
        self.speech_embed = torch.nn.Embedding(config.speech_tokens + 3,
                                               config.hidden_size)
        self.speech_head = torch.nn.Linear(config.hidden_size,
                                           config.speech_tokens + 3)

        self.speech_tokenizer = s3tokenizer.load_model(
            "speech_tokenizer_v2_25hz")
        self._keys_to_ignore_on_save = set()
        for k in self.speech_tokenizer.state_dict().keys():
            self._keys_to_ignore_on_save.add('speech_tokenizer.' + k)

    def _get_speech_tokens(self, speech: torch.Tensor,
                           speech_lens: torch.Tensor):
        # 1 speech tokens
        speech_codes, speech_codes_lens = self.speech_tokenizer.quantize(
            speech, speech_lens)
        speech_codes = speech_codes.clone()
        # NOTE(Mddct): in cosyvoice2
        # speech head layout: speech_tokens..., sos_eos, task_id, fill_token
        # input layout: sos_eos text task speech_token sos_eos
        speech_token_in, speech_token_out = add_sos_eos(
            speech_codes,
            self.sos_eos + self.config.speech_tokens,  # omit
            self.sos_eos + self.config.speech_tokens,
            -100,
        )
        speech_token_in = speech_token_in[:, 1:]
        speech_codes_lens = speech_codes_lens + 1
        return speech_token_in, speech_token_out, speech_codes_lens

    def get_input_embedding(
        self,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        speech_tokens: torch.Tensor,
        speech_tokens_lens: torch.Tensor,
    ):
        B = text.size(0)
        # NOTE(Mddct): in cosyvoice2
        # speech head layout: speech_tokens..., sos_eos, task_id, fill_token
        # input layout: sos_eos text task speech_token sos_eos
        sos_emb = self.eossos_task_emb.weight[self.sos_eos].reshape(
            1, 1, -1).repeat(B, 1, -1)
        task_emb = self.eossos_task_emb.weight[self.task_id].reshape(
            1, 1 - 1).repeat(B, 1, -1)

        # text tokens + speech tokens
        # NOTE:
        # [text_emb, speech_codes_emb]
        text_emb = self.llm.get_input_embeddings()(text)
        speech_in_emb = self.speech_embed(speech_tokens)

        ##  [sos_eos, text, taskid, speech_in_emb ]
        task_speech = torch.cat((task_emb, speech_in_emb), dim=1)
        task_speech_padding_mask = make_pad_mask(speech_tokens_lens + 1)
        soseos_text = torch.cat((sos_emb, text_emb), dim=1)
        soseos_text_padding_mask = make_pad_mask(text_lens + 1)
        input, _ = combine_shif_left(
            soseos_text,
            task_speech,
            soseos_text_padding_mask,
            task_speech_padding_mask,
        )
        input_lens = text_lens + speech_tokens_lens + 2

        return input, input_lens

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        speech: torch.Tensor,
        speech_lens: torch.Tensor,
        **kwargs,
    ):
        # 1 speech tokens
        speech_token_in, speech_token_out, speech_token_out_lens = self._get_speech_tokens(
            speech, speech_lens)
        speech_padding_mask = make_pad_mask(speech_token_out_lens)

        # 2 get input embeds
        input_embeds, input_lens = self.get_input_embedding(
            text, text_lens, speech_token_in, speech_token_out_lens -
            1)  # -1 means: sos_eos in tail of speech tokens
        if self.training:
            # 3 targets
            ignore_id_prepand = torch.zeros(
                input_embeds.size(0),
                2 + text.size(1) -
                1,  # 1 means: task predict first speech token
                dtype=torch.long,
                device=text.device) + (-100)
            ignore_input_valid_padding_mask = make_pad_mask(2 + text_lens - 1)
            targets, _ = combine_shif_left(
                ignore_id_prepand.unsqueeze(2),
                speech_token_out.unsqueeze(2),
                ignore_input_valid_padding_mask,
                speech_padding_mask,
            )
            targets = targets.squeeze(2)
            assert targets.shape[1] == input_embeds.shape[1]

        # 4 attention mask
        attention_mask = make_non_pad_mask(input_lens)
        # 5 forward and loss
        outputs = self.llm.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            # labels=targets,
        )
        hidden_states = outputs.hidden_states
        speech_logits = self.speech_head(hidden_states)
        loss = None
        if self.training:
            loss = self.loss_function(logits=speech_logits,
                                      labels=targets,
                                      vocab_size=self.config.speech_tokens + 3,
                                      **kwargs)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=speech_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        prompt_speech: torch.Tensor,
        prompt_speech_lens: torch.Tensor,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        topk: float,
        topp: float,
        do_sample: bool,
        eos_token_id: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        (prompts_speech_tokens_in, _,
         prompts_speech_lens) = self._get_speech_tokens(
             prompt_speech, prompt_speech_lens)
        inputs_embeds, input_lens = self.get_input_embedding(
            text,
            text_lens,
            prompts_speech_tokens_in,
            prompts_speech_lens,
        )
        attention_mask = make_non_pad_mask(input_lens)
        model_outputs = super().generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            top_p=topp,
            tpp_k=topk,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        return model_outputs

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()
