import torch
# from simpletts.utils.common import (add_sos_eos, combine_shif_left,
#                                     make_non_pad_mask, make_pad_mask)
from transformers import AutoModelForCausalLM, PreTrainedModel
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
        "speech_tokens": 6561 + 3  # 3: eos_sos task  fill
    }
    llm_config = Qwen2Config(**qwen_config_json)
    llm = AutoModelForCausalLM.from_config(llm_config)
    ttsllm = TTSLLM(llm_config, llm)
    llm_config.speech_tokens = 6561
    missing_keys, unexpected_keys = ttsllm.load_state_dict(rename_state_dict)
    for key in missing_keys:
        print("missing tensor: {}".format(key))
    for key in unexpected_keys:
        print("unexpected tensor: {}".format(key))
    return ttsllm


class TTSLLM(PreTrainedModel):

    supports_gradient_checkpointing = True

    def __init__(
        self,
        config,
        llm: torch.nn.Module,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.llm = llm
        self.eossos_task_emb = torch.nn.Embedding(2, config.hidden_size)

        self.speech_head = torch.nn.Linear(config.hidden_size,
                                           config.speech_tokens)
        self.speech_embed = torch.nn.Embedding(config.speech_tokens,
                                               config.hidden_size)

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
        pass
