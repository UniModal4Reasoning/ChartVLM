import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import json
from auxiliary_decoder.train.utils.callbacks import Iteratorize, Stream
from auxiliary_decoder.train.utils.prompter import Prompter


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def infer_auxiliary_decoder(instruction, input=None, max_token=512, model_path="${PATH_TO_PRETRAINED_MODEL}/ChartVLM/base/", **kwargs): 
    load_8bit: bool = False
    base_model: str = os.path.join(model_path,'auxiliary_decoder', 'base')
    prompt_template: str = "alpaca"  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0"  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False
    trust_remote_code: bool=False
    lora_weights = os.path.join(model_path,'auxiliary_decoder')
    
    
    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True, trust_remote_code=trust_remote_code
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    temperature=0.1
    top_p=0.75
    top_k=40
    num_beams=4
    max_new_tokens=max_token
    stream_output=False

    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }



    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=False,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    output = tokenizer.decode(generation_output[0])
  
    
    return output