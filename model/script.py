import os
import pandas as pd
from huggingface_hub import login
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM, AutoTokenizer


class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        self.out, _ = module_outputs


def get_acts(example, tokenizer, model, layers):
    """
    Get given layer activations for the statements.
    Return dictionary of stacked activations.
    """
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()
        handle = model.model.layers[layer].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)
    # get activations
    acts = {layer: [] for layer in layers}
    tokenized_input = tokenizer(example, return_tensors="pt", padding=True)

    input_ids = tokenized_input['input_ids'].to(model.device)
    attention_masks = tokenized_input['attention_mask'].to(model.device)

    labels = input_ids.clone()
    labels = attention_masks.int() * labels + (1 - attention_masks.int()) * -100

    model_output = model(input_ids, attention_mask=attention_masks, labels=labels, output_hidden_states=True)
    shift_logits = model_output.logits[..., :-1, :].contiguous()

    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
    loss = loss.sum(-1).cpu().numpy().tolist()
    for layer, hook in zip(layers, hooks):
        # Return only the activation from the last token
        acts[layer].append(hook.out[torch.arange(hook.out.shape[0]), attention_masks.sum(dim=-1).cpu() - 1])

    for layer, act in acts.items():
        acts[layer].append(torch.stack(act)[0, :].cpu().float())

    # remove hooks
    for handle in handles:
        handle.remove()

    return loss, acts


def load_model(script_args):
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit
        # load_in_4bit=script_args.load_in_4bit
    )
    # Copy the model to each device
    torch_dtype = torch.bfloat16
    torch.set_grad_enabled(False)
    max_memory = {0: '15GiB', 1: '15GiB'}

    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name, device_map="auto",
                                                 max_memory=max_memory
                                                 , quantization_config=quantization_config
                                                 )

    num_layers = len(model.model.layers)
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, num_layers


def main(parser):
    # Load arguments
    if not os.path.exists('output_tensors'):
        os.mkdir('output_tensors')
    script_args = parser.parse_args()
    model_name_base = os.path.basename(script_args.model_name)
    # Load model, tokenizer
    model, tokenizer, num_layers = load_model(script_args)
    print(model)
    print("Model layers: ", num_layers)
    print("Model loading finish")

    # Load data
    print("Reading data")
    df = pd.read_csv(script_args.dataset_name)

    eval_dataset = Dataset.from_pandas(df)
    dataloader = DataLoader(eval_dataset, batch_size=1)

    # Start extracting activations
    layers = [i for i in range(num_layers)]
    perturbed_text_acts = {l: [] for l in layers}
    original_text_acts = {l: [] for l in layers}
    perturbed_scores, original_output, perturbed_output, ground_truth_output, result_type = [], [], [], [], []
    cnt = 0
    for batch in tqdm(dataloader, total=len(dataloader)):
        ## forward pass for getting all elements
        loss, perturbed_text_act = get_acts(batch['perturbed_text'], tokenizer, model, layers)
        _, original_text_act = get_acts(batch['original_text'], tokenizer, model, layers)

        ## extract each layer of features
        for layer in layers:
            perturbed_text_acts[layer].append(perturbed_text_act[layer][0][0, :])
            original_text_acts[layer].append(original_text_act[layer][0][0, :])

        ## get all other features
        perturbed_scores.append(batch['perturbed_score'])
        original_output.append(batch['original_output'])
        perturbed_output.append(batch['perturbed_output'])
        ground_truth_output.append(batch['ground_truth_output'])
        if batch['result_type'][0] == 'Successful':
            result_type.append(1)
        else:
            result_type.append(0)

    # Write to disk based on layers
    for layer in layers:
        torch.save(torch.stack(original_text_acts[layer]),
                   f"output_tensors/{model_name_base}_all_layer_{layer}_original_text.pt")
        torch.save(torch.stack(perturbed_text_acts[layer]),
                   f"output_tensors/{model_name_base}_all_layer_{layer}_perturbed_text.pt")
        torch.save(torch.tensor(perturbed_scores),
                   f"output_tensors/{model_name_base}_all_layer_{layer}_perturbed_scores.pt")
        torch.save(torch.tensor(original_output),
                   f"output_tensors/{model_name_base}_all_layer_{layer}_original_output.pt")
        torch.save(torch.tensor(perturbed_output),
                   f"output_tensors/{model_name_base}_all_layer_{layer}_perturbed_output.pt")
        torch.save(torch.tensor(ground_truth_output),
                   f"output_tensors/{model_name_base}_all_layer_{layer}_ground_truth_output.pt")
        torch.save(torch.tensor(result_type), f"output_tensors/{model_name_base}_all_layer_{layer}_result_type.pt")


if __name__ == '__main__':
    os.environ['HF_TOKEN'] = "my token"
    # Log in using the token from environment variables
    huggingface_token = os.environ.get('HF_TOKEN')
    login(token=huggingface_token)

    # df = pd.read_csv('../data/custom_attack_log.csv')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="The name of the model")
    parser.add_argument("--load_in_8bit", type=bool, help="Load model in 8 bit")
    parser.add_argument("--dataset_name", type=str, help="Load model in 8 bit")
    main(parser)
