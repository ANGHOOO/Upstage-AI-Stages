from transformers import BartForConditionalGeneration, BartConfig, AutoTokenizer
import torch


def get_model_and_tokenizer(config, device):
    model_name = config['general']['model_name']
    bart_config = BartConfig().from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(model_name, config=bart_config)
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)
    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)
    return generate_model, tokenizer


def get_trainer_model(config, device):
    return get_model_and_tokenizer(config, device)


def get_inference_model(config, device):
    generate_model, tokenizer = get_model_and_tokenizer(config, device)
    ckt_path = config['inference']['ckt_path']
    generate_model.load_state_dict(torch.load(ckt_path))
    return generate_model, tokenizer
