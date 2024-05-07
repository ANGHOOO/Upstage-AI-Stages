import yaml
import torch
import os 
from dataset import Preprocess, DatasetForTrain, DatasetForVal
from model import get_trainer_model, get_inference_model
from train import load_trainer_for_train, train_model
from evaluate import inference


def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_file_path = os.path.join(data_path, 'concat_train.csv')
    val_file_path = os.path.join(data_path, 'dev.csv')

    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    encoder_input_train, decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val, decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)

    tokenized_encoder_inputs = tokenizer(
        encoder_input_train,
        return_tensors='pt',
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False
    )
    tokenized_decoder_inputs = tokenizer(
        decoder_input_train,
        return_tensors='pt',
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False
    )
    tokenized_decoder_outputs = tokenizer(
        decoder_output_train,
        return_tensors='pt',
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False
    )

    train_inputs_dataset = DatasetForTrain(tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_outputs, len(encoder_input_train))
    val_tokenized_encoder_inputs = tokenizer(
        encoder_input_val,
        return_tensors='pt',
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False
    )
    val_tokenized_decoder_inputs = tokenizer(
        decoder_input_val,
        return_tensors='pt',
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len']
    )
    val_tokenized_decoder_outputs = tokenizer(
        decoder_output_val,
        return_tensors='pt',
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['decoder_max_len'],
        return_token_type_ids=False
    )

    val_inputs_dataset = DatasetForVal(val_tokenized_encoder_inputs, val_tokenized_decoder_inputs, val_tokenized_decoder_outputs, len(encoder_input_val))
    return train_inputs_dataset, val_inputs_dataset


def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    generate_model, tokenizer = get_trainer_model(config, device)
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    data_path = config['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer)

    trainer = load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset)
    train_model(config, trainer)


if __name__ == "__main__":
    main()
