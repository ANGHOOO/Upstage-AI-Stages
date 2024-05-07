from torch.utils.data import DataLoader
from tqdm import tqdm


def prepare_test_dataset(config, preprocessor, tokenizer):
    test_file_path = os.path.join(config['general']['data_path'], 'test.csv')
    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id = test_data['fname']
    encoder_input_test, decoder_input_test = preprocessor.make_input(test_data, is_test=True)

    test_tokenized_encoder_inputs = tokenizer(
        encoder_input_test,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        truncation=True,
        max_length=config['tokenizer']['encoder_max_len'],
        return_token_type_ids=False
    )

    test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))
    return test_data, test_encoder_inputs_dataset


def inference(config, generate_model, tokenizer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config, preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'])

    summary = []
    text_ids = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = generate_model.generate(
                input_ids=item['input_ids'].to(device),
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping'],
                max_length=config['inference']['generate_max_length'],
                num_beams=config['inference']['num_beams']
            )
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)

    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token, " ") for sentence in preprocessed_summary]

    output = pd.DataFrame({
        "fname": test_data['fname'],
        "summary": preprocessed_summary,
    })

    result_path = config['inference']['result_path']
    os.makedirs(result_path, exist_ok=True)
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)

    return output
