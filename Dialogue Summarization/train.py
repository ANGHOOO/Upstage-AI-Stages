import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
import wandb
from utils import compute_metrics


def load_trainer_for_train(config, generate_model, tokenizer, train_inputs_dataset, val_inputs_dataset):
    training_args = Seq2SeqTrainingArguments(
        output_dir=config['general']['output_dir'],
        overwrite_output_dir=config['training']['overwrite_output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        optim=config['training']['optim'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        seed=config['training']['seed'],
        logging_dir=config['training']['logging_dir'],
        logging_strategy=config['training']['logging_strategy'],
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval'],
        report_to=config['training']['report_to']
    )

    wandb.init(entity=config['wandb']['entity'], project=config['wandb']['project'], name=config['wandb']['name'])
    os.environ["WANDB_LOG_MODEL"] = "true"
    os.environ["WANDB_WATCH"] = "false"

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )

    trainer = Seq2SeqTrainer(
        model=generate_model,
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics=lambda pred: compute_metrics(config, tokenizer, pred),
        callbacks=[early_stopping]
    )

    return trainer


def train_model(config, trainer):
    trainer.train()
    wandb.finish()
