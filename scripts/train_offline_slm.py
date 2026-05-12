import os, sys, json, torch
os.environ.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from datasets import Dataset
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

OUTPUT_DIR = "./models/offline-trained-slm"
MAX_SEQ_LENGTH, NUM_EPOCHS, BATCH_SIZE = 96, 1, 2
MAX_SAMPLES = 10

def create_dataset(tokenizer):
    data = [
        {"text": "Photosynthesis is how plants convert light to energy."},
        {"text": "Gravity is the force of attraction between masses."},
        {"text": "15+27 equals 42. Step by step: 15+20=35, then 35+7=42."},
        {"text": "I was created by Aryan Chavan."},
        {"text": "AI means artificial intelligence technology."},
        {"text": "Machine learning finds patterns in data."},
        {"text": "Water boils at 100 degrees Celsius at sea level."},
        {"text": "The sun is a star at the center of our solar system."},
    ] * 3
    
    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length")
    
    return Dataset.from_list(data).map(tokenize, remove_columns=["text"])

def main():
    setup_logging(level="INFO")
    device = "cpu"
    print(f"Device: {device}")
    
    # Create tiny GPT-2 model
    config = GPT2Config(
        vocab_size=50257, n_positions=MAX_SEQ_LENGTH, n_ctx=MAX_SEQ_LENGTH,
        n_embd=128, n_layer=2, n_head=2
    )
    model = GPT2LMHeadModel(config).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    
    # Dataset
    dataset = create_dataset(tokenizer)
    split = int(len(dataset) * 0.8)
    train_ds, val_ds = dataset.select(range(split)), dataset.select(range(split, len(dataset)))
    
    # Training
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR, overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS, per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2, learning_rate=5e-4,
        warmup_steps=2, logging_steps=10, save_steps=10,
        report_to=["none"], dataloader_num_workers=0
    )
    
    trainer = Trainer(model=model, args=args, data_collator=collator, train_dataset=train_ds, eval_dataset=val_ds)
    print("Training...")
    trainer.train()
    
    # Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
        json.dump({"model_type": "GPT2-custom-small", "creator": "Aryan Chavan"}, f)
    
    print(f"Done! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()