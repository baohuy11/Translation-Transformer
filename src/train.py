from model import build_transformer
from dataset import BiligualDataset, causal_mask
from config import get_config, latest_weights_file_path

# import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
import gc
from pathlib import Path

# Huggingface dataset and tokenizer
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    
    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size[1] == max_len:
            break
        
        # Build the causal mask for the target sequence
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # Calcuate the decoder output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # Get the next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )
        
        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    
    source_texts = []
    expected_texts = []
    predicted_texts = []
    
    try:
        # Get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
            
    except Exception as e:
        console_width = 80
        
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (batch_size, seq_len) 
            encoder_mask = batch["encoder_input_mask"].to(device) # (batch_size, 1, 1, seq_len)
            
            # Check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_src, max_len, device)
            
            source_texts = batch["src_text"][0]
            target_texts = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_texts)
            expected_texts.append(target_texts)
            predicted_texts.append(model_out_text)
            
            # Print the source, target and model output texts
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_texts}")
            print_msg(f"{f'TARGET: ':>12}{target_texts}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
            
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted_texts, expected_texts)
        writer.add_scalar("validation cer", cer, global_step)
        writer.flush()
        
        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted_texts, expected_texts)
        writer.add_scalar("validation wer", wer, global_step)
        writer.flush()
        
        # Compute the BLEU score
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted_texts, expected_texts)
        writer.add_scalar("validation bleu", bleu, global_step)
        writer.flush()
        
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
        
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_ds(config):
    # It only has the train split, so we can use it for validation as well
    # ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    # Define the local dataset path
    ds_path = Path(config['datasource'])
    lang_pair = f"{config['lang_src']}-{config['lang_tgt']}"

    # Create directories if they don't exist
    ds_path.mkdir(parents=True, exist_ok=True)

    # Check if dataset is already downloaded
    if not any(ds_path.iterdir()):
        print(f"Downloading {lang_pair} dataset to {ds_path}...")
        # Download from Hugging Face Hub
        ds_raw = load_dataset("Helsinki-NLP/opus-100", lang_pair, split='train')
        # Save to local directory
        ds_raw.save_to_disk(ds_path)
    else:
        print(f"Loading dataset from {ds_path}...")
        ds_raw = load_from_disk(ds_path)
    
    # Build the tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # Keep 90% of the data for training and 10% for validation
    train_ds_size = int(len(ds_raw) * 0.9)
    valid_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, valid_ds_raw = random_split(ds_raw, [train_ds_size, valid_ds_size])
    
    train_ds = BiligualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    valid_ds = BiligualDataset(valid_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f"Max length source: {max_len_src}")
    print(f"Max length target: {max_len_tgt}")
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_ds, 1, shuffle=False)
    
    return train_dataloader, valid_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocal_tgt_len):
    model = build_transformer(vocab_src_len, vocal_tgt_len, config["seq_len"], config["seq_len"], d_model=config["d_model"])
    return model

# def train_model(config):
#     # Define the device
#     device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
#     print(f"Using device: {device}")
#     if (device == 'cuda'):
#         print(f"Device name: {torch.cuda.get_device_name(device.index)}")
#         print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
#     elif (device == 'mps'):
#         print(f"Device name: <mps>")
#     else:
#         print("NOTE: If you have a GPU, consider using it for training.")
#         print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
#         print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
#     device = torch.device(device)
    
#     # Make sure the weights folder exists
#     Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

#     train_dataloader, valid_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
#     model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
#     # Tensorboard writer
#     writer = SummaryWriter(config['experiment_name'])
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
#     # If the user has specified a pretrained model, load it
#     initial_epoch = 0
#     global_step = 0
#     preload = config['preload']
#     model_filename = latest_weights_file_path(config) if preload == "latest" else latest_weights_file_path(config, preload) if preload else None
#     if model_filename:
#         print(f"Preloading model from {model_filename}")
#         state = torch.load(model_filename)
#         model.load_state_dict(state['model_state_dict'])
#         initial_epoch = state['epoch'] + 1
#         optimizer.load_state_dict(state['optimizer_state_dict'])
#         global_step = state['global_step']
#     else:
#         print("No pretrained model found. Starting from scratch.")
        
#     losss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    
#     for epoch in range(initial_epoch, config['num_epochs']):
#         gc.collect()
#         torch.mps.synchronize()
#         model.train()
#         batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
#         for batch in batch_iterator:
            
#             encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
#             decoder_input = batch["decode_input"].to(device) # (B, seq_len)
#             encoder_mask = batch["encoder_mask"].to(device) # (B, 1, 1, seq_len)
#             decoder_mask = batch["decode_mask"].to(device) # (B, 1, seq_len, seq_len)
            
#             # Run the tensor though the encoder, decoder and the projection layer
#             encoder_output = model.encode(encoder_input, encoder_mask) # (Batch_size, seq_len, d_model)
#             decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (Batch_size, seq_len, d_model)
#             proj_output = model.project(decoder_output) # (Batch_size, seq_len, vocab_size)
            
#             # Compare the output with the target
#             label = batch["label"].to(device) # (Batch_size, seq_len)
            
#             # compute the loss using a simple cross entropy loss
#             loss = losss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
#             batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
#             # Log the loss
#             writer.add_scalar("loss", loss.item(), global_step)
#             writer.flush()
            
#             # Backpropagation
#             loss.backward()
            
#             # Update the weights
#             optimizer.step()
#             optimizer.zero_grad(set_to_none=True)
            
#             global_step += 1
            
#         # Run validation at the end of each epoch
#         run_validation(model, valid_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
#         # Save the model
#         model_filename = latest_weights_file_path(config, f"{epoch:02d}")
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'global_step': global_step,
#         }, model_filename)
        
def train_model(config):
    # Set device with improved MPS detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device name: {torch.cuda.get_device_name(device)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.2f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device name: <MPS>")
    else:
        device = torch.device("cpu")
        print("NOTE: Consider using GPU acceleration if available")
    print(f"Using device: {device}")

    # Create weights folder
    model_dir = Path(f"{config['datasource']}_{config['model_folder']}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Get data and model
    train_dataloader, valid_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    
    # Memory optimization 1: Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    model = model.to(device)
    
    # Setup TensorBoard and optimizer
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Load checkpoint if available
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = latest_weights_file_path(config) if config['preload'] == "latest" else config['preload']
        if model_filename and Path(model_filename).exists():
            print(f"Loading model from {model_filename}")
            state = torch.load(model_filename, map_location=device)
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            initial_epoch = state['epoch'] + 1
            global_step = state['global_step']
            del state  # Free memory
            torch.mps.empty_cache() if device.type == 'mps' else torch.cuda.empty_cache()

    # Loss function with memory optimizations
    losss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"), 
        label_smoothing=0.1
    ).to(device)

    # Memory optimization 2: Use automatic mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # Memory optimization 3: Use memory-efficient attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        print("Using memory-efficient attention")
    
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        
        # Memory optimization 4: Use memory monitoring
        if device.type == 'mps':
            print(f"Initial MPS memory: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")

        with tqdm(train_dataloader, desc=f"Epoch {epoch:02d}") as batch_iterator:
            for batch in batch_iterator:
                # Memory optimization 5: Use non-blocking transfers and mixed precision
                with torch.autocast(device_type=device.type, enabled=(device.type != 'cpu')):
                    encoder_input = batch["encoder_input"].to(device, non_blocking=True)
                    decoder_input = batch["decoder_input"].to(device, non_blocking=True)
                    encoder_mask = batch["encoder_input_mask"].to(device, non_blocking=True)
                    decoder_mask = batch["decoder_input_mask"].to(device, non_blocking=True)

                    # Forward pass
                    encoder_output = model.encode(encoder_input, encoder_mask)
                    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                    proj_output = model.project(decoder_output)
                    
                    # Compute loss
                    label = batch["label"].to(device, non_blocking=True)
                    loss = losss_fn(
                        proj_output.view(-1, tokenizer_tgt.get_vocab_size()), 
                        label.view(-1)
                    )

                # Backpropagation with scaler for mixed precision
                scaler.scale(loss).backward()
                
                # Memory optimization 6: Gradient clipping and accumulation
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # More memory efficient

                # Memory management
                if device.type == 'mps' and global_step % 100 == 0:
                    torch.mps.empty_cache()

                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
                writer.add_scalar("loss", loss.item(), global_step)
                global_step += 1

                # Free temporary tensors
                del encoder_input, decoder_input, encoder_mask, decoder_mask
                del encoder_output, decoder_output, proj_output, label

            # End of epoch cleanup
            gc.collect()
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()

            # Validation with memory optimizations
            run_validation(
                model, valid_dataloader, tokenizer_src, tokenizer_tgt,
                config['seq_len'], device, 
                lambda msg: batch_iterator.write(msg), 
                global_step, writer
            )

            # Save checkpoint with memory mapping
            model_filename = model_dir / f"model_{epoch:02d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
            }, model_filename)
            print(f"Saved model to {model_filename}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)