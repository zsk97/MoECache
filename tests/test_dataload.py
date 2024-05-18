from MoECache.load_utils import process_dataset
from datasets import load_dataset
from transformers import AutoTokenizer

if __name__ == '__main__':
    dataset = load_dataset("marsggbo/bigbench4switch32_pattern_predictor")
    batch_size = 8

    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-16")
    tokenizer.padding_side = 'left'

    for input_id, decode_id, pattern in process_dataset(dataset, tokenizer, batch_size):
        print("Batch input ", decode_id)