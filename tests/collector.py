import os
import sys
import json
import unittest
import subprocess
from time import time
from pathlib import Path

import torch
from torch import tensor
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from llm_layer_collector.layer_collector import LlmLayerCollector
from llm_layer_collector.compute import compute_embedding, compute_head
from llm_layer_collector.cache import get_shard_files
from llm_layer_collector.helpers import load_shard_tensor
from llm_layer_collector.load_layer import files_to_load_for_layer

PROMPT = "The quick brown fox jumps over the "

def clone_model(model_id: str, model_dir: str):
    repo_url = f"https://huggingface.co/{model_id}"

    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True)
    subprocess.run(["git", "clone", repo_url, model_dir])
    subprocess.run(["git", "lfs", "install"], cwd=model_dir, check=True)
    subprocess.run(["git", "lfs", "pull"], cwd=model_dir, check=True)

def get_cache_file(model_id: str):
    model_str = model_id.replace('/', '_')
    return f"data/{model_str}.json"

def get_model_dir(model_id: str):
    return f"models/{model_id}"

def ensure_model(model_id: str):
    model_dir = get_model_dir(model_id)
    if os.path.exists(model_dir):
        return
    clone_model(model_id, model_dir)

def test_cache(tst: unittest.TestCase, model_dir: str, cache_file: str, num_keys: int):
    collector = LlmLayerCollector(model_dir, cache_file)
    tst.assertEqual(len(collector.layer_files.keys()), num_keys)
    tst.assertTrue(os.path.exists(cache_file))
    tst.assertTrue(os.path.exists(collector.cache_file))
    with open(cache_file, 'r') as f:
        cache = json.load(f)
        tst.assertEqual(len(cache.keys()), num_keys)

def test_embedding(tst: unittest.TestCase, model_dir: str, cache_file: str, state_shape, position_ids_shape, position_embeddings_shape):
    collector = LlmLayerCollector(model_dir, cache_file)
    input_embedder = collector.load_input_embedding()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = tokenizer(PROMPT, return_tensors='pt')['input_ids']
    state = compute_embedding(input_embedder, input_ids, collector.config)
    tst.assertEqual(state.state.shape, state_shape)
    tst.assertEqual(state.position_ids.shape, position_ids_shape)
    tst.assertEqual(state.position_embeddings[0].shape, position_embeddings_shape)
    tst.assertEqual(state.position_embeddings[1].shape, position_embeddings_shape)

def test_norm(tst: unittest.TestCase, model_dir: str, cache_file: str, norm_dim: int):
    collector = LlmLayerCollector(model_dir, cache_file)
    norm = collector.load_norm()
    norm = norm.to('cpu')
    norm = norm.to(dtype=torch.float16)
    tst.assertEqual(norm.weight.shape, (norm_dim,))

def test_head(tst: unittest.TestCase, model_dir: str, cache_file: str, head_shape):
    collector = LlmLayerCollector(model_dir, cache_file)
    head = collector.load_head()
    tst.assertEqual(head.weight.shape, head_shape)

def test_layers(tst: unittest.TestCase, model_dir: str, cache_file: str, end_layer: int):
    start_time = time()
    collector = LlmLayerCollector(model_dir, cache_file)
    layers = collector.load_layer_set(0, end_layer)
    print(f"Time: {time() - start_time:.2f}s")
    tst.assertEqual(len(layers), end_layer+1)

def test_stack(tst: unittest.TestCase, model_dir: str, cache_file: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    chat = [
        {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    ]
    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    original_num_tokens = input_ids.shape[1]
    num_tokens = 4
    current_token = 0
    collector = LlmLayerCollector(model_dir, cache_file)
    input_embed = collector.load_input_embedding()
    head = collector.load_head()
    norm = collector.load_norm()
    layers = collector.load_layer_set(0, collector.config.num_hidden_layers - 1) # Ensure we load all layers....
    state = None
    while current_token < num_tokens:
        state = compute_embedding(input_embed, input_ids, collector.config, state)
        for lyr in layers:
            state.state = lyr(state)
        topk = 1
        result = compute_head(head, norm(state.state), topk)
        tst.assertEqual(result.shape, (1, topk))
        token_list = input_ids.tolist()[0]
        token_list.append(result[0][0].item())
        input_ids = tensor([token_list])
        current_token += 1
        print(current_token)
        print(tokenizer.decode(input_ids[0]))
    tst.assertGreater(input_ids.shape[1], original_num_tokens)

class LlmLayerCollectorTests(unittest.TestCase):
    def test_qwen3_2B(self):
        model_id = "Qwen/Qwen3-1.7B"
        model_dir = get_model_dir(model_id)
        cache_file = get_cache_file(model_id)
        ensure_model(model_id)
        test_cache(self, model_dir, cache_file, 311)
        test_embedding(self, model_dir, cache_file, (1, 8, 2048), (1, 8), (1, 8, 128))
        test_norm(self, model_dir, cache_file, 2048)
        test_head(self, model_dir, cache_file, (151936, 2048))
        test_layers(self, model_dir, cache_file, 10)
        test_stack(self, model_dir, cache_file)

    # def test_gemma3_1B(self):
    #     model_id = "google/gemma-3-1b-it"
    #     model_dir = get_model_dir(model_id)
    #     cache_file = get_cache_file(model_id)
    #     ensure_model(model_id)

    # def test_gpt_oss_20B(self):
    #     model_id = "openai/gpt-oss-20b"
    #     model_dir = get_model_dir(model_id)
    #     cache_file = get_cache_file(model_id)
    #     ensure_model(model_id
    #     test_cache(self, model_dir, cache_file, 459)
    #     test_embedding(self, model_dir, cache_file, (1, 8, 2048), (1, 8), (1, 8, 128))
    #     test_norm(self, model_dir, cache_file, 2048)
    #     test_head(self, model_dir, cache_file, (151936, 2048))
    #     test_layers(self, model_dir, cache_file, 10)

    def test_exceptions(self):
        model_id = "Qwen/Qwen3-1.7B"
        model_dir = get_model_dir(model_id)
        cache_file = get_cache_file(model_id)
        collector = LlmLayerCollector(model_dir, cache_file)

        try:
            os.mkdir('shard_test')
            get_shard_files(collector.shard_pattern, 'shard_test')
            self.fail("Should have thrown an exception")
        except Exception:
            pass
        os.rmdir('shard_test')
        
        try:
            load_shard_tensor(collector.layer_files, collector.model_dir, 'bad_layer', 'cpu', torch.float16)
            self.fail("Should have thrown an exception")
        except ValueError:
            pass

        try:
            os.mkdir('bad_dir')
            LlmLayerCollector('bad_dir', cache_file)
            self.fail("Should have thrown an exception")
        except FileNotFoundError:
            pass

        os.rmdir('bad_dir')

        try:
            os.remove(cache_file)
            collector._read_cache()
            self.fail("Should have thrown an exception")
        except FileNotFoundError:
            pass
        
        try:
            files_to_load_for_layer('bad_key', [])
            self.fail("Should have thrown an exception")
        except Exception:
            pass

# If you want to run these tests directly from the command line:
if __name__ == '__main__':
    unittest.main()
