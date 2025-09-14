import os
import sys
import json
import unittest
from time import time

import torch
from torch import tensor
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from llm_layer_collector.layer_collector import LlmLayerCollector
from llm_layer_collector.compute import compute_embedding, compute_head
from llm_layer_collector.cache import get_shard_files
from llm_layer_collector.helpers import load_shard_tensor
from llm_layer_collector.load_layer import files_to_load_for_layer

CACHE_FILE_Q30B_A3: str = 'data/Qwen3-30B-A3B-Thinking-2507-cache.json'
MODEL_DIR_Q30B_A3: str = 'models/Qwen/Qwen3-30B-A3B-Thinking-2507/data'

CACHE_FILE_1B: str = 'data/Llama3.2-1B-instruct-cache.json'
MODEL_DIR_1B: str = 'models/Llama-3.2-1B-Instruct'

CACHE_FILE_Q2B: str = 'data/Qwen3-1.7B-cache.json'
MODEL_DIR_Q2B: str = 'models/Qwen3-1.7B'

CACHE_FILE_G1B: str = 'data/Gemma-3-270m-it-cache.json'
MODEL_DIR_G1B: str = 'models/Gemma-3-270m-it'

CACHE_FILE_8B: str = 'data/Meta-Llama-3-8B-cache.json'
MODEL_DIR_8B: str = 'models/Meta-Llama-3-8B'

CACHE_FILE_70B: str = 'data/Meta-Llama-3.3-70B-cache.json'
MODEL_DIR_70B: str = 'models/llama-3.3-70B-Instruct'

NUM_KEYS_1B = 146
NUM_KEYS_8B = 291

PROMPT = "The quick brown fox jumps over the "

class LlmLayerCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(CACHE_FILE_1B):
            os.remove(CACHE_FILE_1B)
        if os.path.exists(CACHE_FILE_8B):
            os.remove(CACHE_FILE_8B)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(CACHE_FILE_1B):
            os.remove(CACHE_FILE_1B)
        if os.path.exists(CACHE_FILE_8B):
            os.remove(CACHE_FILE_8B)

    def test_cache_1B(self):
        collector = LlmLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B)
        self.assertEqual(len(collector.layer_files.keys()), NUM_KEYS_1B)
        self.assertTrue(os.path.exists(CACHE_FILE_1B))
        self.assertTrue(os.path.exists(collector.cache_file))
        with open(CACHE_FILE_1B, 'r') as f:
            cache = json.load(f)
            self.assertEqual(len(cache.keys()), NUM_KEYS_1B)

    def test_cache_8B(self):
        collector = LlmLayerCollector(MODEL_DIR_8B, CACHE_FILE_8B)
        self.assertEqual(len(collector.layer_files.keys()), NUM_KEYS_8B)
        self.assertTrue(os.path.exists(CACHE_FILE_8B))
        self.assertTrue(os.path.exists(collector.cache_file))
        with open(collector.cache_file, 'r') as f:
            cache = json.load(f)
            self.assertEqual(len(cache.keys()), NUM_KEYS_8B)
    
    def test_gpu_input_embedding(self):
        collector = LlmLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B, device="cuda")
        input_embedder = collector.load_input_embedding()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_1B)
        input_ids = tokenizer(PROMPT, return_tensors="pt")['input_ids']
        state = compute_embedding(input_embedder, input_ids, collector.config)

    def test_gpu_layers(self):
        collector = LlmLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B, device="cuda")
        layers = collector.load_layer_set(0, 5, "cuda")
        self.assertEqual("cuda:0", str(layers[0].self_attn.q_proj.weight.device))

    def test_input_embedding_1B(self):
        collector = LlmLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B)
        input_embedder = collector.load_input_embedding()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_1B)
        input_ids = tokenizer(PROMPT, return_tensors='pt')['input_ids']
        state = compute_embedding(input_embedder, input_ids, collector.config)
        self.assertEqual(state.state.shape, (1, 9, 2048))
        self.assertEqual(state.position_ids.shape, (1, 9))
        self.assertEqual(state.position_embeddings[0].shape, (1, 9, 64))
        self.assertEqual(state.position_embeddings[1].shape, (1, 9, 64))
        self.assertEqual(state.causal_mask.shape, (1, 1, 9, 9))

    def test_input_embedding_8B(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_8B)
        input_ids = tokenizer(PROMPT, return_tensors='pt')['input_ids']
        collector = LlmLayerCollector(MODEL_DIR_8B, CACHE_FILE_8B)
        input_embedder = collector.load_input_embedding()
        state = compute_embedding(input_embedder, input_ids, collector.config)
        self.assertEqual(state.state.shape, (1, 9, 4096))
        self.assertEqual(state.position_ids.shape, (1, 9))
        self.assertEqual(state.position_embeddings[0].shape, (1, 9, 128))
        self.assertEqual(state.position_embeddings[1].shape, (1, 9, 128))
        self.assertEqual(state.causal_mask.shape, (1, 1, 9, 9))

    def test_norm_1B(self):
        collector = LlmLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B)
        norm = collector.load_norm()
        norm = norm.to('cpu')
        norm = norm.to(dtype=torch.float16)
        self.assertEqual(norm.weight.shape, (2048,))
    
    def test_norm_8B(self):
        collector = LlmLayerCollector(MODEL_DIR_8B, CACHE_FILE_8B)
        norm = collector.load_norm()
        self.assertEqual(norm.weight.shape, (4096,))
    
    def test_head_1B(self):
        collector = LlmLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B)
        head = collector.load_head()
        self.assertEqual(head.weight.shape, (128256, 2048))
    
    def test_head_8B(self):
        collector = LlmLayerCollector(MODEL_DIR_8B, CACHE_FILE_8B)
        head = collector.load_head()
        self.assertEqual(head.weight.shape, (128256, 4096))

    def test_layers_1B(self):
        start_time = time()
        collector = LlmLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B)
        layers = collector.load_layer_set(0, 9)
        print(f"Time: {time() - start_time:.2f}s")
        self.assertEqual(len(layers), 10)

    def test_layers_8B(self):
        start_time = time()
        collector = LlmLayerCollector(MODEL_DIR_8B, CACHE_FILE_8B)
        layers = collector.load_layer_set(0, 9)
        print(f"Time: {time() - start_time:.2f}s")
        self.assertEqual(len(layers), 10)

    def test_layers_70B(self):
        start_time = time()
        collector = LlmLayerCollector(MODEL_DIR_70B, CACHE_FILE_70B)
        layers = collector.load_layer_set(0, 1)
        print(f"Time: {time() - start_time:.2f}s")
        self.assertEqual(len(layers), 2)

    def test_stack_1B(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_1B)
        chat = [
            {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
            {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
        ]
        input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        original_num_tokens = input_ids.shape[1]
        num_tokens = 40
        current_token = 0
        collector = LlmLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B)
        input_embedder = collector.load_input_embedding()
        head = collector.load_head()
        norm = collector.load_norm()
        layers = collector.load_layer_set(0, collector.config.num_hidden_layers - 1) # Ensure we load all layers....
        state = None
        while current_token < num_tokens:
            state = compute_embedding(input_embedder, input_ids, collector.config, state)
            for lyr in layers:
                state.state = lyr(state)
            topk = 1
            result = compute_head(head, norm(state.state), topk)
            self.assertEqual(result.shape, (1, topk))
            token_list = input_ids.tolist()[0]
            token_list.append(result[0][0].item())
            input_ids = tensor([token_list])
            current_token += 1
            print(tokenizer.decode(input_ids[0]))
        self.assertGreater(input_ids.shape[1], original_num_tokens)

    def test_stack_Qwen2B(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_Q2B)
        chat = [
            {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
            {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
        ]
        input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        original_num_tokens = input_ids.shape[1]
        num_tokens = 40
        current_token = 0
        collector = LlmLayerCollector(MODEL_DIR_Q2B, CACHE_FILE_Q2B)
        input_embedder = collector.load_input_embedding()
        head = collector.load_head()
        norm = collector.load_norm()
        layers = collector.load_layer_set(0, collector.config.num_hidden_layers - 1) # Ensure we load all layers....
        state = None
        while current_token < num_tokens:
            state = compute_embedding(input_embedder, input_ids, collector.config, state)
            for lyr in layers:
                state.state = lyr(state)
            topk = 1
            result = compute_head(head, norm(state.state), topk)
            self.assertEqual(result.shape, (1, topk))
            token_list = input_ids.tolist()[0]
            token_list.append(result[0][0].item())
            input_ids = tensor([token_list])
            current_token += 1
            print(current_token)
            print(tokenizer.decode(input_ids[0]))
        self.assertGreater(input_ids.shape[1], original_num_tokens)

    def test_stack_QwenMoe(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_Q30B_A3)
        chat = [
            {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
            {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
        ]
        input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        original_num_tokens = input_ids.shape[1]
        num_tokens = 40
        current_token = 0
        collector = LlmLayerCollector(MODEL_DIR_Q30B_A3, CACHE_FILE_Q30B_A3)
        input_embedder = collector.load_input_embedding()
        head = collector.load_head()
        norm = collector.load_norm()
        layers = collector.load_layer_set(0, collector.config.num_hidden_layers - 1) # Ensure we load all layers....
        state = None
        while current_token < num_tokens:
            state = compute_embedding(input_embedder, input_ids, collector.config, state)
            for lyr in layers:
                state.state = lyr(state)
            topk = 1
            result = compute_head(head, norm(state.state), topk)
            self.assertEqual(result.shape, (1, topk))
            token_list = input_ids.tolist()[0]
            token_list.append(result[0][0].item())
            input_ids = tensor([token_list])
            current_token += 1
            print(current_token)
            print(tokenizer.decode(input_ids[0]))
        self.assertGreater(input_ids.shape[1], original_num_tokens)

    def test_stack_Gemma1B(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_G1B)
        chat = [
            {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
            {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
        ]
        input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        original_num_tokens = input_ids.shape[1]
        num_tokens = 40
        current_token = 0
        collector = LlmLayerCollector(MODEL_DIR_G1B, CACHE_FILE_G1B)
        input_embedder = collector.load_input_embedding()
        head = collector.load_head()
        norm = collector.load_norm()
        layers = collector.load_layer_set(0, 15)
        while current_token < num_tokens:
            state = compute_embedding(input_embedder, input_ids, collector.config)
            for lyr in layers:
                state.state = lyr(state)
            topk = 1
            result = compute_head(head, norm(state.state), topk)
            self.assertEqual(result.shape, (1, topk))
            token_list = input_ids.tolist()[0]
            token_list.append(result[0][0].item())
            input_ids = tensor([token_list])
            current_token += 1
            print(current_token)
            print(f"\"{tokenizer.decode(input_ids[0])}\"")
        self.assertGreater(input_ids.shape[1], original_num_tokens)

    def test_stack_1B_no_cache_file(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_1B)
        input_ids = tokenizer(PROMPT, return_tensors='pt')['input_ids']
        original_num_tokens = input_ids.shape[1]
        num_tokens = 4
        current_token = 0
        os.remove(CACHE_FILE_1B)
        collector = LlmLayerCollector(MODEL_DIR_1B)
        self.assertFalse(os.path.exists(CACHE_FILE_1B))
        self.assertTrue(collector.layer_files is not None)
        input_embedder = collector.load_input_embedding()
        head = collector.load_head()
        norm = collector.load_norm()
        layers = collector.load_layer_set(0, 15)
        while current_token < num_tokens:
            state = compute_embedding(input_embedder, input_ids, collector.config)
            for lyr in layers:
                state.state = compute_layer(lyr, state)
            topk = 1
            result = compute_head(head, norm(state.state), topk)
            self.assertEqual(result.shape, (1, topk))
            token_list = input_ids.tolist()[0]
            token_list.append(result[0][0].item())
            input_ids = tensor([token_list])
            current_token += 1
        print(tokenizer.decode(input_ids[0]))
        self.assertGreater(input_ids.shape[1], original_num_tokens)

    def test_exceptions(self):
        collector = LlmLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B)

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
            LlmLayerCollector('bad_dir', CACHE_FILE_1B)
            self.fail("Should have thrown an exception")
        except FileNotFoundError:
            pass

        os.rmdir('bad_dir')

        try:
            os.remove(CACHE_FILE_1B)
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
