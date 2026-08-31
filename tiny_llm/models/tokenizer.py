"""
Byte-Pair Encoding (BPE) tokenizer for Tiny LLM.
Handles encoding/decoding of text to/from token IDs.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

import torch


class BPETokenizer:
    """
    Simple Byte-Pair Encoding tokenizer.
    
    Vocabulary:
    - ID 0: <pad>
    - ID 1: <unk> (unknown)
    - ID 2: <bos> (beginning of sequence)
    - ID 3: <eos> (end of sequence)
    - ID 4+: learned BPE tokens
    """
    
    def __init__(self, vocab_size: int = 2000, special_tokens: Optional[Dict[str, int]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        self.reverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        # Main vocabulary
        self.word_tokenizer = re.compile(r"\w+|[^\w\s]")
        self.vocab: Dict[str, int] = self.special_tokens.copy()
        self.merges: List[Tuple[str, str]] = []
        self.is_trained = False
    
    def train(self, texts: List[str], num_merges: int = 2000):
        """
        Train the tokenizer on a corpus using BPE.
        
        Args:
            texts: List of text documents to train on
            num_merges: Number of merge operations to perform
        """
        # Tokenize all texts into words
        all_words = []
        for text in texts:
            words = self.word_tokenizer.findall(text.lower())
            all_words.extend(words)
        
        if not all_words:
            self.is_trained = True
            return
        
        # Initialize with character-level tokens
        word_freqs: Dict[Tuple[str, ...], int] = {}
        for word in all_words:
            word_tuple = tuple(word) + ("<end>",)
            word_freqs[word_tuple] = word_freqs.get(word_tuple, 0) + 1
        
        # Perform BPE merges
        num_tokens = len(self.special_tokens) + 256  # Start with ASCII chars
        current_vocab = set()
        for word in word_freqs:
            for token in word:
                current_vocab.add(token)
        
        for i in range(num_merges):
            if len(current_vocab) >= self.vocab_size - len(self.special_tokens):
                break
            
            # Count token pairs
            pairs: Dict[Tuple[str, str], int] = {}
            for word, freq in word_freqs.items():
                for j in range(len(word) - 1):
                    pair = (word[j], word[j + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            
            # Merge the pair in all words
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = self._merge_pair(word, best_pair)
                new_word_freqs[new_word] = freq
            
            word_freqs = new_word_freqs
            
            # Update vocabulary
            for word in word_freqs:
                for token in word:
                    current_vocab.add(token)
        
        # Assign vocabulary IDs
        sorted_vocab = sorted(current_vocab)
        for i, token in enumerate(sorted_vocab):
            if token not in self.special_tokens:
                self.vocab[token] = len(self.special_tokens) + i
        
        self.is_trained = True
    
    def _merge_pair(self, word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        """Merge a specific pair in a word."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Add <bos> and <eos> tokens
        
        Returns:
            List of token IDs
        """
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.special_tokens["<bos>"])
        
        # Tokenize into words
        words = self.word_tokenizer.findall(text.lower())
        
        for word in words:
            # Convert word to subword tokens
            word_tokens = list(word)
            
            # Apply learned merges
            for pair in self.merges:
                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        new_tokens.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            
            # Convert to IDs
            for token in word_tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    token_ids.append(self.special_tokens["<unk>"])
        
        if add_special_tokens:
            token_ids.append(self.special_tokens["<eos>"])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
        
        Returns:
            Decoded text
        """
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        tokens = []
        for token_id in token_ids:
            if token_id in reverse_vocab:
                token = reverse_vocab[token_id]
            elif token_id in self.reverse_special_tokens:
                token = self.reverse_special_tokens[token_id]
            else:
                token = "<unk>"
            
            if skip_special_tokens and token in self.special_tokens:
                continue
            
            tokens.append(token)
        
        # Join tokens
        text = "".join(tokens)
        text = text.replace("<end>", " ")
        return text.strip()
    
    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
            "is_trained": self.is_trained,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load tokenizer from file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        self.vocab = {k: int(v) for k, v in data["vocab"].items()}
        self.merges = [tuple(m) for m in data["merges"]]
        self.special_tokens = data["special_tokens"]
        self.vocab_size = data["vocab_size"]
        self.is_trained = data["is_trained"]
        self.reverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens (not IDs)."""
        words = self.word_tokenizer.findall(text.lower())
        tokens = []
        for word in words:
            tokens.extend(list(word))
        return tokens
