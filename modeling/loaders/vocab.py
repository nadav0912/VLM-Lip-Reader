import json

class Vocabulary:
    def __init__(self, word_list):
        """
        Initializes the vocabulary from a raw list of words.
        :param word_list: A list of strings (words/tokens).
        """
        self.word2idx = {}
        self.idx2word = {}
        
        # Define special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        
        # Build the vocab immediately
        self._build_vocab(word_list)

    def _build_vocab(self, word_list):
        # 1. Get unique words and sort them for deterministic order
        unique_words = sorted(list(set(word_list)))
        
        # 2. Start indices
        # Index 0: Padding (Crucial for deep learning masking)
        # Index 1: Unknown (For words not seen during training)
        self.word2idx = {
            self.pad_token: 0,
            self.unk_token: 1
        }
        
        # 3. Add the real words starting from index 2
        for i, word in enumerate(unique_words):
            self.word2idx[word] = i + 2
            
        # 4. Create reverse mapping
        self.idx2word = {i: word for word, i in self.word2idx.items()}
        
        print(f"📚 Vocabulary built: {len(self.word2idx)} tokens ",
              f"({len(unique_words)} unique words + 2 specials).")

    def __len__(self):
        return len(self.word2idx)

    def text_to_id(self, text):
        # Returns the ID of the word, or the ID of <UNK> if not found
        return self.word2idx.get(text, self.word2idx[self.unk_token])

    def id_to_text(self, idx):
        # Returns the word, or <UNK> if the ID is invalid
        return self.idx2word.get(idx, self.unk_token)

    @classmethod
    def from_json_file(cls, json_path):
        """
        Helper method (Alternative Constructor) to load directly from our 
        Single-Word-Dataset format (filename -> word).
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract just the values (the words)
        all_words = list(data.values())
        return cls(all_words)