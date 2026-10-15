from collections import Counter
import time

class BPETokenizer:
    def __init__(self, corpus, vocab_size) -> None:
        # 1. Initialize with all unique characters in the corpus
        self.vocab = list(set(corpus))
        self.merges = {}
        
        # 2. Count word frequencies and split words into characters
        words = corpus.split()
        word_freqs = Counter(words)
        # Represent each word as a tuple of characters
        word_splits = {word: tuple(word) for word in word_freqs.keys()}
        
        # 3. Iteratively find the most frequent pair and merge it
        while len(self.vocab) < vocab_size:
            pairs = Counter()
            for word, freq in word_freqs.items():
                symbols = word_splits[word]
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i+1])] += freq
            
            if not pairs:
                break
                
            # Find the most common pair
            best_pair = max(pairs, key=pairs.get)
            
            # If the best pair only appears once, we can stop (or keep going if you want exact vocab size)
            if pairs[best_pair] < 2:
                print("No more frequent pairs to merge.")
                break

            # Create the new merged token
            new_token = ''.join(best_pair)
            self.vocab.append(new_token)
            self.merges[best_pair] = new_token

            # Update all word splits with the new merge
            for word in word_splits:
                new_split = []
                i = 0
                symbols = word_splits[word]
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(symbols[i])
                        i += 1
                word_splits[word] = tuple(new_split)

        # Create token-to-id mappings
        self.tokentoid = {t: i for i, t in enumerate(self.vocab)}
        self.idtotoken = {i: t for t, i in self.tokentoid.items()}

    def get_vocab(self):
        return self.vocab

    def encode(self, text):
        # FIX: Use list(text) to preserve spaces and punctuation as individual tokens
        tokens = list(text)
        
        # Apply merges in the exact order they were learned
        for pair, merge in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    new_tokens.append(merge)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        # Convert tokens to their integer IDs (optional, but standard for LLMs)
        # If you want to return strings instead, just return `tokens`
        return [self.tokentoid.get(t, self.tokentoid.get('<UNK>', 0)) for t in tokens]

    def decode(self, token_ids):
        # Convert IDs back to string tokens, then join them
        tokens = [self.idtotoken.get(idx, '<UNK>') for idx in token_ids]
        return "".join(tokens)

# --- Example Usage ---
if __name__ == "__main__":
    corpus = """I was a high school school girl in 2015 so you know I’ve taken a lot of Buzzfeed quizzes. 
    There was the one about my inner spud (tater tot), the one about which possible Illuminati member I am (Bill Gates)."""
    
    start = time.time()
    # Train a small tokenizer
    tokenizer = BPETokenizer(corpus, vocab_size=100)
    end = time.time()
    
    text = "I was a high school girl."
    
    # Encode
    encoded = tokenizer.encode(text)
    print("Encoded IDs:", encoded)
    
    # Decode
    decoded = tokenizer.decode(encoded)
    print("Decoded Text:", decoded)
    
    print(f"Time taken: {end-start:.4f} seconds")