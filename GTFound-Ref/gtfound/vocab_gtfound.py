"""Vocabulary reader used by the GTFound release."""

class GTFoundVocab:
    UNK = "[UNK]"

    def __init__(self, vocab_path):
        with open(vocab_path, encoding="utf-8") as handle:
            self.itos = [line.rstrip("\n") for line in handle]
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi[self.UNK])

    def __len__(self):
        return len(self.itos)
