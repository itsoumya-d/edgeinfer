export class Tokenizer {
  private vocab: Record<string, number> = {};

  constructor(vocab: Record<string, number>) {
    this.vocab = vocab;
  }

  encode(text: string): Int32Array {
    const words = text.toLowerCase().split(/\s+/);
    const tokens = words.map(w => this.vocab[w] || 0);
    return new Int32Array(tokens);
  }

  decode(tokens: Int32Array): string {
    const invVocab = Object.fromEntries(Object.entries(this.vocab).map(([k, v]) => [v, k]));
    return Array.from(tokens).map(t => invVocab[t] || '[UNK]').join(' ');
  }
}
