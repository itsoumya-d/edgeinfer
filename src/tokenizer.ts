// Copyright (c) 2024-2026 Soumya Debnath. All Rights Reserved.
// Licensed under the Business Source License 1.1 (BSL 1.1).
// See LICENSE file for details. Production use requires a paid license.
// Contact: soumyadebnath1661@gmail.com

export interface TokenizerConfig {
  vocab: Record<string, number>;
  /**
   * BPE merge rules, either `"a b"` (older HuggingFace tokenizer.json) or
   * `["a", "b"]` (newer tokenizer.json). Accepted and normalised, but note
   * that `encode()` currently implements WordPiece only — see `applyBPE`.
   */
  merges?: Array<string | [string, string]>;
  specialTokens?: Record<string, number>;
  maxLength?: number;
  padTokenId?: number;
  unkTokenId?: number;
  clsTokenId?: number;
  sepTokenId?: number;
}

export interface EncodingResult {
  inputIds: Int32Array;
  attentionMask: Int32Array;
  tokenTypeIds?: Int32Array;
}

/**
 * Production BPE/WordPiece tokenizer for ONNX model inference.
 * Supports loading vocabularies from JSON, encoding text to token IDs,
 * and decoding token IDs back to text.
 */
export class Tokenizer {
  private vocab: Map<string, number>;
  private inverseVocab: Map<number, string>;
  private merges: Array<[string, string]>;
  private specialTokens: Map<string, number>;
  private maxLength: number;
  private padTokenId: number;
  private unkTokenId: number;
  private clsTokenId: number;
  private sepTokenId: number;
  private bpeCache: Map<string, string[]>;

  constructor(config: TokenizerConfig) {
    this.vocab = new Map(Object.entries(config.vocab));
    this.inverseVocab = new Map<number, string>();
    for (const [token, id] of this.vocab) {
      this.inverseVocab.set(id, token);
    }

    // `merges` may arrive in either HuggingFace form:
    //   older tokenizer.json: ["a b", "lo w"]         (space-separated string)
    //   newer tokenizer.json: [["a","b"], ["lo","w"]] (pre-split pair)
    // Calling .split() on the array form throws "m.split is not a function",
    // so both shapes are handled here.
    this.merges = ((config.merges || []) as Array<string | [string, string]>).map(m => {
      if (Array.isArray(m)) {
        return [m[0], m[1]] as [string, string];
      }
      const parts = String(m).split(' ');
      return [parts[0], parts[1]] as [string, string];
    });

    this.specialTokens = new Map(Object.entries(config.specialTokens || {}));
    this.maxLength = config.maxLength || 512;
    this.padTokenId = config.padTokenId ?? 0;
    this.unkTokenId = config.unkTokenId ?? 100;
    this.clsTokenId = config.clsTokenId ?? 101;
    this.sepTokenId = config.sepTokenId ?? 102;
    this.bpeCache = new Map();
  }

  /**
   * Load tokenizer configuration from a JSON URL (tokenizer.json or vocab.json).
   */
  static async fromUrl(url: string): Promise<Tokenizer> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load tokenizer from ${url}: ${response.status}`);
    const config = await response.json();

    // Handle HuggingFace tokenizer.json format
    if (config.model && config.model.vocab) {
      const specialTokens: Record<string, number> = {};
      if (config.added_tokens) {
        for (const t of config.added_tokens) {
          specialTokens[t.content] = t.id;
        }
      }
      return new Tokenizer({
        vocab: config.model.vocab,
        merges: config.model.merges || [],
        specialTokens,
        maxLength: config.truncation?.max_length || 512,
        padTokenId: specialTokens['[PAD]'] ?? 0,
        unkTokenId: specialTokens['[UNK]'] ?? 100,
        clsTokenId: specialTokens['[CLS]'] ?? 101,
        sepTokenId: specialTokens['[SEP]'] ?? 102,
      });
    }

    // Handle simple vocab.json format (key -> id mapping)
    return new Tokenizer({ vocab: config });
  }

  /**
   * Encode text into token IDs with attention mask.
   * Handles WordPiece tokenization for BERT-like models.
   */
  encode(text: string, options?: { addSpecialTokens?: boolean; maxLength?: number; padding?: boolean }): EncodingResult {
    const addSpecial = options?.addSpecialTokens !== false;
    const maxLen = options?.maxLength || this.maxLength;
    const padding = options?.padding ?? false;

    // Pre-tokenize: lowercase, split on whitespace and punctuation
    const words = this.preTokenize(text);
    
    // WordPiece / BPE tokenization
    const tokenIds: number[] = [];
    
    if (addSpecial) {
      tokenIds.push(this.clsTokenId);
    }

    for (const word of words) {
      const subTokens = this.tokenizeWord(word);
      for (const subToken of subTokens) {
        if (tokenIds.length >= maxLen - (addSpecial ? 1 : 0)) break;
        tokenIds.push(subToken);
      }
    }

    if (addSpecial) {
      tokenIds.push(this.sepTokenId);
    }

    // Truncate to maxLength
    const truncated = tokenIds.slice(0, maxLen);
    
    // Guard against empty encoding producing empty tensors (crashes ONNX)
    if (truncated.length === 0) {
      const fallback = new Int32Array([this.padTokenId]);
      return {
        inputIds: fallback,
        attentionMask: new Int32Array([0]),
      };
    }

    // Build attention mask
    const attentionMask = new Int32Array(padding ? maxLen : truncated.length);
    attentionMask.fill(1, 0, truncated.length);

    // Pad if requested
    const inputIds = new Int32Array(padding ? maxLen : truncated.length);
    inputIds.set(truncated);
    if (padding) {
      inputIds.fill(this.padTokenId, truncated.length);
    }

    return { inputIds, attentionMask };
  }

  /**
   * Decode token IDs back to text.
   */
  decode(tokens: Int32Array | number[]): string {
    const tokenArray = tokens instanceof Int32Array ? Array.from(tokens) : tokens;
    const words: string[] = [];

    for (const id of tokenArray) {
      if (id === this.padTokenId || id === this.clsTokenId || id === this.sepTokenId) continue;
      const token = this.inverseVocab.get(id);
      if (!token) {
        words.push('[UNK]');
      } else if (token.startsWith('##')) {
        // WordPiece continuation token
        if (words.length > 0) {
          words[words.length - 1] += token.slice(2);
        } else {
          words.push(token.slice(2));
        }
      } else {
        words.push(token);
      }
    }

    return words.join(' ');
  }

  /**
   * Get vocabulary size.
   */
  get vocabSize(): number {
    return this.vocab.size;
  }

  /**
   * Pre-tokenize text: lowercase, split on whitespace and punctuation boundaries.
   */
  private preTokenize(text: string): string[] {
    const normalized = text.toLowerCase().trim();
    if (!normalized) return [];
    
    // Split on whitespace and keep punctuation as separate tokens
    // Uses Unicode-aware patterns to handle emojis, CJK, Arabic, accented chars
    const tokens: string[] = [];
    let current = '';

    for (const char of normalized) {
      if (/\s/.test(char)) {
        if (current) tokens.push(current);
        current = '';
      } else if (/[.,!?;:'"()\[\]{}\-\/\\@#$%^&*~`|<>]/.test(char) || /\p{P}/u.test(char)) {
        if (current) tokens.push(current);
        tokens.push(char);
        current = '';
      } else {
        current += char;
      }
    }
    if (current) tokens.push(current);
    return tokens;
  }

  /**
   * Tokenize a single word using WordPiece algorithm.
   */
  private tokenizeWord(word: string): number[] {
    // Check if whole word is in vocabulary
    if (this.vocab.has(word)) {
      return [this.vocab.get(word)!];
    }

    // WordPiece: greedily match longest subword from left to right
    const tokens: number[] = [];
    let start = 0;

    while (start < word.length) {
      let end = word.length;
      let found = false;

      while (start < end) {
        let substr = word.slice(start, end);
        if (start > 0) {
          substr = '##' + substr;
        }

        if (this.vocab.has(substr)) {
          tokens.push(this.vocab.get(substr)!);
          found = true;
          break;
        }
        end--;
      }

      if (!found) {
        // Character not in vocab, use [UNK]
        tokens.push(this.unkTokenId);
        start++;
      } else {
        start = end;
      }
    }

    return tokens;
  }

  /**
   * Apply BPE merges to a word (for BPE-style tokenizers like GPT-2).
   */
  private applyBPE(word: string): string[] {
    const cached = this.bpeCache.get(word);
    if (cached) return cached;

    let pairs = this.getCharPairs(word.split(''));
    if (pairs.length === 0) return [word];

    let tokens = word.split('');

    for (const [first, second] of this.merges) {
      const newTokens: string[] = [];
      let i = 0;

      while (i < tokens.length) {
        const j = tokens.indexOf(first, i);
        if (j === -1) {
          newTokens.push(...tokens.slice(i));
          break;
        }

        newTokens.push(...tokens.slice(i, j));

        if (j < tokens.length - 1 && tokens[j + 1] === second) {
          newTokens.push(first + second);
          i = j + 2;
        } else {
          newTokens.push(tokens[j]);
          i = j + 1;
        }
      }
      tokens = newTokens;
    }

    this.bpeCache.set(word, tokens);
    return tokens;
  }

  private getCharPairs(chars: string[]): Array<[string, string]> {
    const pairs: Array<[string, string]> = [];
    for (let i = 0; i < chars.length - 1; i++) {
      pairs.push([chars[i], chars[i + 1]]);
    }
    return pairs;
  }
}
