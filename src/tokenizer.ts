// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Soumya Debnath

import { withDownloadDeadline, DownloadOptions } from './download';

export interface TokenizerConfig {
  vocab: Record<string, number>;
  /** Keep the legacy uncased default; set false for cased WordPiece vocabularies. */
  lowercase?: boolean;
  /**
   * BPE merge rules, either `"a b"` (older HuggingFace tokenizer.json) or
   * `["a", "b"]` (newer tokenizer.json). Accepted and normalised. When present,
   * `encode()` applies these merges and preserves case by default; otherwise it
   * uses WordPiece.
   */
  merges?: Array<string | [string, string]>;
  specialTokens?: Record<string, number>;
  maxLength?: number;
  padTokenId?: number;
  unkTokenId?: number;
  clsTokenId?: number;
  sepTokenId?: number;
  /**
   * Optional marker prepended to the first BPE piece of each word, e.g. `"▁"`
   * for SentencePiece vocabularies or `"Ġ"` for byte-level BPE vocabularies.
   * Without it, pieces are looked up exactly as produced by the merge table.
   */
  bpeSpaceMarker?: string;
}

export interface EncodingResult {
  inputIds: Int32Array;
  attentionMask: Int32Array;
  tokenTypeIds?: Int32Array;
  /** int64 view for ONNX models declared as `tensor(int64)`. */
  inputIdsInt64: BigInt64Array;
  attentionMaskInt64: BigInt64Array;
  tokenTypeIdsInt64?: BigInt64Array;
}

/**
 * Production BPE/WordPiece tokenizer for ONNX model inference.
 * Supports loading vocabularies from JSON, encoding text to token IDs,
 * and decoding token IDs back to text.
 */
export class Tokenizer {
  private lowercase: boolean;
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
  private mergeRanks: Map<string, number>;
  private usesBPE: boolean;
  private bpeSpaceMarker: string;

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
    this.usesBPE = this.merges.length > 0;
    this.bpeSpaceMarker = config.bpeSpaceMarker ?? '';

    this.mergeRanks = new Map();
    this.merges.forEach(([first, second], rank) => {
      this.mergeRanks.set(`${first} ${second}`, rank);
    });

    // Older WordPiece-only callers relied on an implicit uncased default. A
    // merge table means the vocabulary is BPE (typically cased), so preserving
    // case is the only safe default there.
    this.lowercase = config.lowercase ?? !this.usesBPE;
  }

  /**
   * Load tokenizer configuration from a JSON URL (tokenizer.json or vocab.json).
   */
  static async fromUrl(url: string, options: DownloadOptions = {}): Promise<Tokenizer> {
    const config = await withDownloadDeadline(async signal => {
      const response = await fetch(url, { signal });
      if (!response.ok) {
        await response.body?.cancel();
        throw new Error(`Failed to load tokenizer from ${url}: ${response.status}`);
      }
      return response.json();
    }, options);

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
        lowercase: config.normalizer === undefined
          ? undefined
          : Tokenizer.normalizerLowercase(config.normalizer),
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

  private static normalizerLowercase(normalizer: unknown): boolean {
    if (normalizer === undefined) return true; // Legacy default for omitted config.
    if (normalizer === null) return false; // Explicit identity normalizer.
    if (typeof normalizer !== 'object') throw new Error('Unsupported tokenizer normalizer');
    const value = normalizer as Record<string, unknown>;
    if (value.type === 'Lowercase') return true;
    // Other normalization (accent stripping, CJK spacing, Unicode normalization)
    // is not implemented. Reject it instead of silently changing token IDs.
    if (value.type === 'BertNormalizer' && typeof value.lowercase === 'boolean'
      && value.strip_accents === false && value.clean_text === false && value.handle_chinese_chars === false) {
      return value.lowercase;
    }
    throw new Error('Unsupported tokenizer normalizer; only identity, Lowercase, or casing-only BertNormalizer is supported');
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
        inputIdsInt64: Tokenizer.toInt64(fallback),
        attentionMaskInt64: new BigInt64Array([0n]),
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

    return {
      inputIds,
      attentionMask,
      inputIdsInt64: Tokenizer.toInt64(inputIds),
      attentionMaskInt64: Tokenizer.toInt64(attentionMask),
    };
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
      } else if (token.startsWith('\u2581') || token.startsWith('\u0120')) {
        // SentencePiece (`▁`) and byte-level BPE (`Ġ`) word markers.
        words.push(token.slice(1));
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
   * Resolve a special token (e.g. `</s>`, `<|endoftext|>`) to its ID, whether it
   * was declared in `specialTokens` or is present in the main vocabulary.
   */
  getSpecialTokenId(token: string): number | undefined {
    return this.specialTokens.get(token) ?? this.vocab.get(token);
  }

  /**
   * Pre-tokenize text: lowercase, split on whitespace and punctuation boundaries.
   */
  private preTokenize(text: string): string[] {
    const normalized = (this.lowercase ? text.toLowerCase() : text).trim();
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
   * Tokenize a single word using the tokenizer's model type: BPE when a merge
   * table is present, WordPiece otherwise.
   */
  private tokenizeWord(word: string): number[] {
    if (this.usesBPE) return this.tokenizeWordBPE(word);

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
   * Apply the merge table to a word, then map the resulting pieces to IDs.
   * Unknown pieces fall back to the unknown token rather than throwing, which
   * matches ONNX Runtime's text-tokenizer behaviour.
   */
  private tokenizeWordBPE(word: string): number[] {
    const whole = this.vocab.get(
      this.bpeSpaceMarker ? this.bpeSpaceMarker + word : word
    );
    if (whole !== undefined) return [whole];

    const pieces = this.applyBPE(word);
    const ids: number[] = [];

    for (let i = 0; i < pieces.length; i++) {
      const piece = i === 0 && this.bpeSpaceMarker
        ? this.bpeSpaceMarker + pieces[i]
        : pieces[i];
      const id = this.vocab.get(piece);
      if (id !== undefined) {
        ids.push(id);
        continue;
      }
      ids.push(this.unkTokenId);
    }

    return ids;
  }

  /**
   * Apply BPE merges to a word (for BPE-style tokenizers like GPT-2/Llama).
   * Merge rules are applied in ascending rank order until no adjacent pair is
   * present in the merge table; caching is keyed on the original word.
   */
  private applyBPE(word: string): string[] {
    const cached = this.bpeCache.get(word);
    if (cached) return cached;

    let tokens = word.split('');
    if (tokens.length <= 1 || this.mergeRanks.size === 0) {
      this.bpeCache.set(word, tokens);
      return tokens;
    }

    while (tokens.length > 1) {
      let bestRank = Number.POSITIVE_INFINITY;
      for (let i = 0; i < tokens.length - 1; i++) {
        const rank = this.mergeRanks.get(`${tokens[i]} ${tokens[i + 1]}`);
        if (rank !== undefined && rank < bestRank) bestRank = rank;
      }
      if (bestRank === Number.POSITIVE_INFINITY) break;

      const merged: string[] = [];
      for (let i = 0; i < tokens.length; i++) {
        if (
          i < tokens.length - 1 &&
          this.mergeRanks.get(`${tokens[i]} ${tokens[i + 1]}`) === bestRank
        ) {
          merged.push(tokens[i] + tokens[i + 1]);
          i++;
        } else {
          merged.push(tokens[i]);
        }
      }

      if (merged.length === tokens.length) break;
      tokens = merged;
    }

    this.bpeCache.set(word, tokens);
    return tokens;
  }

  private static toInt64(values: Int32Array): BigInt64Array {
    const out = new BigInt64Array(values.length);
    for (let i = 0; i < values.length; i++) out[i] = BigInt(values[i]);
    return out;
  }
}
