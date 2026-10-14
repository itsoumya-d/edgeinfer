import {test} from 'node:test';
import assert from 'node:assert/strict';
import {Tokenizer} from '../dist/index.mjs';
const vocab={Apple:1,apple:2};
const encode=tokenizer=>Array.from(tokenizer.encode('Apple apple',{addSpecialTokens:false}).inputIds);
test('legacy default remains uncased',()=>assert.deepEqual(encode(new Tokenizer({vocab})),[2,2]));
test('explicit cased tokenizer preserves distinct vocabulary IDs',()=>assert.deepEqual(encode(new Tokenizer({vocab,lowercase:false})),[1,2]));
for(const [normalizer,expected] of [[null,[1,2]],[{type:'Lowercase'},[2,2]],[{type:'BertNormalizer',lowercase:false,strip_accents:false,clean_text:false,handle_chinese_chars:false},[1,2]]]) test(`supported normalizer ${JSON.stringify(normalizer)}`,async t=>{
 t.mock.method(globalThis,'fetch',async()=>Response.json({model:{vocab},normalizer}));
 assert.deepEqual(encode(await Tokenizer.fromUrl('https://example.invalid')) ,expected);
});
for(const normalizer of [{type:'NFC'},{type:'BertNormalizer',lowercase:false},{type:'StripAccents'},{type:'Sequence',normalizers:[]}]) test(`rejects unsupported normalization ${JSON.stringify(normalizer)}`,async t=>{
 t.mock.method(globalThis,'fetch',async()=>Response.json({model:{vocab},normalizer}));
 await assert.rejects(Tokenizer.fromUrl('https://example.invalid'),/Unsupported tokenizer normalizer/);
});
