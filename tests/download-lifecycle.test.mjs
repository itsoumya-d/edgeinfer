import { test } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { ModelCache, Tokenizer } from '../dist/index.mjs';

async function server(t, handler) {
 const timers = []; const instance = http.createServer((req, res) => handler(req, res, fn => timers.push(setTimeout(fn, 500))));
 await new Promise(resolve => instance.listen(0, '127.0.0.1', resolve));
 t.after(async () => { timers.forEach(clearTimeout); instance.closeAllConnections(); await new Promise(resolve => instance.close(resolve)); });
 return `http://127.0.0.1:${instance.address().port}`;
}
function cacheFixture(t) {
 const original = Object.getOwnPropertyDescriptor(globalThis, 'caches');
 const values = new Map(), puts = [];
 globalThis.caches = {open: async () => ({match: async url => values.get(url)?.clone(), put: async (url, response) => {puts.push(url); values.set(url, response);}})};
 t.after(() => { if(original) Object.defineProperty(globalThis, 'caches', original); else delete globalThis.caches; });
 return {values, puts};
}
for (const resource of ['model', 'tokenizer']) for (const phase of ['headers', 'body']) {
 test(`${resource} timeout covers stalled ${phase} and does not cache partial data`, async t => {
  const cache = cacheFixture(t);
  const url = await server(t, (req, res, later) => {
   if(phase === 'body') { res.writeHead(200, {'content-type':'application/json'}); res.write(resource === 'tokenizer' ? '{' : 'partial'); }
   later(() => res.end(resource === 'tokenizer' ? (phase === 'body' ? '"vocab":1}' : '{"word":1}') : 'complete'));
  });
  const result = resource === 'model' ? new ModelCache().getModel(url, false, {downloadTimeoutMs:80}) : Tokenizer.fromUrl(url, {downloadTimeoutMs:80});
  await assert.rejects(result, {name:'TimeoutError'}); assert.deepEqual(cache.puts, []);
 });
}
test('caller aborts a stalled response body without cache commit', async t => {
 const cache = cacheFixture(t), controller = new AbortController();
 const url = await server(t, (req,res) => {res.writeHead(200);res.write('partial');controller.abort();});
 await assert.rejects(new ModelCache().getModel(url, false, {signal:controller.signal,downloadTimeoutMs:1000}), {name:'AbortError'});
 assert.deepEqual(cache.puts, []);
});
test('already aborted request never calls fetch', async t => {
 const controller = new AbortController();controller.abort();
 const fetch = t.mock.method(globalThis,'fetch', () => {throw new Error('must not fetch');});
 await assert.rejects(new ModelCache().getModel('https://example.invalid',false,{signal:controller.signal}),{name:'AbortError'});
 assert.equal(fetch.mock.callCount(),0);
});
test('complete response is cached, cache hit skips network, force download replaces complete entry', async t => {
 const cache=cacheFixture(t);let requests=0;
 const url=await server(t,(req,res)=>res.end(String(++requests)));
 const client=new ModelCache();
 assert.equal(new TextDecoder().decode(await client.getModel(url)), '1');
 assert.equal(new TextDecoder().decode(await client.getModel(url)), '1');
 assert.equal(new TextDecoder().decode(await client.getModel(url,true)), '2');
 assert.equal(requests,2);assert.equal(cache.puts.length,2);
});
test('404 and truncated bodies are never cached', async t => {
 const cache=cacheFixture(t);
 const url=await server(t,(req,res)=>{if(req.url==='/missing'){res.writeHead(404);res.end('missing');}else{res.writeHead(200,{'content-length':'100'});res.end('short');}});
 await assert.rejects(new ModelCache().getModel(url+'/missing'),/HTTP 404/);
 await assert.rejects(new ModelCache().getModel(url+'/short',false,{downloadTimeoutMs:100}));
 assert.deepEqual(cache.puts,[]);
});
test('invalid deadlines fail before fetching', async t => {
 const fetch=t.mock.method(globalThis,'fetch',()=>{throw new Error('must not fetch');});
 for(const downloadTimeoutMs of [0,-1,Infinity,NaN,2147483648]) await assert.rejects(new ModelCache().getModel('unused',false,{downloadTimeoutMs}),RangeError);
 assert.equal(fetch.mock.callCount(),0);
});
test('completion removes cancellation listener and deadline timer', async t => {
 const controller=new AbortController();let added=0,removed=0;
 const add=controller.signal.addEventListener.bind(controller.signal), remove=controller.signal.removeEventListener.bind(controller.signal);
 t.mock.method(controller.signal,'addEventListener',(...args)=>{added++;return add(...args);});
 t.mock.method(controller.signal,'removeEventListener',(...args)=>{removed++;return remove(...args);});
 t.mock.method(globalThis,'fetch',async()=>new Response('complete'));
 const clear=t.mock.method(globalThis,'clearTimeout');
 await new ModelCache().getModel('https://example.invalid',true,{signal:controller.signal});
 assert.equal(added,1);assert.equal(removed,1);assert.ok(clear.mock.callCount()>0);
});

test('tokenizer load failure releases an already-created inference session', async t => {
 const {EdgeInfer, RuntimeManager}=await import('../dist/index.mjs');let releases=0;
 t.mock.method(RuntimeManager,'createSession',async()=>({session:{release:async()=>{releases++;}},provider:'wasm'}));
 t.mock.method(globalThis,'fetch',async()=>new Response('missing',{status:404}));
 await assert.rejects(EdgeInfer.fromBuffer(new ArrayBuffer(0),{tokenizerUrl:'https://example.invalid'}),/404/);
 assert.equal(releases,1);
});
test('cancellation while creating a session releases it before returning', async t => {
 const {EdgeInfer, RuntimeManager}=await import('../dist/index.mjs');let releases=0;const controller=new AbortController();
 t.mock.method(RuntimeManager,'createSession',async()=>{controller.abort();return {session:{release:async()=>{releases++;}},provider:'wasm'};});
 await assert.rejects(EdgeInfer.fromBuffer(new ArrayBuffer(0),{signal:controller.signal}),{name:'AbortError'});assert.equal(releases,1);
});
test('failed forced download preserves the prior complete cache entry', async t => {
 const cache=cacheFixture(t);const url=await server(t,(req,res)=>{res.writeHead(500);res.end('error');});
 cache.values.set(url,new Response('original'));
 await assert.rejects(new ModelCache().getModel(url,true),/HTTP 500/);
 assert.equal(await cache.values.get(url).text(),'original');assert.deepEqual(cache.puts,[]);
});
test('release failure does not hide the original tokenizer failure', async t => {
 const {EdgeInfer,RuntimeManager}=await import('../dist/index.mjs');
 t.mock.method(RuntimeManager,'createSession',async()=>({session:{release:async()=>{throw new Error('cleanup failure');}},provider:'wasm'}));
 t.mock.method(globalThis,'fetch',async()=>new Response('missing',{status:404}));
 await assert.rejects(EdgeInfer.fromBuffer(new ArrayBuffer(0),{tokenizerUrl:'https://example.invalid'}),/404/);
});

test('model and tokenizer loads work without AbortSignal.throwIfAborted', async t => {
 const cache = cacheFixture(t);
 const descriptor = Object.getOwnPropertyDescriptor(AbortSignal.prototype, 'throwIfAborted');
 Object.defineProperty(AbortSignal.prototype, 'throwIfAborted', {configurable:true,value:undefined});
 t.after(() => Object.defineProperty(AbortSignal.prototype, 'throwIfAborted', descriptor));
 t.mock.method(globalThis,'fetch',async url => new Response(url.includes('tokenizer') ? '{"hello":1}' : 'model'));
 assert.equal(new TextDecoder().decode(await new ModelCache().getModel('https://example.invalid/model')), 'model');
 assert.ok(await Tokenizer.fromUrl('https://example.invalid/tokenizer'));
 const controller = new AbortController();controller.abort();
 await assert.rejects(new ModelCache().getModel('https://example.invalid/aborted',false,{signal:controller.signal}),{name:'AbortError'});
 assert.equal(cache.puts.length,1);
});
for (const [status, headers] of [[206, {}], [200, {'content-range':'bytes 0-3/100'}]]) {
 test(`partial response ${status} cannot overwrite a complete cached model`, async t => {
  const cache=cacheFixture(t), url='https://example.invalid/model';
  cache.values.set(url,new Response('complete'));
  t.mock.method(globalThis,'fetch',async () => new Response('part',{status,headers}));
  await assert.rejects(new ModelCache().getModel(url,true),/partial/i);
  assert.deepEqual(cache.puts,[]);
  assert.equal(await cache.values.get(url).text(),'complete');
 });
}
