# Compression in cpp-stream

A walkthrough of what `--compress` does and why it improved FPS from ~4 to ~8
at 4096×4096.

---

## The bottleneck: the Python broadcast

Without any compression, each 4096×4096 frame is:

```
4096 × 4096 × 4 bytes (float32) = 67 108 864 bytes ≈ 64 MB
```

The C++ generator sends this over ZeroMQ to the Python bridge, which then
broadcasts it over a WebSocket to the browser. The bottleneck is not the
network card — everything runs on localhost — it is the **TCP stack and
kernel memory copying**. On a typical machine, broadcasting 64 MB over a
local WebSocket takes ~215 ms per frame, giving a ceiling of roughly 4–5 FPS
no matter how fast the C++ and GPU are.

The only way to push more frames per second is to send less data per frame.

---

## `--compress`: zlib compression of float32 data

`--compress` passes each chunk through **zlib's `compress2` function** before
sending it over ZeroMQ. zlib implements the DEFLATE algorithm: it finds
repeated byte patterns in the input and replaces them with shorter references.

### Why does it compress at all?

Float32 wave data is *spatially smooth*: neighbouring cells have very similar
values. When you write those floats as raw bytes, adjacent rows of 32-bit
values differ only in their lower bits. DEFLATE's LZ77 stage finds these
near-repetitions and encodes them as back-references, and its Huffman stage
then encodes the short residuals efficiently.

The result is roughly a 4× reduction in bytes — from 64 MB to ~16 MB per
frame — which lifts the broadcast ceiling from ~5 FPS to ~8 FPS.

### Protocol change

`--compress` switches from **protocol v1** (plain float32) to **protocol v2**,
which adds two extra fields to the chunk header:

| Offset | Field           | Meaning                                          |
|--------|-----------------|--------------------------------------------------|
| 32     | `flags`         | bit 0 = compressed                               |
| 36     | `payload_bytes` | exact byte count of compressed data that follows |

The receiver reads `payload_bytes` to know how many bytes to decompress,
then inflates them back to raw float32 before calling `setData`.

---

## The inflate concurrent-read fix

The browser uses the **Compression Streams API** (`DecompressionStream`) to
inflate the zlib payload. This API is modelled as a *transform stream* with a
writable side (compressed input) and a readable side (decompressed output).

### The deadlock

The original code wrote all data, closed the writer, and only then started
reading:

```javascript
await writer.write(data);   // write all compressed bytes
await writer.close();       // signal end of input
const reader = ds.readable.getReader();
while (true) { ... }        // now read all decompressed output
```

Transform streams have an **internal output queue** with a bounded size. When
the decompressor produces more output than the queue can hold, it signals
*backpressure* to the writable side. The writable side then pauses: the
`writer.write()` promise does not resolve until the readable side is consumed.

But in the original code the readable was not being consumed during
`writer.write()`. For large frames (64 MB of float32 that decompresses to
16 MB) the output queue could fill up, causing `writer.write()` to stall
forever — a **deadlock**. For small frames (512×512) the queue was large
enough that this didn't trigger, which is why it worked at small sizes.

### The fix

Start reading from the readable *before* writing to the writable, so the two
run concurrently:

```javascript
const reader = ds.readable.getReader();

// Start consuming output immediately — runs in parallel with the write below.
const readAll = (async () => {
  const parts = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
  }
  return parts;
})();

// Write input and close.
await writer.write(data);
await writer.close();

// Wait for the reader to drain the last bytes.
const parts = await readAll;
```

Because JavaScript is single-threaded, "concurrent" here means the two async
functions interleave at `await` boundaries. Each time `writer.write()` awaits
backpressure relief, the event loop runs the `readAll` coroutine which
consumes a chunk, freeing queue space, allowing `writer.write()` to proceed.
The deadlock cannot form.
