# `ChainedTable`: a chained hash table split between generated code and Rust

`ChainedTable` (`src/table.rs`) is the hash table that stage 7's group-by uses. The work
is split as in Umbra:

- **Generated code, on every row:** hashing the key, walking the chain, comparing keys,
  and updating the entry.
- **Rust, once per new key:** inserting an entry and growing the table.

The directory stays ⅛ to ¼ full, so a lookup rarely walks past another entry.

This document covers:
- how a chained hash table works, and how this one is built;
- the choices behind it, and what it was measured against;
- why an entry the kernel holds stays valid however much the table grows;
- the assumptions and caveats.

## 1. Chained hash tables in general

A hash table maps a key to an entry in O(1) expected time by turning the key into a
number (its **hash**) and using that number to pick a place to look.

In a **chained** table that place is a **chain head**: a slot in an array called the
**directory**. Each head points to a linked list (a **chain**) of every entry whose hash
picked that slot.

```text
directory            chains
┌─────┐
│  ●──┼──▶ [k=7] ──▶ [k=19] ──▶ ∅      hash(7) and hash(19) both pick slot 0
│  ∅  │                                 nothing picked slot 1
│  ●──┼──▶ [k=4] ──▶ ∅
│  ∅  │
└─────┘
```

- **Find:** hash the key, pick slot `hash % size`, and walk its chain, comparing each
  entry's key to the one sought.
- **Insert:** make a new entry and link it in at the front of its slot's chain. That is
  O(1): one pointer copied into the entry, one pointer written into the slot.
- **Grow:** a chain's length is about `entries / size` (the **load factor**). To keep
  chains short, the directory grows once the load factor gets too high, and every entry
  is relinked into the bigger directory, since `hash % size` changes when `size` does.

The alternative is **open addressing** (Rust's hashbrown, Abseil's Swiss tables). There
the entries sit in the array itself, and a collision moves on to the next free slot.
Open addressing is more cache-friendly, but growing it moves every entry to a new array.
Section 4 explains why that rules it out here, and what it measured at.

## 2. This table

```text
 TableHead                directory: 2^k chain heads    entries: chunks of 256, never moved
┌───────────┐            ┌─────┐
│ dir    ───┼───────────▶│  ●──┼──▶ [next ● | hash | key… | state…] ──▶ [next ∅ | …]
│ mask      │            │  ∅  │
│ len       │            │  ●──┼──▶ [next ∅ | hash | key… | state…]
│ chunks ───┼──┐         └─────┘
└───────────┘  └──▶ [chunk 0][chunk 1]…   entry k: chunk k / 256, slot k % 256
```

The table has four parts.

| Part | What it is | Who writes it |
|---|---|---|
| **Entries** | Records of an `EntryLayout`: the table's own `next` and `hash`, then the caller's fields (keys and aggregate states). An entry's address is a `SMutPtr<Record>` (`*mut Record` on the host). | The host writes `hash` and `next` (on insert and relink). The kernel writes the caller's fields. |
| **Chunks** | Blocks of 256 entries each, zeroed when allocated, **never moved or freed until the table is dropped**. Entry `k` is slot `k % 256` of chunk `k / 256`. | The host allocates them. |
| **Directory** | `2^k` chain heads, each an entry's address or null, with at least 4 heads per entry (`SPARSE`), so it is ⅛ to ¼ full. It starts at 1024. | The host only. |
| **`TableHead`** | `dir`, `mask` (`size − 1`), `len` and `chunks` (the array of chunk addresses). It is boxed so its address never changes. | The host only. The kernel only reads it. |

A kernel receives a `TableRef`: the head's address, plus the `ChainedTable` itself for
the insert extern. `StagedTable` reads the head through typed fields
(`TableHeadType::dir()` and so on), so the kernel never computes an offset by hand.

### 2.1 Find or insert (generated)

The caller provides three things: the hash, a `same_key` closure that compares an
entry's key to its own, and an `init` closure that fills a new entry. Translated to Rust,
the generated code is:

```rust
unsafe fn find_or_insert(head: *mut TableHead, owner: *mut ChainedTable, hash: u64) -> *mut Record {
    // find: on every row
    let mut at = *(*head).dir.add((hash & (*head).mask) as usize);
    while !at.is_null() {
        if (*at).hash == hash && same_key(at) {     // cheap u64 compare before the key
            return at;
        }
        at = (*at).next;
    }
    // insert: once per new key
    let e = chained_table_insert(owner, hash);     // host: room, hash, link, len + 1
    init(e);                                       // the caller's fields
    e
}
```

`for_each` walks `k` from `0` to `len` and visits entry `k` at chunk `k >> 8`, slot
`k & 255`. Entries come out in insertion order, without walking the directory.

### 2.2 Insert and make room (host)

```rust
fn insert(&mut self, hash: u64) -> *mut Record {
    let len = self.len();
    if len % 256 == 0 {
        self.make_room();                  // a chunk boundary: once per 256 inserts
    }
    let entry = self.entry(len);           // slot len % 256 of chunk len / 256
    let slot = &mut self.dir[(hash & self.head.mask) as usize];
    entry.hash = hash;
    entry.next = *slot;                    // link in at the front of the chain
    *slot = entry;
    self.head.len += 1;
    entry
}

fn make_room(&mut self) {
    let len = self.len();
    if SPARSE * len == self.dir.len() {    // the directory is ¼ full
        self.relink(2 * SPARSE * len);     // new directory, every entry relinked by its stored hash
    }
    self.chunks.push(alloc_zeroed(chunk_layout));
    self.head.chunks = self.chunks.as_ptr();
}
```

`relink` allocates a directory of the new size. For each entry it reads the stored
`hash` (never rehashing the key), picks a slot with the new mask, and links the entry in
at the front. It then points `head.dir` and `head.mask` at the new directory. The
caller's fields are never touched.

The load factor (entries per head) runs between ⅛ and ¼: it is ¼ just before the
directory doubles and ⅛ just after.

### 2.3 Growth in detail

**When it runs.** Growth happens only inside `insert`, before the new entry is written,
and only when `len % 256 == 0`, at `len = 0, 256, 512, …`. A find never grows the table.
Each `make_room` call does one or both of two jobs:

| `len` at the call | New chunk? | Directory doubles? |
|---|---|---|
| 0 | yes (chunk 0) | no: 4 · 0 ≠ 1024 |
| 256 | yes (chunk 1) | yes: 1024 → 2048 |
| 512 | yes (chunk 2) | yes: 2048 → 4096 |
| 768 | yes (chunk 3) | no: 4 · 768 ≠ 4096 |
| 1024 | yes (chunk 4) | yes: 4096 → 8192 |
| 1280, 1536, 1792 | yes | no |
| 2048 | yes | yes: 8192 → 16384 |

A new chunk is added on every call. The directory doubles when `4 · len` reaches its
size, which is at every power of two of `len` from 256 on.

**Why checking only at chunk boundaries is enough.** The directory has to double when
`4 · len == dir.len()`. Its size is always `4 · 256 · 2^k`, so `len` can only meet it
at `256 · 2^k`, a multiple of 256. That is exactly when `insert` calls `make_room`, so
one check covers both jobs.

**What `make_room` changes, step by step:**

1. *Directory ¼ full* (`4 · len == dir.len()`): `relink(2 · dir.len())`.
   1. Allocate a new directory of twice as many null heads, and set `mask` to its size
      minus one.
   2. For `k` in `0..len`, in insertion order: read entry `k`'s stored `hash`, pick
      `slot = hash & mask`, and write `entry.next = dir[slot]` and `dir[slot] = entry`.
   3. Replace `self.dir` (which frees the old directory) and set `head.dir` and
      `head.mask`.
2. *Last chunk full*: allocate a zeroed chunk, push its address onto `self.chunks`, and
   set `head.chunks`, since the push may have moved that array.

**What it does not change:** where an entry is, its `hash`, and the caller's fields.

**What it does change:** each entry's `next`, the directory (a new allocation, with the
old one freed), `mask`, and possibly the address of the chunk array. The old directory
is freed, so a `dir` pointer the kernel cached before the call would be left dangling.
That is why the kernel reloads it (§5).

**Splitting.** Doubling adds one bit to the mask. An entry at slot `s` moves either to
`s` or to `s + old_size`, depending on that bit of its hash, so each chain splits in two.
Relinking walks entries in insertion order and links each one in at the front, so every
chain stays newest first, the same order inserts produce.

**Cost.** Each doubling relinks every entry once. Over `n` inserts that is at most
`256 + 512 + … + n < 2n` relinks, so growth costs O(1) per insert on average. The host
pays it once per power of two. The kernel pays nothing per row for growth: its only
growth-related cost is the call per new key, inside which the host checks the boundary.

### 2.4 Worked example: three doublings

To keep the drawing small, this example scales the table down to **chunks of 4 entries
and a starting directory of 4 heads**, and drops `SPARSE` to 1, so the directory
doubles when `len` equals its size. The real table has chunks of 256 and keeps 4 heads
per entry, so its directory starts at 1024 and doubles when `4 · len` reaches its size;
the mechanism is the same. Each hash below is a small number, and the slot is
`hash & mask`. Every insert is `chained_table_insert`, which calls `make_room` when
`len % 4 == 0`.

The kernel inserts 17 keys, `A` to `Q`. The arrows show chains, newest first.

| key | A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hash | 0 | 4 | 1 | 8 | 12 | 2 | 16 | 5 | 3 | 20 | 24 | 9 | 6 | 17 | 28 | 13 | 32 |

**Start.** `ChainedTable::new` builds a directory of 4 null heads (`mask = 3`, `len = 0`)
and no chunks.

**Insert A to D** (`len` 0 → 4). Inserting A, at `len = 0`, calls `make_room`, which adds
chunk 0. No doubling, since 0 ≠ 4. B (4 & 3 = 0) and D (8 & 3 = 0) share slot 0 with A.

```text
mask = 3, len = 4                       chunks
dir[0] ──▶ D ──▶ B ──▶ A ──▶ ∅          chunk 0: [A B C D]
dir[1] ──▶ C ──▶ ∅
dir[2]     ∅
dir[3]     ∅
```

**Insert E: the first doubling, 4 → 8.** At `len = 4`, `len & 3 == 0`, so the kernel calls
`make_room`. Now `len == dir.len()` (4), so `relink(8)` runs with `mask = 7` and visits
A, B, C, D in that order:

| entry | hash | hash & 7 | result |
|---|---|---|---|
| A | 0 | 0 | `dir[0] = A`, `A.next = ∅` |
| B | 4 | **4** | `dir[4] = B`, `B.next = ∅` (B leaves A's chain: bit 2 of its hash is set) |
| C | 1 | 1 | `dir[1] = C` |
| D | 8 | 0 | `D.next = A`, `dir[0] = D` |

Then chunk 1 is added, and the host writes E into chunk 1, slot 0, and links it at
`12 & 7 = 4`. F, G and H follow.

```text
mask = 7, len = 8                       chunks
dir[0] ──▶ G ──▶ D ──▶ A                chunk 0: [A B C D]
dir[1] ──▶ C                            chunk 1: [E F G H]
dir[2] ──▶ F
dir[3]     ∅
dir[4] ──▶ E ──▶ B
dir[5] ──▶ H
dir[6]     ∅
dir[7]     ∅
```

**Insert I: the second doubling, 8 → 16.** At `len = 8`, `relink(16)` runs with
`mask = 15`. The chain `G → D → A` splits in two: A and G (`16 & 15 = 0`) stay at
slot 0, giving `dir[0] = G → A`, and D moves to `dir[8]` because bit 3 of its hash is
set. `E → B` splits into
`dir[4] = B` and `dir[12] = E`. Chunk 2 is added, and I, J, K, L are inserted.

**Insert M: a new chunk only.** At `len = 12`, `make_room` adds chunk 3. The directory
holds 16 heads and 12 ≠ 16, so it does not double. M, N, O, P are inserted.

```text
mask = 15, len = 16                     chunks
dir[0]  ──▶ G ──▶ A                     chunk 0: [A B C D]
dir[1]  ──▶ N ──▶ C                     chunk 1: [E F G H]
dir[2]  ──▶ F                           chunk 2: [I J K L]
dir[3]  ──▶ I                           chunk 3: [M N O P]
dir[4]  ──▶ J ──▶ B
dir[5]  ──▶ H
dir[6]  ──▶ M
dir[7]      ∅
dir[8]  ──▶ K ──▶ D
dir[9]  ──▶ L
dir[10], dir[11]  ∅
dir[12] ──▶ O ──▶ E
dir[13] ──▶ P
dir[14], dir[15]  ∅
```

**Insert Q: the third doubling, 16 → 32.** At `len = 16`, `relink(32)` runs with
`mask = 31`. Every pair splits: G moves to 16, N to 17, J to 20, K to 24 and O to 28.
Chunk 4 is added, and Q (`32 & 31 = 0`) joins A.

```text
mask = 31, len = 17                     chunks
dir[0]  ──▶ Q ──▶ A                     chunk 0: [A B C D]
dir[1]  ──▶ C                           chunk 1: [E F G H]
dir[2]  ──▶ F                           chunk 2: [I J K L]
dir[3]  ──▶ I                           chunk 3: [M N O P]
dir[4]  ──▶ B                           chunk 4: [Q · · ·]   (3 entries unused, zeroed)
dir[5]  ──▶ H
dir[6]  ──▶ M
dir[8]  ──▶ D
dir[9]  ──▶ L
dir[12] ──▶ E
dir[13] ──▶ P
dir[16] ──▶ G
dir[17] ──▶ N
dir[20] ──▶ J
dir[24] ──▶ K
dir[28] ──▶ O
(the other 16 heads are ∅)
```

**What happened to entry B.** Suppose the kernel holds B's address from when it was
inserted:

| moment | B's address | `B.next` | B's chain |
|---|---|---|---|
| inserted (`len` 1 → 2) | chunk 0 + 1·stride | A | `dir[0]: B → A` |
| after D | same | A | `dir[0]: D → B → A` |
| first doubling | same | ∅ | `dir[4]: B` |
| after E | same | ∅ | `dir[4]: E → B` |
| second doubling | same | ∅ | `dir[4]: B` (E moved to 12) |
| after J | same | ∅ | `dir[4]: J → B` |
| third doubling | same | ∅ | `dir[4]: B` (J moved to 20) |

B's address never changes. Its `hash` and the caller's fields are never written by
growth. Only its `next` link changes, and the kernel follows that link while walking a
chain, never through an entry it holds. The directory, by contrast, has been allocated
four times (4, 8, 16 and 32 heads), and each new one freed the last. The chunk array
moved too (`head.chunks` changed) if pushing chunk 4 made the vector reallocate. The
kernel survives both because it reads `dir` and `chunks` from the head every time it
needs them, and never keeps them across a call to `chained_table_insert`.

The real table follows the same path with 256 in place of 4 and a directory 4× larger.
Its first doubling comes at the 257th insert (1024 → 2048 heads), the second at the
513th, and the third at the 1,025th (4096 → 8192). That is the path
`an_entry_held_across_growth_stays_put` takes with 2,000 keys.

### 2.5 Compared with Umbra

| | Umbra | `ChainedTable` |
|---|---|---|
| hash | generated | generated (`hash_word`) |
| first entry of the chain | C++ `lookup` (inlined by LLVM) | generated: three loads from the head, so a call would cost more than the code |
| next entry | C++ `next` (inlined) | generated: a load of `next` |
| key comparison | generated | generated (`same_key`) |
| allocate and link a new entry | C++ `insert(table, hash, size)` | Rust `chained_table_insert(table, hash)` |
| write keys and aggregates | generated | generated (`init`, then the update) |
| entry layout | `[header \| keys \| payload]` | `[next \| hash \| keys \| aggregates]` |
| calls per row | 0 (LLVM inlines its C++) | 0 per row that finds its key, 1 per new group |
| entry size | given per insert (`size`) | fixed per table (`EntryLayout`) |

Umbra also calls C++ for `lookup` and `next`, but its optimising backend inlines them.
We can't inline Rust into Cranelift code, so the two operations that run on every row
stay generated. They are a handful of loads.

## 3. The choices

| Choice | Why | Given up |
|---|---|---|
| **The Umbra split: lookup and keys generated, insert and growth in Rust.** | A key mixes integers, floats and strings and changes with each query; generated code compares it field by field in registers, with no call per row. Inserting and growing never look at the key (only `hash`), so they are plain Rust: written once, tested on the host, stepped through in a debugger. | A call per new key: about +15 % at 4M groups (§7). |
| **Chaining, not open addressing.** | Growth relinks pointers and never moves an entry, so the kernel can keep an entry's address (§5). Group-by needs exactly this: find a group, then update its states. Stage 7's M5 goes further and keeps it across a whole key binding. | Each probe follows a pointer, and each entry carries 16 bytes of `next` and `hash`. |
| **Entries in fixed chunks of 256, not one growable array.** | A `Vec` that grows by reallocating moves its entries. Chunks never move, and entry `k`'s address is still computed with a shift and a mask. | A level of indirection (`chunks[k >> 8]`), and up to 255 unused entries in the last chunk. |
| **Entries hold keys and aggregates together.** | One pointer from the directory reaches everything a row needs: two cache misses per lookup with many groups, where an index into a separate record vector costs three (§4). | Every entry carries its keys. |
| **A sparse directory: ⅛ to ¼ full (`SPARSE = 4`).** | Chains are nearly always one entry long, so a lookup that finds its key rarely walks past another and the branch predictor rarely misses. This was the largest single gain measured (−18 % to −60 %, §7). | 32–64 bytes of directory per group (§9). |
| **Power-of-two sizes, multiples of 256.** | `hash & mask` replaces `hash % size`, and the directory can only fill at a chunk boundary, so one check in `insert` covers both kinds of growth. | The low bits of the hash pick the slot, so the hash must mix well into those bits (the next row). |
| **A folded-multiply hash** (`hash.rs`, as in wyhash and foldhash). | `(a·b) ^ ((a·b) >> 64)`: a 64×64→128-bit multiply whose two halves are xor-ed together. Every input bit affects the low bits that pick the slot. It needs only `mul` and `mul_high`, which both backends have. | It is not resistant to keys chosen to collide (HashDoS), which doesn't matter for an analytic query engine. |
| **The full hash stored in each entry, compared before the key.** | Comparing two `u64`s rejects almost every entry of the wrong key without reading it, which matters for string keys that live in a pool. The stored hash also lets `relink` work without knowing how to hash the key. | 8 bytes per entry. |
| **Insert at the chain's head.** | O(1): no walk to the tail. The find just walked the chain and failed, so the key is known to be absent. | Chain order is newest first (order is irrelevant here; `for_each` uses insertion order). |
| **Insertion-order iteration over `0..len`.** | Sequential through the chunks, with no walk over empty directory slots. Results come out in first-seen order, which makes them deterministic and easy to test. | — |
| **The table holds no pool.** | String keys belong to the caller: `init` copies the bytes into a `BytesPool` (`copy_in`), and `same_key` reads them back (`StagedPool::value`). The table only stores fields and stays generic. | The caller has to wire up the pool. |
| **Zeroed chunks.** | A new entry's fields start at zero, so `init` only sets what it reads back. | The cost of zeroing 256 entries per chunk, paid once. |
| **No deletion, no shrinking, one kernel call at a time.** | Group-by never removes a group. Each worker thread builds its own table (a partial), and the partials are merged afterwards. | Not a general-purpose map. |
| **Typed record addresses (`Record`).** | Entries, directory heads and chunks are `SMutPtr<Record>` / `*mut Record`, never `*mut u8`, so a record's address can't be passed where any other bytes are expected. | — |

## 4. The alternatives that were measured

**A generated insert with a directory ½ to 1 full.** This was the first version
(committed in `a7cff23`). The kernel did everything except allocate. It was 2–4× slower
than the final table with few groups, because chains of varying length made the branch
predictor miss on most lookups. Making the directory sparser fixed most of that. Moving
the insert to the host then cost about 15 % at 4M groups and nothing with few groups, in
exchange for every structural write being plain Rust.

**FFI hashbrown.** A host `hashbrown::HashTable<u32>` maps a key to an index into a
vector of fixed-size records. The kernel calls one extern per row, which returns the
record's address, and then updates the aggregates itself.

- Keys stay fixed size: a string key is its `u128` view, and only a new group copies a
  long string's bytes into the table's pool.
- It is the simplest design and does well while the groups fit in cache.
- With millions of groups each lookup that finds its key touches three cache lines one
  after another: hashbrown's tag bytes, the bucket holding the index, then the record.
  That makes it 1.7–2.1× slower than this table (§7).

**Two ideas from predictive translation** (Zinsmeister et al., SIGMOD '26, in
`dio/docs`):

- **Loading the chain heads of 16 rows before probing them,** so the loads overlap. It
  was 4–40 % slower: the M4 Max already overlaps rows out of order, and the extra pass adds
  mispredicted branches.
- **Inlining the first entry into the directory.** Not built: entries would move on
  every doubling, so a held entry would no longer stay valid.

The paper's third lesson, a table large enough that chains are short, is the sparse
directory above.

## 5. Why an entry held across growth stays put

The test `an_entry_held_across_growth_stays_put` keeps the address of entry `0`, inserts
1,999 more entries (seven new chunks and three directory doublings, 1024 → 8192 heads),
writes through the old address, and then finds key `0` again. The count it reads back is
the one written through the old address. The host test
`an_entry_does_not_move_when_the_table_grows` checks the same on the host. That works
because of three facts.

**1. Entries are never moved.** An entry lives in a chunk, and a chunk is allocated once
and freed only when the table is dropped. Growth adds a chunk and never reallocates an
existing one. `self.chunks` is a `Vec<*mut Record>`, and pushing to it can reallocate,
but what moves is that vector's array of chunk addresses, never the chunks it points to.

**2. Relinking rewrites links, not entries.** Doubling the directory allocates a new
array of heads and rewrites each entry's `next` in place, where the entry already is.
The entry's address, its `hash` and the caller's fields do not change. Contrast this with
open addressing, where growth copies every entry to a new address.

**3. The kernel never caches what does move.** Three things can change during a call to
`chained_table_insert`: `dir`, `mask` and `chunks`. The generated code reads each of them
from `TableHead` every time it needs them (`slot()` and `chunks()` emit fresh loads) and
never keeps them across the call. The head itself is a `Box`, so its address, which the
kernel holds, stays valid. What the kernel keeps in a variable is the entry's own address
(`Var<SMutPtr<Record>>`), and fact 1 keeps that address valid.

The same rule appears in two other places. Block columns survive `db4_block_grow`
because the kernel reloads the column pointers from the block header after the call
(`relational/jit_stage6_codegen.md`, "Growth: why nothing is reassigned after
`db4_block_grow`"). Stored pool strings stay readable because `StagedPool::value` reads
the pool's buffer table from `PoolHead` each time. In every case the host may move
whatever the kernel finds through a header, and must never move anything the kernel
holds directly.

The extern itself relies on the same separation. It receives `&mut ChainedTable`, and
nothing it can reach through that reference is a Rust reference into the entries: it
works through the raw chunk pointers. The kernel's entry pointers therefore never alias
a live `&mut` borrow.

## 6. Tests

**Generated code** (`tests/table.rs`), on Cranelift and LLVM:

| Test | What it checks |
|---|---|
| `counts_every_key_across_chunks_and_doublings` | 700 keys over 5,000 rows: three chunks, two doublings, counts equal to Rust's `HashMap`. |
| `equal_hashes_are_told_apart_by_the_key` | Every key has the same hash, so one chain holds them all and `same_key` alone decides. |
| `high_hash_bits_split_chains_when_the_directory_doubles` | Hashes differing only above the mask share a head until doubling separates them; relinking keeps every entry findable. |
| `an_entry_held_across_growth_stays_put` | §5. |
| `string_keys_live_in_a_pool` | Keys copied into a `BytesPool`, compared as bytes (long keys of equal length and prefix), and read back. |
| `hash_word_is_the_folded_multiply` | The staged hash against the same arithmetic on the host. |

**Host** (`src/table.rs`, unit tests of `ChainedTable::insert`):

| Test | What it checks |
|---|---|
| `insert_links_every_entry_into_its_chain_across_doublings` | 600 entries, two doublings (1024 → 4096 heads): every entry is in exactly its slot's chain, newest first, and the directory has the size `SPARSE` gives it. |
| `an_entry_does_not_move_when_the_table_grows` | Entry 0's address after 5,000 more inserts. |

**Mutation testing** caught all six injected bugs:
- a relink that drops entries;
- the hash alone deciding a match;
- a chain walk that stops at its head;
- a directory that never doubles (missed before the host tests existed);
- a wrong slot within a chunk, on the host side;
- a wrong slot within a chunk, in the generated `for_each`.

## 7. Measurements

All numbers in ms over 16M rows, `SELECT key, count(*), sum(v) GROUP BY key`, from
`benches/group_count.rs` on an M4 Max. "(d)" is this table; "committed" is the first
version (a generated insert and a directory ½ to 1 full); "FFI index" is a host
`hashbrown::HashTable<u32>` of indices into a vector of fixed-size records, the extern
returning the record and the kernel updating it.

**`u64` key**

| groups | native hashbrown | (d) LLVM | (d) Cranelift | FFI index LLVM | FFI index Cranelift | committed, LLVM |
|---|---|---|---|---|---|---|
| 1k | 22 | **29** | **33** | 53 | 52 | 71 |
| 64k | 42 | **61** | **69** | 116 | 109 | 115 |
| 1M | 140 | **186** | **217** | 296 | 284 | 234 |
| 4M | 259 | **330** | **364** | 609 | 605 | 365 |

**`(u64, string)` key** (half the strings inline, half 28 bytes; every row has its own
copy in the input's pool)

| groups | native hashbrown | (d) LLVM | (d) Cranelift | FFI index LLVM | FFI index Cranelift | committed, LLVM |
|---|---|---|---|---|---|---|
| 1k | 138 | **126** | 191 | 132 | 135 | 159 |
| 64k | 206 | **171** | 293 | 195 | 213 | 245 |
| 1M | 593 | **548** | 1102 | 724 | 756 | 622 |
| 4M | 855 | **729** | 1311 | 1128 | 1167 | 778 |

What is behind the gains:

| Change | Effect |
|---|---|
| Load factor ½–1 → ⅛–¼ | −18 % to −60 % (`u64`, LLVM). Most of the improvement. |
| Load factor → 1/16–⅛ | A further gain only when the groups fit in cache, at twice the directory memory. Not proposed. |
| Insert moved to the host | Free with few groups; about +15 % at 4M groups (330 vs 285 ms for a generated insert with the same directory). That is the price of the simpler code. |
| Loading the chain heads of 16 rows before probing them (predictive translation's "independent loads") | 4–40 % **slower**. Rejected: the M4 Max already overlaps rows out of order, and the extra pass adds mispredicted branches. |
| Inlining the first entry into the directory (predictive translation's "inline first slot") | Not built: entries would move on every doubling, so a held entry would no longer stay valid. |

### 7.1 Running the benchmarks yourself

Run these from the `dio/` directory. The LLVM backend links against the LLVM 22 in
`dio/tools/llvm`, and on macOS it also needs Homebrew's `lib`, for `-lzstd`:

```bash
cd dio
export LIBRARY_PATH=/opt/homebrew/lib
export MLIR_SYS_220_PREFIX=$PWD/tools/llvm
export LLVM_SYS_220_PREFIX=$PWD/tools/llvm
export TABLEGEN_220_PREFIX=$PWD/tools/llvm

# Everything: native, plus chained and ffi_index on Cranelift and on LLVM.
# About 10 minutes, mostly building 16M-row inputs and the string pools.
cargo bench -p rust-lms-std --bench group_count --features llvm

# Cranelift only (no LLVM needed, so none of the exports above either).
cargo bench -p rust-lms-std --bench group_count
```

The names are `group_count/<key>/<groups>/<variant>/<backend>`. For example
`group_count/u64_str/1048576/chained/llvm`, and `group_count/u64/1024/native` for the
native baseline, which has no backend. Criterion reads the argument after `--` as a
regular expression, so you can run a subset:

```bash
# Only the generated table, both keys, both backends:
cargo bench -p rust-lms-std --bench group_count --features llvm -- 'chained'

# Only the u64 key with 4M groups:
cargo bench -p rust-lms-std --bench group_count --features llvm -- 'u64/4194304/'

# Only LLVM, every variant (native is still printed, as it has no backend):
cargo bench -p rust-lms-std --bench group_count --features llvm -- 'llvm|native'
```

To compare against a change, save a baseline, make the change, then compare:

```bash
cargo bench -p rust-lms-std --bench group_count --features llvm -- --save-baseline before
# ... edit table.rs ...
cargo bench -p rust-lms-std --bench group_count --features llvm -- --baseline before
```

Criterion then prints a `change:` line for each benchmark, with its confidence interval.
HTML reports go to `dio/target/criterion/`.

**What the timings include.** Each benchmark checks its result against `native` before
timing anything. A wrong table fails the run with an assert, so it is never timed. Every
timed iteration covers:

- building a fresh table;
- aggregating all 16M rows;
- folding every group into a checksum.

Building the 16M-row inputs and the string pools happens once per group size and is not
timed. Close other heavy programs while a run is going; the numbers above are medians
of 10 samples.

**Seeing the generated code.** Cranelift's IR after optimisation is the first thing to
read when a number surprises you. Set `RUST_LMS_DEBUG_IR_OPT=1` and run the bench binary
once in test mode, which runs each benchmark once without timing it:

```bash
cargo bench -p rust-lms-std --bench group_count --no-run   # prints the binary's path
RUST_LMS_DEBUG_IR_OPT=1 target/release/deps/group_count-<hash> --bench --test 'u64/1024/' 2> ir.txt
```

For LLVM, the same variable prints the MLIR LLVM dialect before LLVM's own optimisation
passes; LLVM's execution engine does not hand back the optimised IR.

## 8. Assumptions

1. **One table per thread.** A table is used by one kernel call at a time, with no
   latches. Parallel group-by builds one table per worker, called a partial, and merges
   them afterwards (`jit_stage7.md` §6).
2. **Entries have a fixed size per table.** Keys and aggregates have fixed width, and a
   string key is its `u128` view, with long bytes in the table's pool.
3. **Entries never move.** Chunks are allocated once and freed with the table, so an entry
   the kernel holds stays valid across any number of inserts. One lookup per key binding
   (stage 7's M5) depends on this.
4. **The kernel never keeps `dir`, `mask` or `chunks` across a call to
   `chained_table_insert`.** `slot()` and `chunks()` reload them from the head each time.
5. **Nothing is deleted.** Group-by never removes a group, and the table never shrinks.
6. **The hash's low bits must be well mixed,** because `hash & mask` picks the slot. The
   folded multiply does this.
7. **A new entry is zeroed except for `hash` and `next`,** and `init` sets every field
   that is read later.

## 9. Caveats

1. **Memory.** A directory ⅛–¼ full has 4–8 heads of 8 bytes per entry, so **32–64 bytes
   per group**, on top of the entry itself (16 bytes of `next` and `hash`, plus keys and
   aggregates). At 4M groups that is 128–256 MB of directory. Native hashbrown holds the
   same 4M groups of 24 bytes in about 210 MB in total (8M buckets of 24 bytes plus a
   tag byte each). Options if this matters:
   - `SPARSE = 2` (16–32 B per group) gives back roughly half the gain;
   - two-phase pre-aggregation (caveat 6) keeps each table small.
2. **The hash of a long string.** The benchmark and `tests/table.rs` hash a string by its
   length and first 4 bytes, which its view already holds. Real data where many long
   strings share a prefix (URLs, paths) would put them all in one chain. Long string keys need `hash_bytes`
   (deferred to M2) before group-by on real data; it hashes the bytes in generated code.
3. **String keys on Cranelift.** At 1M+ groups, (d) on Cranelift is 1.5× slower than
   FFI hashbrown (1102 vs 756 ms). The same table is fast on LLVM, so the likely cause is
   Cranelift's code for `bytes_eq` and the pool lookup. Logged in
   `relational/jit_cleanup.md`.
4. **Inserting costs a call.** It is cheap with few groups and 15 % with millions. A query
   whose rows are nearly all distinct keys (a group-by on a near-unique column) pays it
   on almost every row.
5. **`init` runs on an entry that is already linked.** If `init` ran another
   `find_or_insert` on the same table, that lookup could find the half-initialised entry.
   Nothing does this today; the rule ("`init` must not use this table") is in the doc
   comment of `find_or_insert`.
6. **More groups than cache.** Every design pays a cache miss per row once the groups
   outgrow the caches. HyPer's answer, which Umbra inherits, is two-phase
   pre-aggregation: a small per-thread table that spills into partitions, then one
   in-cache table per partition. This is recalled from Leis et al. 2014 ("Morsel-driven
   parallelism"), which is not in `dio/docs`. It is stage 7's milestone after M3 (`jit_stage7.md`).
7. **What the benchmark covers.**
   - One machine (M4 Max), uniform random keys, and one shape: count and sum.
   - Skewed keys (a few very hot groups) would favour every design, the chained table
     most, since hot entries stay in cache.
   - M3's LANL benchmarks are the real check.
8. **Undefined behaviour boundary.** `chained_table_insert` takes `&mut ChainedTable`
   while the kernel holds raw pointers into the entries. The host writes only the new
   entry, the directory and, when relinking, each entry's `next`, all through raw
   pointers. Nothing it touches is borrowed by a Rust reference, but the claim rests on
   that discipline and is not checked by the compiler.
