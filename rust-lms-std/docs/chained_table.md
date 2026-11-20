# `ChainedTable`: a hash table that generated code fills

`ChainedTable` (`src/table.rs`) is the hash table that stage 7's group-by uses. Lookup,
key comparison and insert all run as generated code inside the kernel. The host only
allocates memory, which happens once every 256 inserts. This document explains how a
chained hash table works, how this one is built, the choices behind it, and why an entry
the kernel holds stays valid however much the table grows.

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
Section 4 explains why that rules it out here.

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
| **Entries** | Records of an `EntryLayout`: the table's own `next` and `hash`, then the caller's fields (keys and aggregate states). | The kernel. The host rewrites only `next`, when relinking. |
| **Chunks** | Blocks of 256 entries each, zeroed when allocated, **never moved or freed until the table is dropped**. Entry `k` is slot `k % 256` of chunk `k / 256`. | The host allocates them. |
| **Directory** | `2^k` chain heads, each a pointer to an entry or null, with `k ≥ 8`. | The host builds it. The kernel writes a head on insert. |
| **`TableHead`** | `dir`, `mask` (`size − 1`), `len` and `chunks` (the array of chunk addresses). It is boxed so its address never changes. | The host rewrites `dir`, `mask` and `chunks`. The kernel advances `len`. |

A kernel receives a `TableRef`: the head's address, plus the `ChainedTable` itself for
the extern. `StagedTable` reads the head through typed fields (`TableHeadType::dir()` and
so on), so the kernel never computes an offset by hand.

### 2.1 Find or insert (generated)

The caller provides three things: the hash, a `same_key` closure that compares an entry's
key to its own, and an `init` closure that fills a new entry. Translated to Rust, the
generated code is:

```rust
unsafe fn find_or_insert(head: *mut TableHead, owner: *mut ChainedTable, hash: u64) -> *mut Entry {
    // find
    let mut at = *(*head).dir.add((hash & (*head).mask) as usize);
    while !at.is_null() {
        if (*at).hash == hash && same_key(at) {     // cheap u64 compare before the key
            return at;
        }
        at = (*at).next;
    }
    // insert
    let len = (*head).len;
    if len & 255 == 0 {                            // chunk boundary: once per 256 inserts
        chained_table_make_room(owner);            // cold extern; may rewrite dir, mask, chunks
    }
    let chunk = *(*head).chunks.add((len >> 8) as usize);   // read after the call
    let e = chunk.add((len & 255) as usize * STRIDE) as *mut Entry;
    (*e).hash = hash;
    init(e);
    let slot = (*head).dir.add((hash & (*head).mask) as usize); // read after the call
    (*e).next = *slot;                             // link in at the front of the chain
    *slot = e;
    (*head).len = len + 1;
    e
}
```

`for_each` walks `k` from `0` to `len` and visits entry `k` at chunk `k >> 8`, slot
`k & 255`. Entries come out in insertion order, without walking the directory.

### 2.2 Make room (host, once per 256 inserts)

```rust
fn make_room(&mut self) {
    let len = self.len();
    if len == self.dir.len() {            // one entry per chain head: load factor 1
        self.relink(2 * len);             // new directory, every entry relinked by its stored hash
    }
    if len / 256 == self.chunks.len() {
        self.chunks.push(alloc_zeroed(chunk_layout));
        self.head.chunks = self.chunks.as_ptr();
    }
}
```

`relink` allocates a directory of the new size. For each entry it reads the stored
`hash` (never rehashing the key), picks a slot with the new mask, and links the entry in
at the front. It then points `head.dir` and `head.mask` at the new directory. The
caller's fields are never touched.

The load factor runs between ½ and 1: it is 1 just before the directory doubles and ½
just after.

### 2.3 Growth in detail

**When it runs.** Growth happens only inside an insert, before the new entry is written,
and only when `len & 255 == 0`, at `len = 0, 256, 512, …`. A find never grows the table.
Each `make_room` call does one or both of two jobs:

| `len` at the call | New chunk? | Directory doubles? |
|---|---|---|
| 0 | yes (chunk 0) | no: 0 ≠ 256 |
| 256 | yes (chunk 1) | yes: 256 → 512 |
| 512 | yes (chunk 2) | yes: 512 → 1024 |
| 768 | yes (chunk 3) | no: 768 ≠ 1024 |
| 1024 | yes (chunk 4) | yes: 1024 → 2048 |
| 1280, 1536, 1792 | yes | no |
| 2048 | yes | yes: 2048 → 4096 |

A new chunk is added on every call. The directory doubles when `len` reaches its size,
which is at every power of two from 256 on.

**Why checking only at chunk boundaries is enough.** The directory has to double when
`len == dir.len()`. Its size is always a power of two of at least 256, and therefore a
multiple of 256. So `len` can only equal it when `len` is a multiple of 256, which is
exactly when the kernel calls `make_room`. One branch per insert covers both jobs, and
there is one extern instead of two.

**What `make_room` changes, step by step:**

1. *Directory full* (`len == dir.len()`): `relink(2 · len)`.
   1. Allocate a new directory of `2 · len` null heads, and set `mask = 2 · len − 1`.
   2. For `k` in `0..len`, in insertion order: read entry `k`'s stored `hash`, pick
      `slot = hash & mask`, and write `entry.next = dir[slot]` and `dir[slot] = entry`.
   3. Replace `self.dir` (which frees the old directory) and set `head.dir` and
      `head.mask`.
2. *Last chunk full* (`len / 256 == chunks.len()`): allocate a zeroed chunk, push its
   address onto `self.chunks`, and set `head.chunks`, since the push may have moved that
   array.

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
pays it once per power of two, and the kernel pays one predictable branch per insert.

### 2.4 Worked example: three doublings

To keep the drawing small, this example scales the table down to **chunks of 4 entries
and a starting directory of 4 heads**. The real values are 256 and 256. The logic is the
same: the kernel calls `make_room` when `len & 3 == 0`, and the directory doubles when
`len` equals its size. Each hash below is a small number, and the slot is `hash & mask`.

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

Then chunk 1 is added. After the call, the kernel reloads `dir`, `mask` and `chunks` from
the head, writes E into chunk 1, slot 0, and links it at `12 & 7 = 4`. F, G and H follow.

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
kernel survives both because it reads `dir` and `chunks` from the head after every
`make_room`.

The real table behaves the same way with 256 in place of 4. Its first doubling comes at
the 257th insert (256 → 512), the second at the 513th, and the third at the 1,025th
(1024 → 2048). That is exactly the path `an_entry_held_across_growth_stays_put` takes
with 2,000 keys.

## 3. The choices

| Choice | Why | Given up |
|---|---|---|
| **Generated lookup, equality and insert; the host only allocates.** | A key is a mix of integers, floats and strings that changes with each query. Generated code compares it field by field in registers. A host table would need either an extern call per row (sql-gen's design) or keys serialised into bytes. Umbra generates the same code. | The table code is staged, so it is harder to step through in a debugger than a plain Rust function. |
| **Chaining, not open addressing.** | Growth relinks pointers and never moves an entry, so the kernel can keep an entry's address (§5). Group-by needs exactly this: find a group, then update its states. M4 goes further and keeps it across a whole key binding. | Each probe follows a pointer, and each entry carries 16 bytes of `next` and `hash`. |
| **Entries in fixed chunks of 256, not one growable array.** | A `Vec` that grows by reallocating moves its entries. Chunks never move, and entry `k`'s address is still computed with a shift and a mask. | A level of indirection (`chunks[k >> 8]`), and up to 255 unused entries in the last chunk. |
| **The directory separate from the entries.** | Only the directory is replaced on growth, and it is just pointers: relinking costs 8 bytes per head plus a write to each entry's `next`. | Two memory areas instead of one. |
| **Power-of-two sizes, at least 256.** | `hash & mask` replaces `hash % size`. Every size the directory reaches (256, 512, …) is a multiple of 256, so "directory full" can only become true at a chunk boundary. One check per insert (`len & 255 == 0`) and **one** extern cover both kinds of growth. | The low bits of the hash pick the slot, so the hash must mix well into those bits (the next row). |
| **A folded-multiply hash** (`hash.rs`, as in wyhash and foldhash). | `(a·b) ^ ((a·b) >> 64)`: a 64×64→128-bit multiply whose two halves are xor-ed together. Every input bit affects the low bits that pick the slot. It needs only `mul` and `mul_high`, which both backends have. | It is not resistant to keys chosen to collide (HashDoS), which doesn't matter for an analytic query engine. |
| **The full hash stored in each entry, compared before the key.** | Comparing two `u64`s rejects almost every entry of the wrong key without reading it, which matters for string keys that live in a pool. The stored hash also lets `relink` work without knowing how to hash the key. | 8 bytes per entry. |
| **Insert at the chain's head.** | O(1): no walk to the tail. The find just walked the chain and failed, so the key is known to be absent. | Chain order is newest first (order is irrelevant here; `for_each` uses insertion order). |
| **Insertion-order iteration over `0..len`.** | Sequential through the chunks, with no walk over empty directory slots. Results come out in first-seen order, which makes them deterministic and easy to test. | — |
| **The table holds no pool.** | String keys belong to the caller: `init` copies the bytes into a `BytesPool` (`copy_in`), and `same_key` reads them back (`StagedPool::value`). The table only stores fields and stays generic. | The caller has to wire up the pool. |
| **Zeroed chunks.** | A new entry's fields start at zero, so `init` only sets what it reads back. | The cost of zeroing 256 entries per chunk, paid once. |
| **No deletion, no shrinking, one kernel call at a time.** | Group-by never removes a group. Each worker thread builds its own table (a partial), and the partials are merged afterwards. | Not a general-purpose map. |

## 4. Why not hashbrown, or a host table behind an extern

The stage 7 design (`relational/jit_stage7.md` §3.5) has the full comparison. In short:

- An extern call per row, into a host map, costs a call and blocks inlining.
- The key has to be packed into bytes for the host to hash and compare.
- An open-addressing table moves entries on growth, so the kernel would have to look up
  its entry again after every insert that might trigger growth.

## 5. Why an entry held across growth stays put

The test `an_entry_held_across_growth_stays_put` keeps the address of entry `0`, inserts
1,999 more entries (seven new chunks and three directory doublings, 256 → 2048), writes
through the old address, and then finds key `0` again. The count it reads back is the one
written through the old address. That works because of three facts.

**1. Entries are never moved.** An entry lives in a chunk, and a chunk is allocated once
and freed only when the table is dropped. Growth adds a chunk and never reallocates an
existing one. `self.chunks` is a `Vec<*mut u8>`, and pushing to it can reallocate, but
what moves is that vector's array of chunk addresses, never the chunks it points to.

**2. Relinking rewrites links, not entries.** Doubling the directory allocates a new
array of heads and rewrites each entry's `next` in place, where the entry already is.
The entry's address, its `hash` and the caller's fields do not change. Contrast this with
open addressing, where growth copies every entry to a new address.

**3. The kernel never caches what does move.** Three things can change during the extern
call: `dir`, `mask` and `chunks`. The generated code reads each of them from `TableHead`
every time it needs them (`slot()` and `chunks()` emit fresh loads), and in `insert` it
reads them after the `make_room` call, as the pseudo-code above shows. The head itself is
a `Box`, so its address, which the kernel holds, stays valid. What the kernel keeps in a
variable is the entry's own address (`Var<SMutPtr<u8>>`), and fact 1 keeps that address
valid.

The same rule appears in two other places. Block columns survive `db4_block_grow`
because the kernel reloads the column pointers from the block header after the call
(`relational/jit_stage6_codegen.md`, "Growth: why nothing is reassigned after `db4_block_grow`"). Stored pool strings stay readable because
`StagedPool::value` reads the pool's buffer table from `PoolHead` each time. In every
case the host may move whatever the kernel finds through a header, and must never move
anything the kernel holds directly.

The extern itself relies on the same separation. It receives `&mut ChainedTable`, and
nothing it can reach through that reference is a Rust reference into the entries: it
works through the raw chunk pointers. The kernel's entry pointers therefore never alias
a live `&mut` borrow.

## 6. Tests (`tests/table.rs`), all on Cranelift and LLVM

| Test | What it checks |
|---|---|
| `counts_every_key_across_chunks_and_doublings` | 700 keys over 5,000 rows: three chunks, two doublings, counts equal to Rust's `HashMap`. |
| `equal_hashes_are_told_apart_by_the_key` | Every key has the same hash, so one chain holds them all and `same_key` alone decides. |
| `high_hash_bits_split_chains_when_the_directory_doubles` | Hashes differing only above the mask share a head until doubling separates them; relinking keeps every entry findable. |
| `an_entry_held_across_growth_stays_put` | §5. |
| `string_keys_live_in_a_pool` | Keys copied into a `BytesPool`, compared as bytes (long keys of equal length and prefix), and read back. |
| `hash_word_is_the_folded_multiply` | The staged hash against the same arithmetic on the host. |

Mutation testing caught all four injected table bugs: a relink that drops entries, the
hash alone deciding a match, a chain walk that stops at its head, and a wrong slot within
a chunk. The last one makes chains cyclic, so it shows up as a hang.

**Known gap:** a directory that never doubles is not caught. Lookups stay correct and
the chains just get longer. Catching it would need a host accessor (the directory's size)
that only tests call, and the project rule is not to add one.
