//! The output `Sink`: where the executor streams the response as it generates.
//!
//! One required method ([`Sink::put`], raw bytes); everything else is a default
//! JSON-token helper (`begin_obj`, `key`, `u64`, …) in the spirit of a serde
//! serializer. The helpers are deliberately *stateless* (each just emits bytes),
//! so the same calls can later be driven from rust-lms-compiled code via extern
//! functions. Comma placement is the caller's job (it knows the structure).

/// A byte sink the executor writes the GraphQL JSON response into.
pub trait Sink {
    /// Append raw bytes. The only method an implementor must provide.
    fn put(&mut self, bytes: &[u8]);

    // --- JSON structure ---

    /// `{`
    fn begin_obj(&mut self) {
        self.put(b"{");
    }
    /// `}`
    fn end_obj(&mut self) {
        self.put(b"}");
    }
    /// `[`
    fn begin_arr(&mut self) {
        self.put(b"[");
    }
    /// `]`
    fn end_arr(&mut self) {
        self.put(b"]");
    }
    /// `,`
    fn comma(&mut self) {
        self.put(b",");
    }

    /// An object key plus its colon: `"name":`. `name` is assumed to be a safe
    /// identifier (GraphQL field name/alias) and is not escaped.
    fn key(&mut self, name: &str) {
        self.put(b"\"");
        self.put(name.as_bytes());
        self.put(b"\":");
    }

    // --- JSON scalars ---

    /// A `u64` literal, formatted without allocating.
    fn u64(&mut self, mut n: u64) {
        if n == 0 {
            self.put(b"0");
            return;
        }
        let mut buf = [0u8; 20];
        let mut i = buf.len();
        while n > 0 {
            i -= 1;
            buf[i] = b'0' + (n % 10) as u8;
            n /= 10;
        }
        self.put(&buf[i..]);
    }

    /// An `i64` literal, formatted without allocating.
    fn i64(&mut self, n: i64) {
        if n < 0 {
            self.put(b"-");
            // `-(i64::MIN)` overflows; widen through u64.
            self.u64((n as i128).unsigned_abs() as u64);
        } else {
            self.u64(n as u64);
        }
    }

    /// `true` / `false`.
    fn bool(&mut self, b: bool) {
        self.put(if b { b"true" } else { b"false" });
    }

    /// `null`.
    fn null(&mut self) {
        self.put(b"null");
    }

    /// A JSON string literal (quoted and escaped).
    fn string(&mut self, s: &str) {
        self.put(b"\"");
        let bytes = s.as_bytes();
        let mut start = 0;
        for (i, &b) in bytes.iter().enumerate() {
            let esc: &[u8] = match b {
                b'"' => b"\\\"",
                b'\\' => b"\\\\",
                b'\n' => b"\\n",
                b'\r' => b"\\r",
                b'\t' => b"\\t",
                _ => continue,
            };
            if start < i {
                self.put(&bytes[start..i]);
            }
            self.put(esc);
            start = i + 1;
        }
        if start < bytes.len() {
            self.put(&bytes[start..]);
        }
        self.put(b"\"");
    }
}

/// A `Sink` that collects into an in-memory buffer (used by the synchronous
/// `execute` entry point and unit tests).
#[derive(Default)]
pub struct VecSink(pub Vec<u8>);

impl VecSink {
    pub fn new() -> Self {
        Self::default()
    }
    /// Consume the buffer as a UTF-8 string (the response is always UTF-8).
    pub fn into_string(self) -> String {
        String::from_utf8(self.0).expect("response is valid UTF-8")
    }
}

impl Sink for VecSink {
    fn put(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }
}
