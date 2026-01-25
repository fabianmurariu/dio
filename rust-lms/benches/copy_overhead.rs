use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

fn bench_copy_overhead(c: &mut Criterion) {
    let size = 10_000usize;
    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

    // Allocate destination buffer (simulating WASM linear memory)
    let mut dest = vec![0u8; size * 8 + 64]; // data + result space

    c.bench_function("data_copy_10k", |b| {
        b.iter(|| {
            // Copy data like WASM benchmark does
            for (i, &value) in black_box(&data).iter().enumerate() {
                let offset = i * 8;
                dest[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            }
            black_box(dest[0])
        });
    });

    c.bench_function("memcpy_10k", |b| {
        b.iter(|| {
            // Direct memcpy (fastest possible copy)
            let src_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(black_box(&data).as_ptr() as *const u8, data.len() * 8)
            };
            dest[..src_bytes.len()].copy_from_slice(src_bytes);
            black_box(dest[0])
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(5))
        .sample_size(100);
    targets = bench_copy_overhead
}

criterion_main!(benches);
