//! Benchmarks for CakeLamp tensor operations.
//!
//! Run with: cargo bench --manifest-path crates/cakelamp-core/Cargo.toml

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use cakelamp_core::ops;
use cakelamp_core::tensor::Tensor;

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for size in [32, 64, 128, 256] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size}x{size}")),
            &size,
            |b, &size| {
                let a = Tensor::rand(vec![size, size]);
                let bt = Tensor::rand(vec![size, size]);
                b.iter(|| black_box(ops::matmul(&a, &bt)));
            },
        );
    }
    group.finish();
}

fn bench_element_wise(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_wise");
    let sizes = [1000, 10_000, 100_000];

    for &n in &sizes {
        let a = Tensor::rand(vec![n]);
        let b = Tensor::rand(vec![n]);

        group.bench_with_input(
            BenchmarkId::new("add", n),
            &n,
            |bench, _| bench.iter(|| black_box(ops::add(&a, &b))),
        );
        group.bench_with_input(
            BenchmarkId::new("mul", n),
            &n,
            |bench, _| bench.iter(|| black_box(ops::mul(&a, &b))),
        );
    }
    group.finish();
}

fn bench_unary_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("unary_ops");
    let n = 100_000;
    let a = Tensor::rand(vec![n]);

    group.bench_function("exp", |b| b.iter(|| black_box(ops::exp(&a))));
    group.bench_function("relu", |b| b.iter(|| black_box(ops::relu(&a))));
    group.bench_function("sigmoid", |b| b.iter(|| black_box(ops::sigmoid(&a))));
    group.bench_function("tanh", |b| b.iter(|| black_box(ops::tanh(&a))));
    group.finish();
}

fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions");
    let n = 100_000;
    let a = Tensor::rand(vec![n]);

    group.bench_function("sum", |b| b.iter(|| black_box(ops::sum(&a))));
    group.bench_function("mean", |b| b.iter(|| black_box(ops::mean(&a))));
    group.bench_function("max", |b| b.iter(|| black_box(ops::max(&a))));

    let a2d = Tensor::rand(vec![1000, 100]);
    group.bench_function("sum_dim0_1000x100", |b| {
        b.iter(|| black_box(ops::sum_dim(&a2d, 0, false)));
    });
    group.bench_function("sum_dim1_1000x100", |b| {
        b.iter(|| black_box(ops::sum_dim(&a2d, 1, false)));
    });
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    for &(rows, cols) in &[(64, 10), (256, 10), (64, 128)] {
        let a = Tensor::rand(vec![rows, cols]);
        group.bench_with_input(
            BenchmarkId::new("softmax", format!("{rows}x{cols}")),
            &(rows, cols),
            |b, _| b.iter(|| black_box(ops::softmax(&a, 1))),
        );
        group.bench_with_input(
            BenchmarkId::new("log_softmax", format!("{rows}x{cols}")),
            &(rows, cols),
            |b, _| b.iter(|| black_box(ops::log_softmax(&a, 1))),
        );
    }
    group.finish();
}

fn bench_broadcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcast");
    // (N,1) + (1,M) => (N,M)
    let a = Tensor::rand(vec![256, 1]);
    let b = Tensor::rand(vec![1, 256]);
    group.bench_function("256x1_plus_1x256", |bench| {
        bench.iter(|| black_box(ops::add(&a, &b)));
    });

    // Same-shape (no broadcast needed, fast path)
    let c = Tensor::rand(vec![256, 256]);
    let d = Tensor::rand(vec![256, 256]);
    group.bench_function("256x256_plus_256x256", |bench| {
        bench.iter(|| black_box(ops::add(&c, &d)));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_element_wise,
    bench_unary_ops,
    bench_reductions,
    bench_softmax,
    bench_broadcast,
);
criterion_main!(benches);
