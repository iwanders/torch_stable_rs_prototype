## Torch Stable Rust Prototype.

A very minimal rust wrapper for libtorch, using the [Torch Stable API](https://docs.pytorch.org/cppdocs/stable.html) only.

This is mostly my project to gain a better understanding of how (lib/py)torch works under the hood. I do not recommend using this.

## torch_stable
Very minimal (handwritten) bindings for the [LibTorch Stable ABI](https://docs.pytorch.org/docs/2.11/notes/libtorch_stable_abi.html).
This system works through a small set of C functions that provide a limited subset of the functionality from libtorch.

The crate is structured after the [stable](https://github.com/pytorch/pytorch/tree/main/torch/csrc/stable),
[aoti_torch](https://github.com/pytorch/pytorch/tree/main/torch/csrc/inductor/aoti_torch) and [headeronly](https://github.com/pytorch/pytorch/tree/main/torch/headeronly) directories.

There's some support tooling in the contrib submodule, but it's mostly there for testing / relic of old.

The functionality in this crate is a subset of the upstream functionality, it does not follow Rust lifetimes or safety guarantees.

## Flash Powder

> What makes light and works through oxidization? (Magnesium) Flash Powder.

This is my attempt at minimal rust bindings for LibTorch, through the stable abi only.

For my project I only need a small interface surface, so that's what I'm building here. The stable ABI doesn't expose
all functionality, so not all functionality in LibTorch can be exercised this way.

It follows the rust semantics as closely as possible. This means;

- No unsafe in the public interface, safe behaviour as you'd expect.
- No interior mutability, all methods are const correct.
- Modifying one tensor will not modify another, unless it has a mutable borrow.
- Rust style lifetimes on tensors, either tied together with an explicit lifetime, or completely separate.

There are three structures fundamental to achieving this:

- `Tensor`; Owning tensor, this owns the data. (think `Vec<u8>`)
- `Ten<'_>`; Const borrow of Tensor, this has a parent, its lifetime cannot exceed the parent. (think `&[u8]`)
- `TenMut<'_>`; Mutable borrow of Tensor, this has a mutable parent, its lifetime cannot exceed the parent. (think `&mut [u8]`)

Under the hood, each of these is a `StableTensor` and its own tensor handle on the LibTorch side.

This doesn't map perfectly to Torch's operations, for example the `.to()` method in libtorch sometimes returns a copy, but not always.
So there's some arbitrary choices here, like `.to()` in this crate  always makes a copy.

## Valgrind
There's some helper tooling in `./torch_stable/valgrind` to create suppression files against a C++ binary.
These ensure that we ignore some uninitialised values that valgrind finds in the bowels of LibTorch.

Run with these suppressions using valgrind through the runner;
```
./torch_stable/valgrind/valgrind.sh --leak-check=full target/debug/deps/torch_stable-5f3b6c1dd8420412
```
