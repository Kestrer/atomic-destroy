# Atomic Destroy

This crate provides a type which can hold a value and can be atomically destroyed.

It does not require the standard library.

# Examples

```rust
# use atomic_destroy::AtomicDestroy;
let value = AtomicDestroy::new(Box::new(5));
assert_eq!(**value.get().unwrap(), 5);
value.destroy();
// The Box's destructor is run here.
assert!(value.get().is_none());
```
