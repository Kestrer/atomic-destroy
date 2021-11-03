#![no_std]
//! Atomically destroyable types.
//!
//! # Examples
//! ```
//! # use atomic_destroy::AtomicDestroy;
//! let value = AtomicDestroy::new(Box::new(5));
//! assert_eq!(**value.get().unwrap(), 5);
//! value.destroy();
//! // The Box's destructor is run here.
//! assert!(value.get().is_none());
//! ```
#![warn(clippy::pedantic, clippy::cargo)]

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::ops::Deref;
use core::ptr::drop_in_place;
use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::sync::Arc;

/// An atomically destroyable value.
#[derive(Debug)]
pub struct AtomicDestroy<T> {
    /// The number of people current using the value. When this is 0 and `drop_state` is 1,
    /// drop the value.
    held_count: AtomicUsize,
    /// Whether the value should be dropped at the next opportunity. 0 means don't drop the value,
    /// 1 means drop the value when possible and 2 means the value is already dropped to avoid
    /// double free.
    drop_state: AtomicU8,
    /// The value itself.
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T> AtomicDestroy<T> {
    /// Create a new atomically destroyable value.
    #[must_use]
    pub const fn new(value: T) -> Self {
        Self {
            held_count: AtomicUsize::new(0),
            drop_state: AtomicU8::new(0),
            value: UnsafeCell::new(MaybeUninit::new(value)),
        }
    }

    /// Create an atomically destroyable value that has already been dropped.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            held_count: AtomicUsize::new(0),
            drop_state: AtomicU8::new(2),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    /// Create an atomically destroyable value from an `Option<T>`.
    #[must_use]
    pub fn maybe_new(value: Option<T>) -> Self {
        match value {
            Some(v) => Self::new(v),
            None => Self::empty(),
        }
    }

    /// Get a shared reference to the value if it hasn't been destroyed.
    #[must_use]
    pub fn get(&self) -> Option<Value<'_, T>> {
        // Prematurely make sure that the value won't be dropped.
        self.held_count.fetch_add(1, Ordering::SeqCst);

        // Created here so that the destructor is always run.
        let value = Value(self);

        if value.0.drop_state.load(Ordering::SeqCst) > 0 {
            // The value is dropped or is attempting to drop. Don't interfere.
            None
        } else {
            Some(value)
        }
    }

    /// Get an `Arc` reference to the value if it hasn't been destroyed.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn get_arc(self: Arc<Self>) -> Option<ArcValue<T>> {
        core::mem::forget(self.get()?);
        Some(ArcValue(self))
    }

    /// Run a function using the value.
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        self.get().map(|v| f(&*v))
    }

    /// Destroy the value. If someone is currently using the value the destructor will be run when
    /// they are done.
    pub fn destroy(&self) {
        if self
            .drop_state
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
            && self.held_count.load(Ordering::SeqCst) == 0
            && self.drop_state.swap(2, Ordering::SeqCst) == 1
        {
            // SAFETY: This code is only run if `drop_state` was zero. As this code sets it to one
            // and nothing else can set it back, this block can only be run once.
            //
            // If we also know that `held_count` is zero then no code can currently be reading the
            // value. Moreover, no code in the future can read from the value as `drop_state` is
            // permanently nonzero.
            unsafe {
                self.drop_value();
            }
        }
    }

    /// Drop the value, not checking if anyone else is using it.
    ///
    /// # Safety
    ///
    /// This function must only be called once, and `value` must not be accessed from this point
    /// onwards.
    unsafe fn drop_value(&self) {
        drop_in_place((*self.value.get()).as_mut_ptr());
    }
}

// These can probably be relaxed but I want to play it safe
unsafe impl<T: Send + Sync> Send for AtomicDestroy<T> {}
unsafe impl<T: Send + Sync> Sync for AtomicDestroy<T> {}

impl<T> Drop for AtomicDestroy<T> {
    fn drop(&mut self) {
        if self.drop_state.load(Ordering::SeqCst) < 2 {
            // SAFETY: We have unique access and the value is about to be destroyed.
            unsafe { self.drop_value() };
        }
    }
}

impl<T: Clone> Clone for AtomicDestroy<T> {
    fn clone(&self) -> Self {
        Self::maybe_new(self.get().as_deref().cloned())
    }
}

/// A "locked" shared reference to the value of an `AtomicDestroy`. While one of these exists the
/// value inside the `AtomicDestroy` is guaranteed not to be dropped.
#[derive(Debug)]
pub struct Value<'a, T>(&'a AtomicDestroy<T>);

impl<T> Deref for Value<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: Held count is guaranteed to be >0 here, and so the value cannot be dropped.
        unsafe { &*(*self.0.value.get()).as_ptr() }
    }
}

impl<T> Drop for Value<'_, T> {
    fn drop(&mut self) {
        if self.0.held_count.fetch_sub(1, Ordering::SeqCst) == 1
            && self
                .0
                .drop_state
                .compare_exchange(1, 2, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            // SAFETY: This can only happen when the value has not been dropped yet, as `drop_state`
            // is still 1.
            //
            // We also know that there are no other readers as `held_count` is zero.
            unsafe {
                self.0.drop_value();
            }
        }
    }
}

/// A "locked" `Arc` reference to the value of an `AtomicDestroy`. While one of these exists the
/// value inside the `AtomicDestroy` is guaranteed not to be dropped.
#[derive(Debug)]
#[cfg(feature = "alloc")]
pub struct ArcValue<T>(Arc<AtomicDestroy<T>>);

#[cfg(feature = "alloc")]
impl<T> Deref for ArcValue<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        // SAFETY: See `Value::deref`
        unsafe { &*(*self.0.value.get()).as_ptr() }
    }
}

#[cfg(feature = "alloc")]
impl<T> Drop for ArcValue<T> {
    fn drop(&mut self) {
        drop(Value(&*self.0));
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::prelude::rust_2021::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    use crate::AtomicDestroy;

    // Boxes are used here to better catch double frees and memory leaks.

    #[test]
    fn test_simple() {
        let value = Arc::new(AtomicDestroy::new(Box::new(5)));
        assert_eq!(**value.get().unwrap(), 5);
        assert_eq!(**value.get().unwrap(), 5);
        #[cfg(feature = "alloc")]
        assert_eq!(**value.clone().get_arc().unwrap(), 5);
        value.destroy();
        assert!(value.get().is_none());
        #[cfg(feature = "alloc")]
        assert!(value.get_arc().is_none());
    }

    #[test]
    fn test_keep_alive() {
        let value = Arc::new(AtomicDestroy::new(Box::new(5)));
        let contents_1 = value.get().unwrap();
        #[cfg(feature = "alloc")]
        let contents_2 = value.clone().get_arc().unwrap();
        #[cfg(not(feature = "alloc"))]
        let contents_2 = value.get().unwrap();
        assert_eq!(**contents_1, 5);
        assert_eq!(**contents_2, 5);

        value.destroy();
        assert_eq!(**contents_1, 5);
        assert_eq!(**contents_2, 5);
        assert!(value.get().is_none());

        drop(contents_1);
        assert_eq!(**contents_2, 5);
        assert!(value.get().is_none());

        drop(contents_2);
        assert!(value.get().is_none());
    }

    #[test]
    fn test_empty() {
        assert!(<AtomicDestroy<()>>::empty().get().is_none());
    }

    #[test]
    fn stress_test() {
        let limit = Instant::now() + Duration::from_secs(3);
        let value = Arc::new(AtomicDestroy::new(Box::new(5)));

        let mut threads = (0..5)
            .map(|_| {
                let value = value.clone();
                thread::spawn(move || {
                    while Instant::now() < limit {
                        match value.get() {
                            Some(v) => assert_eq!(**v, 5),
                            None => break,
                        }
                        #[cfg(feature = "alloc")]
                        match value.clone().get_arc() {
                            Some(v) => assert_eq!(**v, 5),
                            None => break,
                        }
                    }
                })
            })
            .collect::<Vec<_>>();

        thread::sleep(Duration::from_secs(1));

        threads.extend((0..5).map(|_| {
            let value = value.clone();
            thread::spawn(move || {
                for _ in 0..800 {
                    value.destroy();
                }
            })
        }));

        for thread in threads {
            thread.join().unwrap();
        }
    }
}
