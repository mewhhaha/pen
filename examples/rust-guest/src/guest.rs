//! The game as a Wasm module, before `adapt` gives it the guest ABI's
//! shape. Rust cannot name `externref`, the host capability Blot's `main`
//! takes, so the module exports `pen:main` (the program) and imports
//! `pen:guest/1` `exchange`, which sends an array and returns the host's;
//! `adapt` wraps them as `main(host)` and `call_array_f32_array_f32`. The
//! array arena (`blot:allocate`, `blot:reset`) is the guest ABI's own: the
//! host writes each request into memory `blot:allocate` hands it.
use std::cell::UnsafeCell;

/// The host's last array: one at a time, read before the next arrives.
struct Arena(UnsafeCell<Vec<u32>>);

// SAFETY: a wasm32-unknown-unknown module has one thread.
unsafe impl Sync for Arena {}

static ARENA: Arena = Arena(UnsafeCell::new(Vec::new()));

#[link(wasm_import_module = "pen:guest/1")]
unsafe extern "C" {
    /// Sends the array at `packet` (a u32 length, then that many f32 lanes)
    /// and returns the host's answer, in the arena.
    fn exchange(packet: *const u32) -> *const u32;
}

/// Space for `bytes` of the host's array, 4-aligned.
#[unsafe(export_name = "blot:allocate")]
extern "C" fn allocate(bytes: u32) -> *mut u32 {
    // SAFETY: nothing else holds the arena while the host writes into it.
    let arena = unsafe { &mut *ARENA.0.get() };
    arena.clear();
    arena.resize((bytes as usize).div_ceil(4), 0);
    arena.as_mut_ptr()
}

/// Empties the arena; the host calls it around each call.
#[unsafe(export_name = "blot:reset")]
extern "C" fn reset(_: u32) -> u32 {
    // SAFETY: as in allocate.
    unsafe { &mut *ARENA.0.get() }.clear();
    0
}

#[unsafe(export_name = "pen:main")]
extern "C" fn main() {
    let mut outgoing = Vec::new();
    crate::run(|response| {
        outgoing.clear();
        outgoing.push(response.len() as u32);
        outgoing.extend(response.iter().map(|lane| lane.to_bits()));
        // SAFETY: the host reads `outgoing` during the call and answers with
        // an array in the arena, which stays until the next allocate.
        unsafe {
            let answer = exchange(outgoing.as_ptr());
            let lanes = std::slice::from_raw_parts(answer.add(1), *answer as usize);
            lanes.iter().map(|word| f32::from_bits(*word)).collect()
        }
    });
}
