//! Target-independent browser host policy logic shared between the `wasm32` host
//! (`video`, `kernels`, `audio`, `fetch_bounded`) and native unit tests.

/// Maximum simultaneous in-flight HTTP video fetches.
pub const MAX_CONCURRENT_VIDEO_FETCHES: usize = 2;
/// Maximum payload bytes accepted for a single video stream (32 MiB).
pub const MAX_VIDEO_STREAM_BYTES: usize = 32 * 1024 * 1024;
/// Cumulative byte budget across all resident video streams (64 MiB).
pub const MAX_VIDEO_TOTAL_BYTES: usize = 64 * 1024 * 1024;
/// Evict resident CPU + GPU video entries after this many unused frames.
pub const VIDEO_EVICT_IDLE_FRAMES: u64 = 180;
/// Base retry backoff after transient fetch/parse errors (frames).
pub const VIDEO_RETRY_BASE_FRAMES: u64 = 60;
/// Maximum retry backoff cap (frames).
pub const VIDEO_RETRY_MAX_FRAMES: u64 = 600;

/// Computes exponential retry backoff in frames for failure attempt `next_attempts` (1-based).
pub fn video_retry_backoff_frames(next_attempts: u32) -> u64 {
    (VIDEO_RETRY_BASE_FRAMES << next_attempts.min(4)).min(VIDEO_RETRY_MAX_FRAMES)
}

/// Computes remaining byte budget available for fetching a new video stream.
pub fn video_remaining_budget(current_total_bytes: usize) -> usize {
    MAX_VIDEO_TOTAL_BYTES.saturating_sub(current_total_bytes).min(MAX_VIDEO_STREAM_BYTES)
}

/// Validates an HTTP `Content-Length` header value against `max_bytes` before allocating.
pub fn content_length_within_limit(header_value: &str, max_bytes: usize) -> bool {
    match header_value.trim().parse::<u64>() {
        Ok(declared) => declared <= max_bytes as u64,
        Err(_) => true,
    }
}

/// Selects video stream keys to evict based on idle frame threshold and LRU byte budget.
///
/// Each entry in `streams` is `(key, last_used_frame, byte_len, touched_this_frame)`.
/// Returns keys to evict in order: idle streams first, then oldest untouched streams until
/// the remaining cumulative byte total is at most `max_total_bytes`.
pub fn select_video_evictions(
    streams: &[(String, u64, usize, bool)],
    now_frame: u64,
    total_bytes: usize,
    idle_limit: u64,
    max_total_bytes: usize,
) -> Vec<String> {
    let mut evict = Vec::new();
    let mut remaining_bytes = total_bytes;

    for (key, last_used, bytes, touched) in streams {
        if !*touched && now_frame.saturating_sub(*last_used) >= idle_limit {
            evict.push(key.clone());
            remaining_bytes = remaining_bytes.saturating_sub(*bytes);
        }
    }

    if remaining_bytes > max_total_bytes {
        let mut candidates: Vec<(&str, u64, usize)> = streams
            .iter()
            .filter(|(k, _, _, touched)| !*touched && !evict.iter().any(|e| e == k))
            .map(|(k, last, bytes, _)| (k.as_str(), *last, *bytes))
            .collect();
        candidates.sort_by_key(|&(_, last, _)| last);
        for (key, _, bytes) in candidates {
            if remaining_bytes <= max_total_bytes {
                break;
            }
            evict.push(key.to_owned());
            remaining_bytes = remaining_bytes.saturating_sub(bytes);
        }
    }
    evict
}

/// Computes how many events are available for browser kernel replies this frame after
/// reserving `cap_events` when a `CAP` announcement is pending.
pub fn kernel_reply_event_budget(
    max_events: usize,
    base_count: usize,
    pending_cap: bool,
    cap_events: usize,
    max_kernel_reply_events: usize,
) -> usize {
    let available = max_events.saturating_sub(base_count);
    let cap_reserve = if pending_cap && available >= cap_events { cap_events } else { 0 };
    available.saturating_sub(cap_reserve).min(max_kernel_reply_events)
}

/// Returns whether a completed async browser kernel batch belongs to the live host epoch
/// (rejecting replies whose host epoch was fenced by a snapshot restore/reset while awaiting).
pub fn kernel_epoch_valid(batch_epoch: u64, live_host_epoch: u64) -> bool {
    batch_epoch == live_host_epoch
}

/// Clamps noise buffer sample count for WebAudio procedural noise generators.
pub fn noise_buffer_len(sample_rate: f32) -> usize {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped positive sample rate")]
    {
        sample_rate.clamp(4096.0, 48_000.0) as usize
    }
}

/// Clamps half-second granular loop frame count for WebAudio granular synths.
pub fn granular_loop_frames(sample_rate: f32) -> usize {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "clamped positive half-second frame count")]
    {
        (sample_rate * 0.5).clamp(256.0, 96_000.0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_backoff_doubles_and_caps_cleanly() {
        assert_eq!(video_retry_backoff_frames(1), 120);
        assert_eq!(video_retry_backoff_frames(2), 240);
        assert_eq!(video_retry_backoff_frames(3), 480);
        assert_eq!(video_retry_backoff_frames(4), VIDEO_RETRY_MAX_FRAMES);
        assert_eq!(video_retry_backoff_frames(99), VIDEO_RETRY_MAX_FRAMES);
    }

    #[test]
    fn video_budget_and_content_length_bounds() {
        assert_eq!(video_remaining_budget(0), MAX_VIDEO_STREAM_BYTES);
        assert_eq!(video_remaining_budget(MAX_VIDEO_TOTAL_BYTES - 1024), 1024);
        assert_eq!(video_remaining_budget(MAX_VIDEO_TOTAL_BYTES + 100), 0);
        assert!(content_length_within_limit("1024", 1024));
        assert!(!content_length_within_limit("1025", 1024));
        assert!(content_length_within_limit("malformed", 1024));
    }

    #[test]
    fn video_eviction_stops_once_byte_budget_is_satisfied() {
        let streams = vec![
            ("idle.y4m".into(), 10, 10_000, false),
            ("old_untouched.y4m".into(), 190, 25_000, false),
            ("newer_untouched.y4m".into(), 195, 25_000, false),
            ("active.y4m".into(), 200, 30_000, true),
        ];
        // Total = 90_000, max = 60_000.
        // Idle eviction removes "idle.y4m" (10_000 -> 80_000 left).
        // LRU eviction must remove ONLY "old_untouched.y4m" (25_000 -> 55_000 <= 60_000),
        // keeping "newer_untouched.y4m" and "active.y4m" resident!
        let evicted = select_video_evictions(&streams, 200, 90_000, VIDEO_EVICT_IDLE_FRAMES, 60_000);
        assert_eq!(evicted, vec!["idle.y4m".to_owned(), "old_untouched.y4m".to_owned()]);
    }

    #[test]
    fn kernel_budget_reserves_cap_and_rejects_fenced_epochs() {
        // Reserve 3 CAP events before granting reply slots.
        assert_eq!(kernel_reply_event_budget(64, 60, true, 3, 32), 1);
        assert_eq!(kernel_reply_event_budget(64, 60, false, 3, 32), 4);
        assert_eq!(kernel_reply_event_budget(64, 10, true, 3, 16), 16);
        assert!(kernel_epoch_valid(7, 7));
        assert!(!kernel_epoch_valid(7, 8));
    }

    #[test]
    fn audio_buffer_lengths_clamp_safely_for_extreme_sample_rates() {
        assert_eq!(noise_buffer_len(48_000.0), 48_000);
        assert_eq!(noise_buffer_len(96_000.0), 48_000);
        assert_eq!(noise_buffer_len(-100.0), 4096);
        assert_eq!(granular_loop_frames(48_000.0), 24_000);
        assert_eq!(granular_loop_frames(0.0), 256);
    }
}
