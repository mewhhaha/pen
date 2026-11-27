//! Authoritative client-server replication: per-client baseline snapshot-delta
//! encoding (bitmask-filtered changed lanes + entity spawns/resizes/removals)
//! and a bounded reliable / unreliable RPC channel with 32-bit sequence ack
//! bitmasks, standalone ACK packets, and retransmission.
use std::collections::{BTreeMap, HashSet, VecDeque};

use anyhow::{Result, bail, ensure};

const TAG_SNAPSHOT: u8 = 0x10;
const TAG_ACK: u8 = 0x11;
const TAG_RPC_RELIABLE: u8 = 0x12;
const TAG_RPC_UNRELIABLE: u8 = 0x13;
const TAG_RPC_ACK: u8 = 0x14;
const TAG_RESYNC_REQ: u8 = 0x15;

const SNAPSHOT_FLAG_HAS_BASELINE: u8 = 0x01;
const ENTITY_FLAG_FULL_REPLACE: u8 = 0x01;

/// Safety limits bounding memory expansion on untrusted snapshot deltas.
pub const MAX_SNAPSHOT_ENTITIES: usize = 4096;
pub const MAX_ENTITY_LANES: usize = 256;
pub const MAX_SNAPSHOT_TOTAL_LANES: usize = 65_536;
pub const MAX_CLIENT_BASELINES: usize = 64;

/// Safety limits bounding RPC queues and wire payloads.
pub const MAX_RPC_PAYLOAD: usize = 16_384;
pub const MAX_RPC_SEND_WINDOW: usize = 32;
pub const MAX_RPC_PENDING_BYTES: usize = 262_144;
pub const MAX_RPC_DELIVERED: usize = 256;

/// One replicated entity's numeric state lanes.
#[derive(Clone, Debug, PartialEq)]
pub struct EntityState {
    pub id: u32,
    pub lanes: Vec<f32>,
}

/// Complete authoritative snapshot for a single server tick.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorldSnapshot {
    pub tick: u32,
    pub entities: BTreeMap<u32, Vec<f32>>,
}

/// Compact delta packet from `baseline_tick` (`None` = full snapshot) to `tick`.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotDelta {
    pub tick: u32,
    pub baseline_tick: Option<u32>,
    pub updated: Vec<DeltaEntity>,
    pub removed: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeltaEntity {
    pub id: u32,
    pub total_lanes: u16,
    /// True when the entity is spawned or resized (all `0..total_lanes` must be supplied,
    /// including when `total_lanes == 0`).
    pub full_replace: bool,
    /// `(lane_index, value)` for changed lanes (or all lanes when `full_replace` / full snapshot).
    pub changed: Vec<(u16, f32)>,
}

/// Outcome of applying a [`SnapshotDelta`] on the client.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyOutcome {
    /// Delta accepted and applied as the new latest snapshot.
    Applied,
    /// Delta tick is not newer than the client's latest applied tick.
    Stale,
    /// Delta references a baseline no longer held by the client; caller should
    /// transmit [`ClientReplicator::encode_resync_request`] to trigger a full snapshot.
    NeedsFullSnapshot,
}

impl SnapshotDelta {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.updated.len() <= MAX_SNAPSHOT_ENTITIES, "snapshot updated entity count exceeds limit");
        ensure!(self.removed.len() <= MAX_SNAPSHOT_ENTITIES, "snapshot removed entity count exceeds limit");
        let is_full_snapshot = self.baseline_tick.is_none();
        if is_full_snapshot {
            ensure!(self.removed.is_empty(), "full snapshot must not carry removals");
        }
        let mut seen_ids = HashSet::with_capacity(self.updated.len());
        let mut total_lanes = 0usize;
        for ent in &self.updated {
            ensure!(seen_ids.insert(ent.id), "duplicate entity {} in snapshot delta", ent.id);
            let lanes = usize::from(ent.total_lanes);
            ensure!(lanes <= MAX_ENTITY_LANES, "entity {} lane count {} exceeds limit", ent.id, lanes);
            total_lanes = total_lanes.saturating_add(lanes);
            ensure!(total_lanes <= MAX_SNAPSHOT_TOTAL_LANES, "snapshot total lane count exceeds limit");
            let require_all = is_full_snapshot || ent.full_replace;
            if require_all {
                ensure!(
                    ent.changed.len() == lanes,
                    "full/spawned/resized entity {} must supply all {} lanes (got {})",
                    ent.id,
                    lanes,
                    ent.changed.len()
                );
            } else {
                ensure!(ent.changed.len() <= lanes, "entity {} changed lane count exceeds total_lanes", ent.id);
            }
            let mut seen_lanes = if require_all { vec![false; lanes] } else { Vec::new() };
            for &(idx, val) in &ent.changed {
                ensure!(val.is_finite() && usize::from(idx) < lanes, "invalid delta lane");
                if require_all {
                    let slot = &mut seen_lanes[usize::from(idx)];
                    ensure!(!*slot, "duplicate lane {idx} in full entity {}", ent.id);
                    *slot = true;
                }
            }
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut out = Vec::new();
        out.push(TAG_SNAPSHOT);
        let flags = if self.baseline_tick.is_some() { SNAPSHOT_FLAG_HAS_BASELINE } else { 0 };
        out.push(flags);
        out.extend_from_slice(&self.tick.to_le_bytes());
        out.extend_from_slice(&self.baseline_tick.unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&u32::try_from(self.updated.len())?.to_le_bytes());
        for ent in &self.updated {
            out.extend_from_slice(&ent.id.to_le_bytes());
            out.push(if ent.full_replace { ENTITY_FLAG_FULL_REPLACE } else { 0 });
            out.extend_from_slice(&ent.total_lanes.to_le_bytes());
            out.extend_from_slice(&u16::try_from(ent.changed.len())?.to_le_bytes());
            for &(idx, val) in &ent.changed {
                out.extend_from_slice(&idx.to_le_bytes());
                out.extend_from_slice(&val.to_le_bytes());
            }
        }
        out.extend_from_slice(&u32::try_from(self.removed.len())?.to_le_bytes());
        for &id in &self.removed {
            out.extend_from_slice(&id.to_le_bytes());
        }
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() >= 14 && bytes[0] == TAG_SNAPSHOT, "invalid snapshot delta packet");
        let flags = bytes[1];
        ensure!(flags & !SNAPSHOT_FLAG_HAS_BASELINE == 0, "unknown snapshot flags {flags:#04x}");
        let mut pos = 2;
        let tick = read_u32(bytes, &mut pos)?;
        let raw_baseline = read_u32(bytes, &mut pos)?;
        let baseline_tick = if (flags & SNAPSHOT_FLAG_HAS_BASELINE) != 0 { Some(raw_baseline) } else { None };
        let updated_count = read_u32(bytes, &mut pos)? as usize;
        ensure!(updated_count <= MAX_SNAPSHOT_ENTITIES, "snapshot updated entity count exceeds limit");
        let mut updated = Vec::with_capacity(updated_count.min(256));
        for _ in 0..updated_count {
            let id = read_u32(bytes, &mut pos)?;
            let eflags = read_u8(bytes, &mut pos)?;
            ensure!(eflags & !ENTITY_FLAG_FULL_REPLACE == 0, "unknown delta entity flags");
            let full_replace = (eflags & ENTITY_FLAG_FULL_REPLACE) != 0;
            let total_lanes = read_u16(bytes, &mut pos)?;
            ensure!(usize::from(total_lanes) <= MAX_ENTITY_LANES, "delta entity lane count exceeds limit");
            let changed_count = usize::from(read_u16(bytes, &mut pos)?);
            ensure!(changed_count <= usize::from(total_lanes), "changed count exceeds total_lanes");
            let mut changed = Vec::with_capacity(changed_count);
            for _ in 0..changed_count {
                let idx = read_u16(bytes, &mut pos)?;
                let val = f32::from_bits(read_u32(bytes, &mut pos)?);
                ensure!(val.is_finite() && idx < total_lanes, "invalid delta lane");
                changed.push((idx, val));
            }
            updated.push(DeltaEntity { id, total_lanes, full_replace, changed });
        }
        let removed_count = read_u32(bytes, &mut pos)? as usize;
        ensure!(removed_count <= MAX_SNAPSHOT_ENTITIES, "snapshot removed entity count exceeds limit");
        let mut removed = Vec::with_capacity(removed_count.min(256));
        for _ in 0..removed_count {
            removed.push(read_u32(bytes, &mut pos)?);
        }
        ensure!(pos == bytes.len(), "trailing bytes in snapshot delta");
        let delta = Self { tick, baseline_tick, updated, removed };
        delta.validate()?;
        Ok(delta)
    }
}

/// Server-side authoritative replication state with bounded per-tick history.
pub struct ServerReplicator {
    max_history: usize,
    history: VecDeque<WorldSnapshot>,
    client_acked_tick: BTreeMap<u32, u32>,
}

impl ServerReplicator {
    pub fn new(max_history: usize) -> Self {
        Self { max_history: max_history.max(2), history: VecDeque::new(), client_acked_tick: BTreeMap::new() }
    }

    /// Publishes the authoritative world state for `tick`.
    pub fn commit(&mut self, snapshot: WorldSnapshot) {
        self.history.push_back(snapshot);
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Records the latest snapshot tick acknowledged by `client_id`.
    pub fn ack_client(&mut self, client_id: u32, tick: u32) {
        match self.client_acked_tick.get_mut(&client_id) {
            Some(entry) => {
                if tick > *entry {
                    *entry = tick;
                }
            }
            None => {
                self.client_acked_tick.insert(client_id, tick);
            }
        }
    }

    /// Clears `client_id`'s acknowledged baseline so the next [`Self::build_delta`]
    /// sends a full snapshot (e.g., after receiving a resync request).
    pub fn reset_client_baseline(&mut self, client_id: u32) {
        self.client_acked_tick.remove(&client_id);
    }

    /// Computes a delta packet for `client_id` against its last acked baseline
    /// (falling back to a full snapshot if the baseline was evicted or unknown).
    pub fn build_delta(&self, client_id: u32) -> Result<SnapshotDelta> {
        let latest = self.history.back().ok_or_else(|| anyhow::anyhow!("no committed snapshot"))?;
        let acked = self.client_acked_tick.get(&client_id).copied();
        let baseline = acked.and_then(|t| self.history.iter().find(|s| s.tick == t));
        let Some(base) = baseline else {
            let updated = latest
                .entities
                .iter()
                .map(|(&id, lanes)| {
                    ensure!(lanes.len() <= MAX_ENTITY_LANES, "entity {id} exceeds max lanes");
                    let total_lanes = u16::try_from(lanes.len())?;
                    let changed = lanes
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| Ok((u16::try_from(i)?, v)))
                        .collect::<Result<Vec<_>>>()?;
                    Ok(DeltaEntity { id, total_lanes, full_replace: true, changed })
                })
                .collect::<Result<Vec<_>>>()?;
            let delta = SnapshotDelta { tick: latest.tick, baseline_tick: None, updated, removed: Vec::new() };
            delta.validate()?;
            return Ok(delta);
        };

        let mut updated = Vec::new();
        for (&id, lanes) in &latest.entities {
            ensure!(lanes.len() <= MAX_ENTITY_LANES, "entity {id} exceeds max lanes");
            let total_lanes = u16::try_from(lanes.len())?;
            match base.entities.get(&id) {
                Some(prev) if prev.len() == lanes.len() => {
                    let changed: Vec<(u16, f32)> = lanes
                        .iter()
                        .zip(prev)
                        .enumerate()
                        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
                        .map(|(i, (&v, _))| Ok((u16::try_from(i)?, v)))
                        .collect::<Result<Vec<_>>>()?;
                    if !changed.is_empty() {
                        updated.push(DeltaEntity { id, total_lanes, full_replace: false, changed });
                    }
                }
                _ => {
                    // New entity or resized lane count (including 0-lane spawn/resize).
                    let changed = lanes
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| Ok((u16::try_from(i)?, v)))
                        .collect::<Result<Vec<_>>>()?;
                    updated.push(DeltaEntity { id, total_lanes, full_replace: true, changed });
                }
            }
        }
        let removed = base.entities.keys().filter(|id| !latest.entities.contains_key(id)).copied().collect();
        let delta = SnapshotDelta { tick: latest.tick, baseline_tick: Some(base.tick), updated, removed };
        delta.validate()?;
        Ok(delta)
    }
}

/// Client-side state reconstructed from full & delta snapshots.
#[derive(Default)]
pub struct ClientReplicator {
    baselines: BTreeMap<u32, WorldSnapshot>,
    latest_tick: Option<u32>,
    needs_resync: bool,
}

impl ClientReplicator {
    pub fn latest_tick(&self) -> Option<u32> {
        self.latest_tick
    }

    pub fn needs_resync(&self) -> bool {
        self.needs_resync
    }

    pub fn current(&self) -> Option<&WorldSnapshot> {
        self.latest_tick.and_then(|t| self.baselines.get(&t))
    }

    /// Applies a [`SnapshotDelta`], validating all bounds and entity/baseline invariants.
    pub fn apply(&mut self, delta: &SnapshotDelta) -> Result<ApplyOutcome> {
        delta.validate()?;
        if let Some(latest) = self.latest_tick
            && delta.tick <= latest
        {
            return Ok(ApplyOutcome::Stale);
        }
        let mut next = match delta.baseline_tick {
            None => WorldSnapshot { tick: delta.tick, entities: BTreeMap::new() },
            Some(base_tick) => {
                let Some(existing) = self.baselines.get(&base_tick) else {
                    self.needs_resync = true;
                    return Ok(ApplyOutcome::NeedsFullSnapshot);
                };
                let mut base = existing.clone();
                base.tick = delta.tick;
                base
            }
        };
        for &id in &delta.removed {
            next.entities.remove(&id);
        }
        for ent in &delta.updated {
            let lanes = usize::from(ent.total_lanes);
            if ent.full_replace || delta.baseline_tick.is_none() {
                let mut vec = vec![0.0f32; lanes];
                for &(idx, val) in &ent.changed {
                    vec[usize::from(idx)] = val;
                }
                next.entities.insert(ent.id, vec);
            } else {
                let entry = next
                    .entities
                    .get_mut(&ent.id)
                    .ok_or_else(|| anyhow::anyhow!("partial delta for missing entity {}", ent.id))?;
                ensure!(
                    entry.len() == lanes,
                    "partial delta lane count {} mismatches baseline {} for entity {}",
                    lanes,
                    entry.len(),
                    ent.id
                );
                for &(idx, val) in &ent.changed {
                    entry[usize::from(idx)] = val;
                }
            }
        }
        ensure!(next.entities.len() <= MAX_SNAPSHOT_ENTITIES, "reconstructed snapshot exceeds entity limit");
        let total_lanes: usize = next.entities.values().map(Vec::len).sum();
        ensure!(total_lanes <= MAX_SNAPSHOT_TOTAL_LANES, "reconstructed snapshot exceeds lane limit");
        self.latest_tick = Some(delta.tick);
        self.needs_resync = false;
        self.baselines.insert(delta.tick, next);
        while self.baselines.len() > MAX_CLIENT_BASELINES {
            if let Some(&oldest) = self.baselines.keys().next() {
                self.baselines.remove(&oldest);
            }
        }
        Ok(ApplyOutcome::Applied)
    }

    pub fn encode_ack(&self) -> Option<Vec<u8>> {
        let tick = self.latest_tick?;
        let mut out = Vec::with_capacity(5);
        out.push(TAG_ACK);
        out.extend_from_slice(&tick.to_le_bytes());
        Some(out)
    }

    pub fn decode_ack(bytes: &[u8]) -> Result<u32> {
        ensure!(bytes.len() == 5 && bytes[0] == TAG_ACK, "invalid ack packet");
        Ok(u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]))
    }

    /// Encodes a 1-byte full-snapshot resync request to recover from an evicted baseline.
    pub fn encode_resync_request(&self) -> [u8; 1] {
        [TAG_RESYNC_REQ]
    }

    pub fn is_resync_request(bytes: &[u8]) -> bool {
        bytes == [TAG_RESYNC_REQ]
    }
}

/// Reliable-ordered + unreliable RPC datagram channel with bounded windows,
/// 32-bit ack bitmask, standalone ACK frames, and automatic retransmission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RpcMessage {
    pub kind: u16,
    pub reliable: bool,
    pub payload: Vec<u8>,
}

#[derive(Default)]
pub struct RpcChannel {
    next_send_seq: u32,
    next_recv_seq: u32,
    recv_bits: u32,
    pending_bytes: usize,
    pending_send: BTreeMap<u32, (u16, Vec<u8>)>,
    reorder_buf: BTreeMap<u32, RpcMessage>,
    delivered: VecDeque<RpcMessage>,
}

impl RpcChannel {
    /// Queues a reliable or unreliable RPC and returns the encoded wire datagram.
    pub fn send(&mut self, kind: u16, reliable: bool, payload: Vec<u8>) -> Result<Vec<u8>> {
        ensure!(payload.len() <= MAX_RPC_PAYLOAD, "RPC payload exceeds {MAX_RPC_PAYLOAD} bytes");
        if !reliable {
            let mut out = Vec::with_capacity(3 + payload.len());
            out.push(TAG_RPC_UNRELIABLE);
            out.extend_from_slice(&kind.to_le_bytes());
            out.extend_from_slice(&payload);
            return Ok(out);
        }
        ensure!(self.pending_send.len() < MAX_RPC_SEND_WINDOW, "reliable RPC send window full");
        ensure!(
            self.pending_bytes.saturating_add(payload.len()) <= MAX_RPC_PENDING_BYTES,
            "reliable RPC pending byte budget exceeded"
        );
        let seq = self.next_send_seq;
        self.next_send_seq = self.next_send_seq.wrapping_add(1);
        self.pending_bytes += payload.len();
        let encoded = self.encode_reliable(seq, kind, &payload);
        self.pending_send.insert(seq, (kind, payload));
        Ok(encoded)
    }

    /// Encodes a standalone ACK frame so one-way reliable traffic is acknowledged
    /// without needing to send a return reliable RPC.
    pub fn encode_ack(&self) -> [u8; 9] {
        let mut out = [0u8; 9];
        out[0] = TAG_RPC_ACK;
        out[1..5].copy_from_slice(&self.next_recv_seq.to_le_bytes());
        out[5..9].copy_from_slice(&self.recv_bits.to_le_bytes());
        out
    }

    /// Returns encoded packets for all unacknowledged reliable RPCs (retransmit).
    pub fn retransmit_pending(&self) -> Vec<Vec<u8>> {
        self.pending_send.iter().map(|(&seq, (kind, payload))| self.encode_reliable(seq, *kind, payload)).collect()
    }

    pub fn pending_count(&self) -> usize {
        self.pending_send.len()
    }

    /// Processes an incoming RPC or ACK wire packet, updating ack state and
    /// queueing newly ordered messages for [`Self::poll`].
    pub fn receive(&mut self, bytes: &[u8]) -> Result<()> {
        ensure!(!bytes.is_empty(), "empty RPC packet");
        match bytes[0] {
            TAG_RPC_UNRELIABLE => {
                ensure!(bytes.len() >= 3 && bytes.len() - 3 <= MAX_RPC_PAYLOAD, "invalid unreliable RPC");
                let kind = u16::from_le_bytes([bytes[1], bytes[2]]);
                if self.delivered.len() < MAX_RPC_DELIVERED {
                    self.delivered.push_back(RpcMessage { kind, reliable: false, payload: bytes[3..].to_vec() });
                }
            }
            TAG_RPC_ACK => {
                ensure!(bytes.len() == 9, "invalid RPC ack packet");
                let ack = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
                let ack_bits = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
                self.process_acks(ack, ack_bits)?;
            }
            TAG_RPC_RELIABLE => {
                ensure!(bytes.len() >= 15 && bytes.len() - 15 <= MAX_RPC_PAYLOAD, "invalid reliable RPC");
                let seq = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
                let ack = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
                let ack_bits = u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]);
                let kind = u16::from_le_bytes([bytes[13], bytes[14]]);
                let payload = bytes[15..].to_vec();
                self.process_acks(ack, ack_bits)?;
                let delta = seq.wrapping_sub(self.next_recv_seq);
                // Strictly bound reorder buffer to the 32-message receive window.
                if delta < 32 {
                    self.recv_bits |= 1u32 << delta;
                    self.reorder_buf.entry(seq).or_insert(RpcMessage { kind, reliable: true, payload });
                    while let Some(msg) = self.reorder_buf.remove(&self.next_recv_seq) {
                        if self.delivered.len() >= MAX_RPC_DELIVERED {
                            // Put back and stop advancing until caller drains delivered.
                            self.reorder_buf.insert(self.next_recv_seq, msg);
                            break;
                        }
                        self.delivered.push_back(msg);
                        self.next_recv_seq = self.next_recv_seq.wrapping_add(1);
                        self.recv_bits >>= 1;
                    }
                }
            }
            other => bail!("unknown RPC packet tag {other:#04x}"),
        }
        Ok(())
    }

    /// Pops the next delivered RPC message.
    pub fn poll(&mut self) -> Option<RpcMessage> {
        self.delivered.pop_front()
    }

    fn encode_reliable(&self, seq: u32, kind: u16, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(15 + payload.len());
        out.push(TAG_RPC_RELIABLE);
        out.extend_from_slice(&seq.to_le_bytes());
        out.extend_from_slice(&self.next_recv_seq.to_le_bytes());
        out.extend_from_slice(&self.recv_bits.to_le_bytes());
        out.extend_from_slice(&kind.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn process_acks(&mut self, next_expected: u32, bits: u32) -> Result<()> {
        // Validate that `next_expected` does not exceed `next_send_seq` (modulo 2^32).
        let ahead = next_expected.wrapping_sub(self.next_send_seq);
        ensure!(
            ahead == 0 || ahead >= 0x8000_0000,
            "ACK sequence {next_expected} exceeds sent sequence {}",
            self.next_send_seq
        );
        let mut freed = 0usize;
        self.pending_send.retain(|&seq, (_, payload)| {
            // Distance from `seq` to `next_expected` (positive if seq < next_expected).
            let behind = next_expected.wrapping_sub(seq);
            let acked_cumul = behind > 0 && behind < 0x8000_0000;
            let diff = seq.wrapping_sub(next_expected);
            let acked_bit = diff < 32 && (bits & (1u32 << diff)) != 0;
            if acked_cumul || acked_bit {
                freed += payload.len();
                false
            } else {
                true
            }
        });
        self.pending_bytes = self.pending_bytes.saturating_sub(freed);
        Ok(())
    }
}

fn read_u8(bytes: &[u8], pos: &mut usize) -> Result<u8> {
    ensure!(*pos < bytes.len(), "unexpected EOF reading u8");
    let v = bytes[*pos];
    *pos += 1;
    Ok(v)
}

fn read_u16(bytes: &[u8], pos: &mut usize) -> Result<u16> {
    ensure!(*pos + 2 <= bytes.len(), "unexpected EOF reading u16");
    let v = u16::from_le_bytes([bytes[*pos], bytes[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32> {
    ensure!(*pos + 4 <= bytes.len(), "unexpected EOF reading u32");
    let v = u32::from_le_bytes([bytes[*pos], bytes[*pos + 1], bytes[*pos + 2], bytes[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_delta_compresses_unchanged_lanes_handles_tick_zero_and_zero_lane_entities() {
        let mut server = ServerReplicator::new(16);
        let mut client = ClientReplicator::default();

        // Tick 0 initial snapshot must be accepted.
        let mut s0 = WorldSnapshot { tick: 0, entities: BTreeMap::new() };
        s0.entities.insert(10, vec![1.0, 2.0, 3.0, 4.0]);
        s0.entities.insert(20, vec![5.0, 6.0]);
        server.commit(s0.clone());

        let full = server.build_delta(1).unwrap();
        let full_bytes = full.encode().unwrap();
        assert_eq!(client.apply(&SnapshotDelta::decode(&full_bytes).unwrap()).unwrap(), ApplyOutcome::Applied);
        assert_eq!(client.current(), Some(&s0));
        server.ack_client(1, ClientReplicator::decode_ack(&client.encode_ack().unwrap()).unwrap());

        // Tick 1: lane 0 of 10 changes, 20 despawns, 30 spawns with 0 lanes!
        let mut s1 = WorldSnapshot { tick: 1, entities: BTreeMap::new() };
        s1.entities.insert(10, vec![9.5, 2.0, 3.0, 4.0]);
        s1.entities.insert(30, vec![]);
        server.commit(s1.clone());

        let delta = server.build_delta(1).unwrap();
        let delta_bytes = delta.encode().unwrap();
        assert_eq!(delta.baseline_tick, Some(0));
        assert_eq!(delta.removed, vec![20]);
        assert!(delta_bytes.len() < full_bytes.len());
        assert_eq!(client.apply(&SnapshotDelta::decode(&delta_bytes).unwrap()).unwrap(), ApplyOutcome::Applied);
        assert_eq!(client.current(), Some(&s1));
        server.ack_client(1, ClientReplicator::decode_ack(&client.encode_ack().unwrap()).unwrap());

        // Tick 2: entity 10 resizes to 0 lanes.
        let mut s2 = WorldSnapshot { tick: 2, entities: BTreeMap::new() };
        s2.entities.insert(10, vec![]);
        s2.entities.insert(30, vec![]);
        server.commit(s2.clone());
        let d2 = server.build_delta(1).unwrap();
        assert_eq!(client.apply(&d2).unwrap(), ApplyOutcome::Applied);
        assert_eq!(client.current(), Some(&s2));
    }

    #[test]
    fn snapshot_rejects_sparse_oom_amplification_and_recovers_missing_baseline() {
        // Sparse full snapshot declaring 200 lanes with 0 supplied values must fail decode & apply.
        let sparse = SnapshotDelta {
            tick: 1,
            baseline_tick: None,
            updated: vec![DeltaEntity { id: 1, total_lanes: 200, full_replace: true, changed: vec![] }],
            removed: vec![],
        };
        assert!(sparse.validate().is_err());
        let mut client = ClientReplicator::default();
        assert!(client.apply(&sparse).is_err());

        // Missing baseline requests resync and recovers on full snapshot.
        let mut server = ServerReplicator::new(4);
        let mut snap = WorldSnapshot { tick: 5, entities: BTreeMap::new() };
        snap.entities.insert(1, vec![1.0]);
        server.commit(snap.clone());
        let missing = SnapshotDelta { tick: 5, baseline_tick: Some(2), updated: vec![], removed: vec![] };
        assert_eq!(client.apply(&missing).unwrap(), ApplyOutcome::NeedsFullSnapshot);
        assert!(client.needs_resync());
        assert!(ClientReplicator::is_resync_request(&client.encode_resync_request()));
        server.reset_client_baseline(1);
        let recovery = server.build_delta(1).unwrap();
        assert_eq!(recovery.baseline_tick, None);
        assert_eq!(client.apply(&recovery).unwrap(), ApplyOutcome::Applied);
        assert!(!client.needs_resync());
        assert_eq!(client.current(), Some(&snap));
    }

    #[test]
    fn reliable_rpc_reorders_acks_one_way_and_enforces_bounds() {
        let mut a = RpcChannel::default();
        let mut b = RpcChannel::default();
        let p0 = a.send(1, true, b"first".to_vec()).unwrap();
        let p1 = a.send(2, true, b"second".to_vec()).unwrap();
        let pu = a.send(9, false, b"ping".to_vec()).unwrap();

        b.receive(&p1).unwrap();
        assert_eq!(b.poll(), None, "must wait for seq 0");
        b.receive(&pu).unwrap();
        assert_eq!(b.poll().unwrap().kind, 9);
        b.receive(&p0).unwrap();
        assert_eq!(b.poll().unwrap().payload, b"first");
        assert_eq!(b.poll().unwrap().payload, b"second");

        // One-way traffic acknowledged via standalone ACK packet without sending a reliable RPC back.
        let ack = b.encode_ack();
        a.receive(&ack).unwrap();
        assert_eq!(a.pending_count(), 0);

        // Far-ahead sequence outside 32-message window is ignored (no unbounded reorder_buf growth).
        let mut forged = p0.clone();
        forged[1..5].copy_from_slice(&1000u32.to_le_bytes());
        b.receive(&forged).unwrap();
        assert!(b.reorder_buf.is_empty());

        // Forged future ACK beyond sent sequence is rejected.
        let mut bad_ack = b.encode_ack();
        bad_ack[1..5].copy_from_slice(&999u32.to_le_bytes());
        assert!(a.receive(&bad_ack).is_err());

        // Send window caps at 32 in-flight reliable RPCs.
        for i in 0..MAX_RPC_SEND_WINDOW {
            a.send(1, true, vec![u8::try_from(i).unwrap()]).unwrap();
        }
        assert!(a.send(1, true, vec![0]).is_err());
    }
}
