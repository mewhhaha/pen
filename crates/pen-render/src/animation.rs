//! Skeletal animation: a skinned mesh's rig (its skeleton's nodes in their
//! rest pose, the joints its vertices are bound to, and its clips), sampled
//! and blended into the joint matrices that move a skinned draw's vertices.
//! The renderer poses skinned draws with it each frame; the glTF importer
//! bakes vertex-animation textures with the same sampling.
//!
//! Mesh JSON holds a rig beside its arrays (`skeleton`, `skin`, `clips`, see
//! [`RigJson`]), and the binary mesh form after its indices (`write_rig`).
//! Matrices are column-major, as WGSL and the render packet lay them out.
use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::matrix::narrow;

/// The most joints a skinned mesh may bind its vertices to, and the most
/// nodes its skeleton may have.
pub const MAX_JOINTS: usize = 1024;

pub type Matrix = [f32; 16];

/// A node's place relative to its parent: translated, rotated (a unit
/// quaternion, x, y, z, w) and scaled, applied scale first.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl Transform {
    pub const IDENTITY: Transform =
        Transform { translation: [0.0; 3], rotation: [0.0, 0.0, 0.0, 1.0], scale: [1.0; 3] };

    /// Its matrix: translation * rotation * scale.
    pub fn matrix(&self) -> [f64; 16] {
        let [x, y, z, w] = self.rotation.map(f64::from);
        let [sx, sy, sz] = self.scale.map(f64::from);
        let [tx, ty, tz] = self.translation.map(f64::from);
        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (xy, xz, yz) = (x * y, x * z, y * z);
        let (wx, wy, wz) = (w * x, w * y, w * z);
        [
            (1.0 - 2.0 * (yy + zz)) * sx,
            2.0 * (xy + wz) * sx,
            2.0 * (xz - wy) * sx,
            0.0,
            2.0 * (xy - wz) * sy,
            (1.0 - 2.0 * (xx + zz)) * sy,
            2.0 * (yz + wx) * sy,
            0.0,
            2.0 * (xz + wy) * sz,
            2.0 * (yz - wx) * sz,
            (1.0 - 2.0 * (xx + yy)) * sz,
            0.0,
            tx,
            ty,
            tz,
            1.0,
        ]
    }
}

/// A node of a skeleton: a joint, or a node above joints that moves them
/// all (an armature's own placement).
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub name: String,
    /// Its parent, which comes before it; None for a root.
    pub parent: Option<u32>,
    pub rest: Transform,
}

/// What a vertex's joint index names: a skeleton node, and the matrix that
/// takes the mesh's bind pose into that node's space.
#[derive(Clone, Debug, PartialEq)]
pub struct Joint {
    pub node: u32,
    pub inverse_bind: Matrix,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Path {
    Translation,
    Rotation,
    Scale,
}

impl Path {
    /// Values a key holds.
    pub fn width(self) -> usize {
        if self == Path::Rotation { 4 } else { 3 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Interpolation {
    Linear,
    Step,
    /// glTF's cubic spline: each key is an in-tangent, a value and an
    /// out-tangent.
    Cubic,
}

/// How one of a node's transform's parts moves over a clip: keys at
/// `times` (seconds, rising), `values` the part's at each.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Channel {
    pub node: u32,
    pub path: Path,
    pub interpolation: Interpolation,
    pub times: Vec<f32>,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Clip {
    pub name: String,
    /// Seconds; a clip's time wraps round it.
    pub duration: f32,
    pub channels: Vec<Channel>,
}

/// One layer of a pose: clip `clip` sampled at `time` seconds, round the
/// clip as often as need be, weighing `weight` among the pose's layers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layer {
    pub clip: u32,
    pub time: f32,
    pub weight: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rig {
    /// Parents before their children.
    pub nodes: Vec<Node>,
    /// What vertices' joint indices name.
    pub joints: Vec<Joint>,
    pub clips: Vec<Clip>,
}

/// A rig as mesh JSON holds it, beside the mesh's arrays.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RigJson {
    pub skeleton: Vec<NodeJson>,
    pub skin: Vec<JointJson>,
    pub clips: Vec<Clip>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeJson {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<u32>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointJson {
    pub node: u32,
    #[serde(rename = "inverseBind")]
    pub inverse_bind: Matrix,
}

impl Rig {
    pub fn from_json(json: RigJson) -> Result<Rig> {
        let RigJson { skeleton, skin, clips } = json;
        let nodes = skeleton
            .into_iter()
            .map(|node| Node {
                name: node.name,
                parent: node.parent,
                rest: Transform { translation: node.translation, rotation: node.rotation, scale: node.scale },
            })
            .collect();
        let joints =
            skin.into_iter().map(|joint| Joint { node: joint.node, inverse_bind: joint.inverse_bind }).collect();
        let rig = Rig { nodes, joints, clips };
        rig.validate()?;
        Ok(rig)
    }

    pub fn json(&self) -> RigJson {
        RigJson {
            skeleton: self
                .nodes
                .iter()
                .map(|node| NodeJson {
                    name: node.name.clone(),
                    parent: node.parent,
                    translation: node.rest.translation,
                    rotation: node.rest.rotation,
                    scale: node.rest.scale,
                })
                .collect(),
            skin: self
                .joints
                .iter()
                .map(|joint| JointJson { node: joint.node, inverse_bind: joint.inverse_bind })
                .collect(),
            clips: self.clips.clone(),
        }
    }

    /// Checks that the rig is whole: parents before their children, joints
    /// and channels naming its nodes, rising key times, values that fit the
    /// keys, and finite numbers throughout.
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.nodes.is_empty() && self.nodes.len() <= MAX_JOINTS, "a skeleton needs 1 to {MAX_JOINTS} nodes");
        ensure!(!self.joints.is_empty() && self.joints.len() <= MAX_JOINTS, "a skin needs 1 to {MAX_JOINTS} joints");
        let count = self.nodes.len();
        for (index, node) in self.nodes.iter().enumerate() {
            ensure!(node.parent.is_none_or(|parent| (parent as usize) < index), "a node's parent must come before it");
            finite_transform(&node.rest)?;
        }
        for joint in &self.joints {
            ensure!((joint.node as usize) < count, "a joint must name a skeleton node");
            ensure!(joint.inverse_bind.iter().all(|value| value.is_finite()), "inverse bind matrices must be finite");
        }
        for clip in &self.clips {
            ensure!(clip.duration.is_finite() && clip.duration >= 0.0, "a clip's duration must be finite");
            for channel in &clip.channels {
                ensure!((channel.node as usize) < count, "a channel must name a skeleton node");
                let keys = channel.times.len();
                let per_key = channel.path.width() * if channel.interpolation == Interpolation::Cubic { 3 } else { 1 };
                ensure!(keys > 0 && channel.values.len() == keys * per_key, "a channel needs a value for each key");
                ensure!(
                    channel.times.iter().all(|time| time.is_finite() && *time >= 0.0)
                        && channel.times.is_sorted()
                        && channel.values.iter().all(|value| value.is_finite()),
                    "a channel's keys must rise in time and hold finite values"
                );
            }
        }
        Ok(())
    }

    /// Each node's place relative to its parent for `layers` blended by
    /// their weights; the rest pose where they weigh nothing. Layers naming
    /// a clip the rig lacks are left out.
    pub fn pose(&self, layers: &[Layer]) -> Vec<Transform> {
        let rest: Vec<Transform> = self.nodes.iter().map(|node| node.rest).collect();
        let layers: Vec<Layer> = layers
            .iter()
            .copied()
            .filter(|layer| (layer.clip as usize) < self.clips.len() && layer.weight > 0.0 && layer.weight.is_finite())
            .collect();
        match layers.as_slice() {
            [] => rest,
            [layer] => self.sampled(&rest, *layer),
            _ => {
                let total: f64 = layers.iter().map(|layer| f64::from(layer.weight)).sum();
                let mut sums = vec![[0.0f64; 10]; rest.len()];
                for layer in &layers {
                    let weight = f64::from(layer.weight) / total;
                    for (sum, local) in sums.iter_mut().zip(self.sampled(&rest, *layer)) {
                        // Quaternions q and -q turn alike: each is added on
                        // the side of the first layer's.
                        let rotation = local.rotation.map(f64::from);
                        let aligned = (0..4).map(|axis| sum[3 + axis] * rotation[axis]).sum::<f64>() >= 0.0;
                        let sign = if aligned { weight } else { -weight };
                        for axis in 0..3 {
                            sum[axis] += weight * f64::from(local.translation[axis]);
                            sum[7 + axis] += weight * f64::from(local.scale[axis]);
                        }
                        for axis in 0..4 {
                            sum[3 + axis] += sign * rotation[axis];
                        }
                    }
                }
                sums.iter()
                    .map(|sum| Transform {
                        translation: [sum[0], sum[1], sum[2]].map(narrow),
                        rotation: normalized([sum[3], sum[4], sum[5], sum[6]]),
                        scale: [sum[7], sum[8], sum[9]].map(narrow),
                    })
                    .collect()
            }
        }
    }

    /// The rest pose with `layer`'s clip's channels in place.
    fn sampled(&self, rest: &[Transform], layer: Layer) -> Vec<Transform> {
        let mut pose = rest.to_vec();
        let clip = &self.clips[layer.clip as usize];
        let time = if clip.duration > 0.0 && layer.time.is_finite() {
            f64::from(layer.time).rem_euclid(f64::from(clip.duration))
        } else {
            0.0
        };
        for channel in &clip.channels {
            let node = &mut pose[channel.node as usize];
            let mut value = [0.0; 4];
            sample(channel, time, &mut value);
            match channel.path {
                Path::Translation => node.translation = [value[0], value[1], value[2]].map(narrow),
                Path::Rotation => node.rotation = normalized(value),
                Path::Scale => node.scale = [value[0], value[1], value[2]].map(narrow),
            }
        }
        pose
    }

    /// Each node's matrix in the mesh's space for `pose`.
    pub fn world(&self, pose: &[Transform]) -> Vec<[f64; 16]> {
        let mut world: Vec<[f64; 16]> = Vec::with_capacity(self.nodes.len());
        for (node, local) in self.nodes.iter().zip(pose) {
            let local = local.matrix();
            let placed = match node.parent {
                Some(parent) => multiply(&world[parent as usize], &local),
                None => local,
            };
            world.push(placed);
        }
        world
    }

    /// The joint matrices for `layers` into `palette`: each joint's node in
    /// the mesh's space, times its inverse bind matrix.
    pub fn palette(&self, layers: &[Layer], palette: &mut Vec<Matrix>) {
        let world = self.world(&self.pose(layers));
        palette.clear();
        palette.extend(
            self.joints
                .iter()
                .map(|joint| multiply(&world[joint.node as usize], &joint.inverse_bind.map(f64::from)).map(narrow)),
        );
    }
}

fn finite_transform(transform: &Transform) -> Result<()> {
    let Transform { translation, rotation, scale } = transform;
    ensure!(
        translation.iter().chain(rotation).chain(scale).all(|value| value.is_finite()),
        "a node's rest transform must be finite"
    );
    ensure!(rotation.iter().any(|value| *value != 0.0), "a node's rotation must be a quaternion");
    Ok(())
}

fn multiply(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut out = [0.0; 16];
    for column in 0..4 {
        for row in 0..4 {
            out[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
        }
    }
    out
}

fn normalized(q: [f64; 4]) -> [f32; 4] {
    let length = q.iter().map(|value| value * value).sum::<f64>().sqrt();
    if length < 1e-12 || !length.is_finite() {
        return [0.0, 0.0, 0.0, 1.0];
    }
    q.map(|value| narrow(value / length))
}

/// The channel's value at `time` into `out` (its first `path.width()`
/// lanes): its first key's before it, its last's after.
fn sample(channel: &Channel, time: f64, out: &mut [f64; 4]) {
    let width = channel.path.width();
    let cubic = channel.interpolation == Interpolation::Cubic;
    let stride = if cubic { width * 3 } else { width };
    // A key's value, past a cubic key's in-tangent.
    let value =
        |key: usize, lane: usize| f64::from(channel.values[key * stride + if cubic { width } else { 0 } + lane]);
    let times = &channel.times;
    let next = times.partition_point(|&key| f64::from(key) <= time);
    if next == 0 || next == times.len() {
        let key = if next == 0 { 0 } else { times.len() - 1 };
        for (lane, slot) in out.iter_mut().enumerate().take(width) {
            *slot = value(key, lane);
        }
        return;
    }
    let key = next - 1;
    let span = f64::from(times[next]) - f64::from(times[key]);
    let s = if span > 0.0 { ((time - f64::from(times[key])) / span).clamp(0.0, 1.0) } else { 0.0 };
    match channel.interpolation {
        Interpolation::Step => {
            for (lane, slot) in out.iter_mut().enumerate().take(width) {
                *slot = value(key, lane);
            }
        }
        Interpolation::Linear if channel.path == Path::Rotation => {
            let a = [0, 1, 2, 3].map(|lane| value(key, lane));
            let b = [0, 1, 2, 3].map(|lane| value(next, lane));
            *out = slerp(a, b, s);
        }
        Interpolation::Linear => {
            for (lane, slot) in out.iter_mut().enumerate().take(width) {
                *slot = value(key, lane) + (value(next, lane) - value(key, lane)) * s;
            }
        }
        Interpolation::Cubic => {
            // Hermite, the tangents scaled by the keys' span.
            let (s2, s3) = (s * s, s * s * s);
            let out_tangent = |lane: usize| f64::from(channel.values[key * stride + 2 * width + lane]);
            let in_tangent = |lane: usize| f64::from(channel.values[next * stride + lane]);
            for (lane, slot) in out.iter_mut().enumerate().take(width) {
                *slot = (2.0 * s3 - 3.0 * s2 + 1.0) * value(key, lane)
                    + (s3 - 2.0 * s2 + s) * span * out_tangent(lane)
                    + (-2.0 * s3 + 3.0 * s2) * value(next, lane)
                    + (s3 - s2) * span * in_tangent(lane);
            }
        }
    }
}

/// The rotation `s` of the way from `a` to `b`, along the shorter arc.
fn slerp(a: [f64; 4], b: [f64; 4], s: f64) -> [f64; 4] {
    let mut cosine: f64 = (0..4).map(|lane| a[lane] * b[lane]).sum();
    let b = if cosine < 0.0 {
        cosine = -cosine;
        b.map(|value| -value)
    } else {
        b
    };
    let (from, to) = if cosine > 0.9995 {
        (1.0 - s, s)
    } else {
        let angle = cosine.clamp(-1.0, 1.0).acos();
        let sine = angle.sin();
        (((1.0 - s) * angle).sin() / sine, (s * angle).sin() / sine)
    };
    [0, 1, 2, 3].map(|lane| a[lane] * from + b[lane] * to)
}

/// The rig after a binary mesh's skin data, as `read_rig` reads it back:
/// counts as u32s, names as a u16 byte count and UTF-8, numbers as f32s,
/// all little-endian.
pub(crate) fn write_rig(rig: &Rig, out: &mut Vec<u8>) {
    let count = |out: &mut Vec<u8>, count: usize| out.extend(u32::try_from(count).unwrap_or(u32::MAX).to_le_bytes());
    let floats = |out: &mut Vec<u8>, values: &[f32]| {
        for value in values {
            out.extend(value.to_le_bytes());
        }
    };
    let name = |out: &mut Vec<u8>, name: &str| {
        let bytes = &name.as_bytes()[..name.len().min(usize::from(u16::MAX))];
        out.extend(u16::try_from(bytes.len()).unwrap_or(u16::MAX).to_le_bytes());
        out.extend(bytes);
    };
    count(out, rig.nodes.len());
    for node in &rig.nodes {
        name(out, &node.name);
        out.extend(node.parent.unwrap_or(u32::MAX).to_le_bytes());
        floats(out, &node.rest.translation);
        floats(out, &node.rest.rotation);
        floats(out, &node.rest.scale);
    }
    count(out, rig.joints.len());
    for joint in &rig.joints {
        out.extend(joint.node.to_le_bytes());
        floats(out, &joint.inverse_bind);
    }
    count(out, rig.clips.len());
    for clip in &rig.clips {
        name(out, &clip.name);
        floats(out, &[clip.duration]);
        count(out, clip.channels.len());
        for channel in &clip.channels {
            out.extend(channel.node.to_le_bytes());
            let path = match channel.path {
                Path::Translation => 0,
                Path::Rotation => 1,
                Path::Scale => 2,
            };
            let interpolation = match channel.interpolation {
                Interpolation::Linear => 0,
                Interpolation::Step => 1,
                Interpolation::Cubic => 2,
            };
            out.extend([path, interpolation, 0, 0]);
            count(out, channel.times.len());
            floats(out, &channel.times);
            floats(out, &channel.values);
        }
    }
}

/// Reads what `write_rig` wrote, all of `bytes`.
pub(crate) fn read_rig(bytes: &[u8]) -> Result<Rig> {
    let mut reader = Reader { bytes, at: 0 };
    let nodes = (0..reader.count(MAX_JOINTS)?)
        .map(|_| {
            let name = reader.name()?;
            let parent = reader.u32()?;
            let [t0, t1, t2, r0, r1, r2, r3, s0, s1, s2] = reader.floats::<10>()?;
            Ok(Node {
                name,
                parent: (parent != u32::MAX).then_some(parent),
                rest: Transform { translation: [t0, t1, t2], rotation: [r0, r1, r2, r3], scale: [s0, s1, s2] },
            })
        })
        .collect::<Result<_>>()?;
    let joints = (0..reader.count(MAX_JOINTS)?)
        .map(|_| Ok(Joint { node: reader.u32()?, inverse_bind: reader.floats::<16>()? }))
        .collect::<Result<_>>()?;
    let clips = (0..reader.count(usize::MAX)?)
        .map(|_| {
            let name = reader.name()?;
            let [duration] = reader.floats::<1>()?;
            let channels = (0..reader.count(usize::MAX)?)
                .map(|_| {
                    let node = reader.u32()?;
                    let [path, interpolation, _, _] = reader.take::<4>()?;
                    let path = match path {
                        0 => Path::Translation,
                        1 => Path::Rotation,
                        2 => Path::Scale,
                        _ => bail!("unknown channel path in binary mesh"),
                    };
                    let interpolation = match interpolation {
                        0 => Interpolation::Linear,
                        1 => Interpolation::Step,
                        2 => Interpolation::Cubic,
                        _ => bail!("unknown channel interpolation in binary mesh"),
                    };
                    let keys = reader.count(usize::MAX)?;
                    let per_key = path.width() * if interpolation == Interpolation::Cubic { 3 } else { 1 };
                    let times = reader.vector(keys)?;
                    let values = reader.vector(keys.saturating_mul(per_key))?;
                    Ok(Channel { node, path, interpolation, times, values })
                })
                .collect::<Result<_>>()?;
            Ok(Clip { name, duration, channels })
        })
        .collect::<Result<_>>()?;
    ensure!(reader.at == bytes.len(), "binary mesh has bytes after its rig");
    let rig = Rig { nodes, joints, clips };
    rig.validate()?;
    Ok(rig)
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self.at.checked_add(N).filter(|&end| end <= self.bytes.len());
        let Some(end) = end else { bail!("binary mesh ends in its rig") };
        let taken = self.bytes[self.at..end].try_into().unwrap();
        self.at = end;
        Ok(taken)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take::<4>()?))
    }

    /// A count, which no more than `most` and the bytes left can hold.
    fn count(&mut self, most: usize) -> Result<usize> {
        let count = self.u32()? as usize;
        ensure!(count <= most && count <= self.bytes.len() - self.at, "binary mesh rig count is out of range");
        Ok(count)
    }

    fn floats<const N: usize>(&mut self) -> Result<[f32; N]> {
        let mut values = [0.0; N];
        for value in &mut values {
            *value = f32::from_le_bytes(self.take::<4>()?);
        }
        Ok(values)
    }

    fn vector(&mut self, count: usize) -> Result<Vec<f32>> {
        let end =
            count.checked_mul(4).and_then(|size| self.at.checked_add(size)).filter(|&end| end <= self.bytes.len());
        let Some(end) = end else { bail!("binary mesh ends in its rig") };
        let values =
            self.bytes[self.at..end].as_chunks::<4>().0.iter().map(|&value| f32::from_le_bytes(value)).collect();
        self.at = end;
        Ok(values)
    }

    fn name(&mut self) -> Result<String> {
        let length = usize::from(u16::from_le_bytes(self.take::<2>()?));
        let end = self.at.checked_add(length).filter(|&end| end <= self.bytes.len());
        let Some(end) = end else { bail!("binary mesh ends in its rig") };
        let name = std::str::from_utf8(&self.bytes[self.at..end])?.to_owned();
        self.at = end;
        Ok(name)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Two joints in a row along y, the second turning about z over a
    /// one-second clip; and a second clip that lifts the root.
    pub(crate) fn arm() -> Rig {
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let rest = Transform::IDENTITY;
        let inverse = |y: f32| [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -y, 0.0, 1.0];
        Rig {
            nodes: vec![
                Node { name: "root".into(), parent: None, rest },
                Node {
                    name: "elbow".into(),
                    parent: Some(0),
                    rest: Transform { translation: [0.0, 1.0, 0.0], ..rest },
                },
            ],
            joints: vec![Joint { node: 0, inverse_bind: inverse(0.0) }, Joint { node: 1, inverse_bind: inverse(1.0) }],
            clips: vec![
                Clip {
                    name: "bend".into(),
                    duration: 1.0,
                    channels: vec![Channel {
                        node: 1,
                        path: Path::Rotation,
                        interpolation: Interpolation::Linear,
                        times: vec![0.0, 1.0],
                        values: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, half, half],
                    }],
                },
                Clip {
                    name: "lift".into(),
                    duration: 2.0,
                    channels: vec![Channel {
                        node: 0,
                        path: Path::Translation,
                        interpolation: Interpolation::Step,
                        times: vec![0.0, 1.0],
                        values: vec![0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                    }],
                },
            ],
        }
    }

    fn moved(matrix: &Matrix, point: [f32; 3]) -> [f32; 3] {
        let [x, y, z] = point;
        [0, 1, 2].map(|row| matrix[row] * x + matrix[4 + row] * y + matrix[8 + row] * z + matrix[12 + row])
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-5)
    }

    #[test]
    fn clips_move_the_joints_they_name() {
        let rig = arm();
        rig.validate().unwrap();
        let mut palette = Vec::new();
        // At rest every joint matrix is the identity: the mesh is bound there.
        rig.palette(&[], &mut palette);
        assert!(palette.iter().all(|matrix| close(moved(matrix, [0.3, 1.5, 0.2]), [0.3, 1.5, 0.2])));
        // Halfway through the bend the elbow has turned 45 degrees about z:
        // a point a unit above it swings toward -x.
        rig.palette(&[Layer { clip: 0, time: 0.5, weight: 1.0 }], &mut palette);
        let tip = moved(&palette[1], [0.0, 2.0, 0.0]);
        let quarter = std::f32::consts::FRAC_PI_8 * 2.0;
        assert!(close(tip, [-quarter.sin(), 1.0 + quarter.cos(), 0.0]), "{tip:?}");
        // Time wraps round the clip, and the root's joint stays put.
        rig.palette(&[Layer { clip: 0, time: 2.5, weight: 1.0 }], &mut palette);
        assert!(close(moved(&palette[1], [0.0, 2.0, 0.0]), tip));
        assert!(close(moved(&palette[0], [0.0, 0.5, 0.0]), [0.0, 0.5, 0.0]));
        // Step keys hold until the next: the lift is 0 until a second in.
        rig.palette(&[Layer { clip: 1, time: 0.99, weight: 1.0 }], &mut palette);
        assert!(close(moved(&palette[0], [0.0; 3]), [0.0; 3]));
        rig.palette(&[Layer { clip: 1, time: 1.0, weight: 1.0 }], &mut palette);
        assert!(close(moved(&palette[0], [0.0; 3]), [0.0, 2.0, 0.0]));
        // A clip the rig lacks, or a layer of no weight, leaves the rest pose.
        rig.palette(
            &[Layer { clip: 7, time: 0.5, weight: 1.0 }, Layer { clip: 0, time: 0.5, weight: 0.0 }],
            &mut palette,
        );
        assert!(close(moved(&palette[1], [0.0, 2.0, 0.0]), [0.0, 2.0, 0.0]));
    }

    #[test]
    fn layers_blend_by_their_weights() {
        let rig = arm();
        let mut palette = Vec::new();
        // The bend at its end (90 degrees) blended evenly with the rest pose
        // (the bend at 0) turns the elbow halfway: 45 degrees.
        rig.palette(
            &[Layer { clip: 0, time: 0.0, weight: 1.0 }, Layer { clip: 0, time: 0.999_999, weight: 1.0 }],
            &mut palette,
        );
        let tip = moved(&palette[1], [0.0, 2.0, 0.0]);
        let quarter = std::f32::consts::FRAC_PI_4;
        assert!((tip[0] + quarter.sin()).abs() < 1e-3 && (tip[1] - 1.0 - quarter.cos()).abs() < 1e-3, "{tip:?}");
        // Translations blend linearly: a quarter of the lift.
        rig.palette(
            &[Layer { clip: 1, time: 0.0, weight: 3.0 }, Layer { clip: 1, time: 1.5, weight: 1.0 }],
            &mut palette,
        );
        assert!(close(moved(&palette[0], [0.0; 3]), [0.0, 0.5, 0.0]));
    }

    #[test]
    fn cubic_keys_pass_through_their_values() {
        let channel = Channel {
            node: 0,
            path: Path::Translation,
            interpolation: Interpolation::Cubic,
            times: vec![0.0, 2.0],
            // In-tangent, value, out-tangent per key; flat tangents.
            values: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0],
        };
        let mut out = [0.0; 4];
        sample(&channel, 0.0, &mut out);
        assert_eq!(out[..3], [1.0; 3]);
        sample(&channel, 1.0, &mut out);
        assert!((out[0] - 2.0).abs() < 1e-9, "the middle of a flat-tangent spline: {out:?}");
        sample(&channel, 5.0, &mut out);
        assert_eq!(out[..3], [3.0; 3]);
    }

    #[test]
    fn rigs_read_back_as_written_and_reject_what_is_not_whole() {
        let rig = arm();
        let mut bytes = Vec::new();
        write_rig(&rig, &mut bytes);
        assert_eq!(read_rig(&bytes).unwrap(), rig);
        assert!(read_rig(&bytes[..bytes.len() - 1]).is_err());
        let json = serde_json::to_string(&rig.json()).unwrap();
        assert_eq!(Rig::from_json(serde_json::from_str(&json).unwrap()).unwrap(), rig);

        let mut broken = rig.clone();
        broken.nodes[0].parent = Some(1);
        assert!(broken.validate().is_err(), "a parent after its child");
        let mut broken = rig.clone();
        broken.joints[1].node = 2;
        assert!(broken.validate().is_err(), "a joint past the skeleton");
        let mut broken = rig.clone();
        broken.clips[0].channels[0].times = vec![1.0, 0.0];
        assert!(broken.validate().is_err(), "falling key times");
        let mut broken = rig;
        broken.clips[0].channels[0].values.pop();
        assert!(broken.validate().is_err(), "a value short");
    }
}
