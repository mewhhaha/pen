//! RFC 5389 STUN Binding Request / Response codec for UDP NAT traversal
//! (public IP:port discovery and hole-punching peer rendezvous).
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use anyhow::{Result, bail, ensure};

pub const STUN_MAGIC_COOKIE: u32 = 0x2112_A442;
const MSG_BINDING_REQUEST: u16 = 0x0001;
const MSG_BINDING_RESPONSE: u16 = 0x0101;
const ATTR_MAPPED_ADDRESS: u16 = 0x0001;
const ATTR_USERNAME: u16 = 0x0006;
const ATTR_MESSAGE_INTEGRITY: u16 = 0x0008;
const ATTR_ERROR_CODE: u16 = 0x0009;
const ATTR_UNKNOWN_ATTRIBUTES: u16 = 0x000A;
const ATTR_REALM: u16 = 0x0014;
const ATTR_NONCE: u16 = 0x0015;
const ATTR_XOR_MAPPED_ADDRESS: u16 = 0x0020;
const FAMILY_IPV4: u8 = 0x01;
const FAMILY_IPV6: u8 = 0x02;

/// 96-bit STUN transaction identifier.
pub type TransactionId = [u8; 12];

/// Builds a 20-byte RFC 5389 STUN Binding Request packet.
pub fn build_binding_request(tx_id: TransactionId) -> [u8; 20] {
    let mut out = [0u8; 20];
    out[0..2].copy_from_slice(&MSG_BINDING_REQUEST.to_be_bytes());
    out[2..4].copy_from_slice(&0u16.to_be_bytes());
    out[4..8].copy_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes());
    out[8..20].copy_from_slice(&tx_id);
    out
}

/// Builds an RFC 5389 STUN Binding Success Response carrying an
/// `XOR-MAPPED-ADDRESS` attribute for `mapped`.
pub fn build_binding_response(tx_id: TransactionId, mapped: SocketAddr) -> Vec<u8> {
    let mut attr = Vec::with_capacity(24);
    let xport = mapped.port() ^ u16::try_from(STUN_MAGIC_COOKIE >> 16).unwrap_or(0);
    match mapped.ip() {
        IpAddr::V4(ip) => {
            attr.extend_from_slice(&ATTR_XOR_MAPPED_ADDRESS.to_be_bytes());
            attr.extend_from_slice(&8u16.to_be_bytes());
            attr.push(0);
            attr.push(FAMILY_IPV4);
            attr.extend_from_slice(&xport.to_be_bytes());
            let xip = u32::from_be_bytes(ip.octets()) ^ STUN_MAGIC_COOKIE;
            attr.extend_from_slice(&xip.to_be_bytes());
        }
        IpAddr::V6(ip) => {
            attr.extend_from_slice(&ATTR_XOR_MAPPED_ADDRESS.to_be_bytes());
            attr.extend_from_slice(&20u16.to_be_bytes());
            attr.push(0);
            attr.push(FAMILY_IPV6);
            attr.extend_from_slice(&xport.to_be_bytes());
            let mut mask = [0u8; 16];
            mask[0..4].copy_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes());
            mask[4..16].copy_from_slice(&tx_id);
            let mut xoctets = ip.octets();
            for (b, m) in xoctets.iter_mut().zip(mask) {
                *b ^= m;
            }
            attr.extend_from_slice(&xoctets);
        }
    }
    let attr_len = u16::try_from(attr.len()).unwrap_or(0);
    let mut out = Vec::with_capacity(20 + attr.len());
    out.extend_from_slice(&MSG_BINDING_RESPONSE.to_be_bytes());
    out.extend_from_slice(&attr_len.to_be_bytes());
    out.extend_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes());
    out.extend_from_slice(&tx_id);
    out.extend_from_slice(&attr);
    out
}

fn validate_header(bytes: &[u8], expected_type: u16) -> Result<(usize, TransactionId)> {
    ensure!(bytes.len() >= 20, "STUN message truncated");
    ensure!(bytes[0] & 0xC0 == 0, "STUN leading two bits must be zero");
    let msg_type = u16::from_be_bytes([bytes[0], bytes[1]]);
    ensure!(msg_type == expected_type, "unexpected STUN message type {msg_type:#06x}");
    let msg_len = usize::from(u16::from_be_bytes([bytes[2], bytes[3]]));
    ensure!(msg_len.is_multiple_of(4), "STUN message length {msg_len} is not 4-byte aligned");
    ensure!(20 + msg_len == bytes.len(), "STUN datagram length mismatch (expected {}, got {})", 20 + msg_len, bytes.len());
    let cookie = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    ensure!(cookie == STUN_MAGIC_COOKIE, "invalid STUN magic cookie");
    let mut tx = [0u8; 12];
    tx.copy_from_slice(&bytes[8..20]);
    Ok((20 + msg_len, tx))
}

fn is_known_comprehension_required(attr_type: u16) -> bool {
    matches!(
        attr_type,
        ATTR_MAPPED_ADDRESS
            | ATTR_USERNAME
            | ATTR_MESSAGE_INTEGRITY
            | ATTR_ERROR_CODE
            | ATTR_UNKNOWN_ATTRIBUTES
            | ATTR_REALM
            | ATTR_NONCE
            | ATTR_XOR_MAPPED_ADDRESS
    )
}

/// Validates every TLV attribute in `bytes[20..end]`, ensuring alignment,
/// bounded padding, and no unknown comprehension-required attributes (`< 0x8000`).
fn walk_attributes(bytes: &[u8], end: usize) -> Result<Vec<(u16, &[u8])>> {
    let mut pos = 20usize;
    let mut attrs = Vec::new();
    while pos < end {
        ensure!(pos + 4 <= end, "STUN attribute header truncated");
        let attr_type = u16::from_be_bytes([bytes[pos], bytes[pos + 1]]);
        let attr_len = usize::from(u16::from_be_bytes([bytes[pos + 2], bytes[pos + 3]]));
        pos += 4;
        let padded = (attr_len + 3) & !3;
        ensure!(pos.checked_add(padded).is_some_and(|next| next <= end), "STUN attribute body/padding truncated");
        if attr_type < 0x8000 && !is_known_comprehension_required(attr_type) {
            bail!("unsupported comprehension-required STUN attribute {attr_type:#06x}");
        }
        attrs.push((attr_type, &bytes[pos..pos + attr_len]));
        pos += padded;
    }
    ensure!(pos == end, "STUN attribute list misaligned");
    Ok(attrs)
}

/// Parses an RFC 5389 STUN Binding Request, validating header, exact length,
/// and all TLV attributes, returning its transaction ID if valid.
pub fn parse_binding_request(bytes: &[u8]) -> Result<TransactionId> {
    let (end, tx) = validate_header(bytes, MSG_BINDING_REQUEST)?;
    let _ = walk_attributes(bytes, end)?;
    Ok(tx)
}

/// Parses an RFC 5389 STUN Binding Response, verifying `expected_tx_id`,
/// validating the full datagram and every TLV attribute before extracting
/// `XOR-MAPPED-ADDRESS` (or falling back to `MAPPED-ADDRESS`).
pub fn parse_binding_response(bytes: &[u8], expected_tx_id: TransactionId) -> Result<SocketAddr> {
    let (end, tx) = validate_header(bytes, MSG_BINDING_RESPONSE)?;
    ensure!(tx == expected_tx_id, "STUN transaction ID mismatch");
    let attrs = walk_attributes(bytes, end)?;
    let mut xor_mapped = None;
    let mut fallback_mapped = None;
    for (attr_type, body) in attrs {
        if attr_type == ATTR_XOR_MAPPED_ADDRESS {
            let addr = decode_address(body, true, expected_tx_id)?;
            if xor_mapped.is_none() {
                xor_mapped = Some(addr);
            }
        } else if attr_type == ATTR_MAPPED_ADDRESS {
            let addr = decode_address(body, false, expected_tx_id)?;
            if fallback_mapped.is_none() {
                fallback_mapped = Some(addr);
            }
        }
    }
    xor_mapped.or(fallback_mapped).ok_or_else(|| anyhow::anyhow!("STUN response contained no mapped address"))
}

fn decode_address(body: &[u8], xored: bool, tx_id: TransactionId) -> Result<SocketAddr> {
    ensure!(body.len() >= 4, "STUN address attribute too short");
    let family = body[1];
    let raw_port = u16::from_be_bytes([body[2], body[3]]);
    let port = if xored { raw_port ^ u16::try_from(STUN_MAGIC_COOKIE >> 16).unwrap_or(0) } else { raw_port };
    match family {
        FAMILY_IPV4 => {
            ensure!(body.len() == 8, "STUN IPv4 address must be exactly 8 bytes, got {}", body.len());
            let raw = u32::from_be_bytes([body[4], body[5], body[6], body[7]]);
            let ip = Ipv4Addr::from(if xored { raw ^ STUN_MAGIC_COOKIE } else { raw });
            Ok(SocketAddr::new(IpAddr::V4(ip), port))
        }
        FAMILY_IPV6 => {
            ensure!(body.len() == 20, "STUN IPv6 address must be exactly 20 bytes, got {}", body.len());
            let mut octets = [0u8; 16];
            octets.copy_from_slice(&body[4..20]);
            if xored {
                let mut mask = [0u8; 16];
                mask[0..4].copy_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes());
                mask[4..16].copy_from_slice(&tx_id);
                for (b, m) in octets.iter_mut().zip(mask) {
                    *b ^= m;
                }
            }
            Ok(SocketAddr::new(IpAddr::V6(Ipv6Addr::from(octets)), port))
        }
        other => bail!("unsupported STUN address family {other:#04x}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stun_ipv4_and_ipv6_xor_mapped_round_trip() {
        let tx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let req = build_binding_request(tx);
        assert_eq!(parse_binding_request(&req).unwrap(), tx);

        let v4: SocketAddr = "203.0.113.42:54321".parse().unwrap();
        let resp4 = build_binding_response(tx, v4);
        assert_eq!(parse_binding_response(&resp4, tx).unwrap(), v4);

        let v6: SocketAddr = "[2001:db8::cafe:1234]:61234".parse().unwrap();
        let resp6 = build_binding_response(tx, v6);
        assert_eq!(parse_binding_response(&resp6, tx).unwrap(), v6);

        assert!(parse_binding_response(&resp4, [0u8; 12]).is_err());
    }

    #[test]
    fn stun_rejects_malformed_lengths_trailing_bytes_and_unknown_required_attrs() {
        let tx = [7u8; 12];
        let v4: SocketAddr = "198.51.100.10:4000".parse().unwrap();

        // Trailing bytes rejected on both request and response.
        let mut req_extra = build_binding_request(tx).to_vec();
        req_extra.extend_from_slice(&[0, 0, 0, 0]);
        assert!(parse_binding_request(&req_extra).is_err());

        let mut resp_extra = build_binding_response(tx, v4);
        resp_extra.push(0xAA);
        assert!(parse_binding_response(&resp_extra, tx).is_err());

        // Unaligned message length rejected.
        let mut unaligned = build_binding_response(tx, v4);
        let bad_len = u16::try_from(unaligned.len() - 20 + 1).unwrap();
        unaligned[2..4].copy_from_slice(&bad_len.to_be_bytes());
        unaligned.push(0);
        assert!(parse_binding_response(&unaligned, tx).is_err());

        // Unknown comprehension-required attribute after XOR-MAPPED-ADDRESS must be rejected.
        let mut with_unknown_req = build_binding_response(tx, v4);
        with_unknown_req.extend_from_slice(&0x0042u16.to_be_bytes()); // < 0x8000
        with_unknown_req.extend_from_slice(&4u16.to_be_bytes());
        with_unknown_req.extend_from_slice(&[1, 2, 3, 4]);
        let new_len = u16::try_from(with_unknown_req.len() - 20).unwrap();
        with_unknown_req[2..4].copy_from_slice(&new_len.to_be_bytes());
        assert!(parse_binding_response(&with_unknown_req, tx).is_err());

        // Unknown comprehension-optional attribute (>= 0x8000, e.g. SOFTWARE 0x8022) with padding succeeds.
        let mut with_opt = build_binding_response(tx, v4);
        with_opt.extend_from_slice(&0x8022u16.to_be_bytes());
        with_opt.extend_from_slice(&3u16.to_be_bytes());
        with_opt.extend_from_slice(b"pen\0"); // 3 bytes + 1 padding byte
        let opt_len = u16::try_from(with_opt.len() - 20).unwrap();
        with_opt[2..4].copy_from_slice(&opt_len.to_be_bytes());
        assert_eq!(parse_binding_response(&with_opt, tx).unwrap(), v4);

        // Oversized IPv4 address attribute (12 bytes instead of 8) rejected.
        let mut bad_v4 = build_binding_response(tx, v4);
        bad_v4[22..24].copy_from_slice(&12u16.to_be_bytes());
        bad_v4.extend_from_slice(&[0, 0, 0, 0]);
        let bad_v4_len = u16::try_from(bad_v4.len() - 20).unwrap();
        bad_v4[2..4].copy_from_slice(&bad_v4_len.to_be_bytes());
        assert!(parse_binding_response(&bad_v4, tx).is_err());
    }
}
