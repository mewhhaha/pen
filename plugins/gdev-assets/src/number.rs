//! Numbers as JavaScript's `String(number)` prints them, as assets.ts wrote
//! F32 values and gdev's tunables.ts its messages: the shortest digits that
//! read back as the same f64, in plain notation from 1e-6 up to 1e21 and in
//! exponent notation (`1e-7`, `1.5e+21`) beyond.

pub(crate) fn javascript(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    if value < 0.0 {
        return format!("-{}", javascript(-value));
    }
    if value.is_infinite() {
        return "Infinity".to_string();
    }
    // Rust's exponent form has the same shortest digits ("1.2345e-7"), but
    // for a value exactly halfway between two, where JavaScript takes the
    // even one.
    let (mut digits, exponent) = digits_of(&format!("{value:e}"));
    if let Some(even) = even_neighbour(value, &digits, exponent) {
        digits = even;
    }
    // The number is 0.DIGITS × 10^point (ECMAScript's n).
    let point = exponent + 1;
    let count = digits.len() as i32;
    if count <= point && point <= 21 {
        format!("{digits}{}", "0".repeat((point - count) as usize))
    } else if 0 < point && point <= 21 {
        format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
    } else if -6 < point && point <= 0 {
        format!("0.{}{digits}", "0".repeat(-point as usize))
    } else {
        let exponent = point - 1;
        let sign = if exponent < 0 { '-' } else { '+' };
        let (first, rest) = digits.split_at(1);
        let fraction = if rest.is_empty() { String::new() } else { format!(".{rest}") };
        format!("{first}{fraction}e{sign}{}", exponent.abs())
    }
}

/// The significant digits and decimal exponent of Rust's exponent form.
fn digits_of(exponent_form: &str) -> (String, i32) {
    let (mantissa, exponent) = exponent_form.split_once('e').expect("an exponent");
    let digits = mantissa.chars().filter(char::is_ascii_digit).collect();
    (digits, exponent.parse().expect("a decimal exponent"))
}

/// The other of the two shortest `digits` (positive `value`'s, at
/// `exponent`) when `value` is exactly halfway between them, its digits one
/// longer and ending in 5, and `digits` ends in an odd digit.
fn even_neighbour(value: f64, digits: &str, exponent: i32) -> Option<String> {
    if digits.bytes().last().is_some_and(|last| (last - b'0').is_multiple_of(2)) {
        return None;
    }
    // Every f64 has at most 767 significant digits, so this is exact.
    let (exact, exact_exponent) = digits_of(&format!("{value:.800e}"));
    let exact = exact.trim_end_matches('0');
    if exact_exponent != exponent || exact.len() != digits.len() + 1 || !exact.ends_with('5') {
        return None;
    }
    let lower = &exact[..digits.len()];
    let neighbour = if digits == lower {
        // One more in the last place; a carry would make it shorter, so not
        // the other of two.
        let (head, last) = lower.split_at(lower.len() - 1);
        let last = last.parse::<u8>().ok().filter(|last| *last < 9)?;
        format!("{head}{}", last + 1)
    } else {
        lower.to_string()
    };
    let (first, rest) = neighbour.split_at(1);
    let reads_back = format!("{first}.{rest}e{exponent}").parse::<f64>().is_ok_and(|read| read == value);
    reads_back.then_some(neighbour)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prints_as_javascript_does() {
        let cases = [
            (0.0, "0"),
            (-0.0, "0"),
            (1.0, "1"),
            (-1.5, "-1.5"),
            (0.1, "0.1"),
            (100.0, "100"),
            (1e21, "1e+21"),
            (1e20, "100000000000000000000"),
            (1.5e300, "1.5e+300"),
            (1e-6, "0.000001"),
            (1e-7, "1e-7"),
            (1.234e-7, "1.234e-7"),
            (5e-324, "5e-324"),
            (0.1f32 as f64, "0.10000000149011612"),
            // Halfway between ...1562 and ...1563: the even one.
            (3.736f32 as f64, "3.7360000610351562"),
            (f64::from_bits(0xc083_0860_a000_0000), "-609.0471801757812"),
            (f64::INFINITY, "Infinity"),
            (f64::NEG_INFINITY, "-Infinity"),
            (f64::NAN, "NaN"),
        ];
        for (value, printed) in cases {
            assert_eq!(javascript(value), printed, "{value:e}");
        }
    }

    /// Against what Deno printed for random f64s and f32s
    /// (tests/fixtures/numbers.json, from tests/fixtures/expected.ts).
    #[test]
    fn prints_as_deno_did() {
        let crate::json::Json::Array(numbers) =
            crate::json::parse(include_str!("../tests/fixtures/numbers.json")).unwrap()
        else {
            panic!("numbers.json is a list");
        };
        assert!(numbers.len() > 1000);
        for number in numbers {
            let crate::json::Json::Array(pair) = number else { panic!("{number:?} is no pair") };
            let [crate::json::Json::String(bits), crate::json::Json::String(printed)] = &pair[..] else {
                panic!("{pair:?} is no pair of strings");
            };
            let value = f64::from_bits(u64::from_str_radix(bits, 16).unwrap());
            assert_eq!(javascript(value), *printed, "{bits}");
        }
    }
}
