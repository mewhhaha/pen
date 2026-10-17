#!/usr/bin/env bash
# Signs Windows executables with Authenticode, when there is a certificate,
# and checks each signature; without one it says so and changes nothing, so
# that unsigned builds go on.
#
#   scripts/sign-windows.sh EXE...
#
# WINDOWS_CERTIFICATE           a PKCS#12 (.pfx) file, or its base64 (as a CI
#                               secret holds it)
# WINDOWS_CERTIFICATE_PASSWORD  its password
# WINDOWS_TIMESTAMP_URL         an RFC 3161 time-stamping server (default
#                               http://timestamp.digicert.com), so that the
#                               signature outlives the certificate; "none"
#                               signs without one
# WINDOWS_CERTIFICATE_CA        the certificates to check the signature
#                               against, when not the system's (a test
#                               certificate's own)
#
# Needs osslsigncode (Linux: apt install osslsigncode), which keeps the
# password out of its command line (-readpass).
set -euo pipefail

if [ $# -eq 0 ]; then
  echo "usage: $0 EXE..." >&2
  exit 2
fi

say() {
  echo "sign-windows: $1"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then echo "- sign-windows: $1" >> "$GITHUB_STEP_SUMMARY"; fi
}

if [ -z "${WINDOWS_CERTIFICATE:-}" ]; then
  say "not signing $*: no certificate (WINDOWS_CERTIFICATE)"
  exit 0
fi
command -v osslsigncode > /dev/null || { say "FAILED: osslsigncode is not on PATH" >&2; exit 1; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
chmod 700 "$work"
if [ -f "$WINDOWS_CERTIFICATE" ]; then
  certificate=$WINDOWS_CERTIFICATE
else
  certificate=$work/certificate.pfx
  printf '%s' "$WINDOWS_CERTIFICATE" | base64 -d > "$certificate" \
    || { say "FAILED: WINDOWS_CERTIFICATE is neither a file nor base64" >&2; exit 1; }
fi
password=$work/password
printf '%s' "${WINDOWS_CERTIFICATE_PASSWORD:-}" > "$password"
timestamp=${WINDOWS_TIMESTAMP_URL:-http://timestamp.digicert.com}

for exe in "$@"; do
  name=$(basename "$exe" .exe)
  options=(sign -pkcs12 "$certificate" -readpass "$password" -h sha256 -n "$name")
  if [ "$timestamp" != none ]; then options+=(-ts "$timestamp"); fi
  osslsigncode "${options[@]}" -in "$exe" -out "$work/signed.exe" > "$work/sign.log" 2>&1 \
    || { cat "$work/sign.log" >&2; say "FAILED: signing $exe" >&2; exit 1; }
  # Its permissions kept, the signed file replaces it.
  cat "$work/signed.exe" > "$exe"
  verify=(verify -in "$exe")
  if [ -n "${WINDOWS_CERTIFICATE_CA:-}" ]; then verify+=(-CAfile "$WINDOWS_CERTIFICATE_CA"); fi
  osslsigncode "${verify[@]}" > "$work/verify.log" 2>&1 \
    || { cat "$work/verify.log" >&2; say "FAILED: the signature of $exe does not verify" >&2; exit 1; }
  say "signed $exe ($(grep -m1 'Subject:' "$work/verify.log" | sed 's/^[[:space:]]*//'))"
done
