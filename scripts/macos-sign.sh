#!/usr/bin/env bash
# Signs the macOS application bundle `pen build` made, with the hardened
# runtime and packaging/macos/entitlements.plist, and notarizes and staples
# it when the owner's credentials are set. Without a certificate it signs ad
# hoc: the bundle then runs on the Mac that signed it, and Gatekeeper
# refuses it anywhere else.
#
#   scripts/macos-sign.sh APP
#
# MACOS_CERTIFICATE           a Developer ID Application certificate with
#                             its private key, .p12, base64
# MACOS_CERTIFICATE_PASSWORD  its password
# MACOS_SIGNING_IDENTITY      the identity in it, "Developer ID Application:
#                             NAME (TEAM)"
# MACOS_NOTARY_KEY            an App Store Connect API key, .p8, base64, with
# MACOS_NOTARY_KEY_ID         its key ID and
# MACOS_NOTARY_ISSUER         its issuer ID; without them it signs only
set -euo pipefail

if [ $# -ne 1 ] || [ ! -d "$1/Contents/MacOS" ]; then
  echo "usage: $0 APP (an application bundle pen build made)" >&2
  exit 2
fi
app=$1
entitlements="$(cd "$(dirname "$0")/.." && pwd)/packaging/macos/entitlements.plist"
name=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$app/Contents/Info.plist")

if [ -z "${MACOS_CERTIFICATE:-}" ]; then
  codesign --force --deep --sign - "$app"
  codesign --verify --deep --strict --verbose=2 "$app"
  echo "macos-sign: signed $app ad hoc (no MACOS_CERTIFICATE): it runs on this Mac only"
  exit 0
fi
: "${MACOS_CERTIFICATE_PASSWORD:?the certificate needs MACOS_CERTIFICATE_PASSWORD}"
: "${MACOS_SIGNING_IDENTITY:?the certificate needs MACOS_SIGNING_IDENTITY}"

# A keychain of its own for the certificate, gone when this ends.
work=$(mktemp -d)
keychain=$work/signing.keychain-db
password=$(uuidgen)
previous=$(security list-keychains -d user | tr -d '"' | xargs)
cleanup() {
  # shellcheck disable=SC2086 # the user's keychains, one word each
  security list-keychains -d user -s $previous || true
  security delete-keychain "$keychain" 2> /dev/null || true
  rm -rf "$work"
}
trap cleanup EXIT
security create-keychain -p "$password" "$keychain"
security set-keychain-settings -lut 21600 "$keychain"
security unlock-keychain -p "$password" "$keychain"
printf '%s' "$MACOS_CERTIFICATE" | base64 --decode > "$work/certificate.p12"
security import "$work/certificate.p12" -k "$keychain" -P "$MACOS_CERTIFICATE_PASSWORD" -T /usr/bin/codesign
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$password" "$keychain" > /dev/null
# shellcheck disable=SC2086 # the user's keychains, one word each
security list-keychains -d user -s "$keychain" $previous

# Inside out: Steam's library, the runtime with its entitlements, the bundle.
sign() {
  codesign --force --timestamp --options runtime --sign "$MACOS_SIGNING_IDENTITY" "$@"
}
for library in "$app"/Contents/MacOS/*.dylib; do
  if [ -e "$library" ]; then sign "$library"; fi
done
sign --entitlements "$entitlements" "$app/Contents/MacOS/$name"
sign --entitlements "$entitlements" "$app"
codesign --verify --deep --strict --verbose=2 "$app"
echo "macos-sign: signed $app as $MACOS_SIGNING_IDENTITY"

if [ -z "${MACOS_NOTARY_KEY:-}" ]; then
  echo "macos-sign: no MACOS_NOTARY_KEY, so not notarized: Gatekeeper warns on other Macs"
  exit 0
fi
: "${MACOS_NOTARY_KEY_ID:?notarizing needs MACOS_NOTARY_KEY_ID}"
: "${MACOS_NOTARY_ISSUER:?notarizing needs MACOS_NOTARY_ISSUER}"
printf '%s' "$MACOS_NOTARY_KEY" | base64 --decode > "$work/notary.p8"
ditto -c -k --keepParent "$app" "$work/notarize.zip"
xcrun notarytool submit "$work/notarize.zip" --key "$work/notary.p8" --key-id "$MACOS_NOTARY_KEY_ID" \
  --issuer "$MACOS_NOTARY_ISSUER" --wait --timeout 30m
xcrun stapler staple "$app"
spctl --assess --type execute --verbose=2 "$app"
echo "macos-sign: notarized and stapled $app"
