#!/bin/bash
set -euo pipefail

DEFAULT_LICENSE_URL="https://creativecommons.org/licenses/by-nc-sa/4.0/"
DEFAULT_ATTRIBUTION="Music Technology Group, Universitat Pompeu Fabra; Pablo Alonso-Jimenez, Xavier Serra, Dmitry Bogdanov"
MODEL_DIRECTORY_NAME="maest-10s-fp16.mlmodelc"
LABELS_FILE_NAME="labels.json"
FILTERBANK_FILE_NAME="maest_mel_filters.bin"
SOURCE_MANIFEST_NAME="model-manifest.json"

usage() {
    cat <<'EOF'
Usage:
  package-model-release.sh \
    --model-dir PATH \
    --version VERSION \
    --output DIRECTORY

The model directory must contain:
  maest-10s-fp16.mlmodelc/
  labels.json
  maest_mel_filters.bin
  model-manifest.json
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

MODEL_DIR=""
VERSION=""
OUTPUT=""

while (( $# > 0 )); do
    case "$1" in
        --model-dir)
            (( $# >= 2 )) || die "--model-dir requires a value"
            MODEL_DIR="$2"
            shift 2
            ;;
        --version)
            (( $# >= 2 )) || die "--version requires a value"
            VERSION="$2"
            shift 2
            ;;
        --output)
            (( $# >= 2 )) || die "--output requires a value"
            OUTPUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "$MODEL_DIR" ]] || die "--model-dir is required"
[[ -n "$VERSION" ]] || die "--version is required"
[[ -n "$OUTPUT" ]] || die "--output is required"
[[ "$VERSION" =~ ^[0-9]+([.][0-9]+)*(-[0-9A-Za-z.-]+)?([+][0-9A-Za-z.-]+)?$ ]] \
    || die "invalid version: $VERSION"
[[ -d "$MODEL_DIR" && ! -L "$MODEL_DIR" ]] \
    || die "model directory does not exist or is a symbolic link: $MODEL_DIR"
MODEL_DIR="$(cd "$MODEL_DIR" && pwd -P)"

for TOOL in python3 ditto zipinfo; do
    command -v "$TOOL" >/dev/null 2>&1 || die "required tool not found: $TOOL"
done

CORE_MODEL_DIR="$MODEL_DIR/$MODEL_DIRECTORY_NAME"
LABELS_FILE="$MODEL_DIR/$LABELS_FILE_NAME"
FILTERBANK_FILE="$MODEL_DIR/$FILTERBANK_FILE_NAME"
SOURCE_MANIFEST="$MODEL_DIR/$SOURCE_MANIFEST_NAME"

[[ -d "$CORE_MODEL_DIR" && ! -L "$CORE_MODEL_DIR" ]] \
    || die "missing Core ML directory: $CORE_MODEL_DIR"
[[ -f "$LABELS_FILE" && ! -L "$LABELS_FILE" && -s "$LABELS_FILE" ]] \
    || die "missing, empty, or symbolic labels file: $LABELS_FILE"
[[ -f "$FILTERBANK_FILE" && ! -L "$FILTERBANK_FILE" && -s "$FILTERBANK_FILE" ]] \
    || die "missing, empty, or symbolic filterbank file: $FILTERBANK_FILE"
[[ -f "$SOURCE_MANIFEST" && ! -L "$SOURCE_MANIFEST" && -s "$SOURCE_MANIFEST" ]] \
    || die "missing, empty, or symbolic manifest source: $SOURCE_MANIFEST"

python3 - "$MODEL_DIR" "$VERSION" "$MODEL_DIRECTORY_NAME" \
    "$LABELS_FILE_NAME" "$FILTERBANK_FILE_NAME" "$SOURCE_MANIFEST_NAME" <<'PY'
import hashlib
import json
import os
import sys

(
    model_dir,
    version,
    model_directory_name,
    labels_name,
    filterbank_name,
    source_manifest_name,
) = sys.argv[1:]


def fail(message):
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def safe_relative_path(value):
    if (
        not value
        or value.startswith("/")
        or value.startswith("~")
        or "\\" in value
        or "\0" in value
    ):
        return False
    return all(part not in ("", ".", "..") for part in value.split("/"))


def contained(path, root):
    root = os.path.realpath(root)
    path = os.path.realpath(path)
    return path == root or path.startswith(root + os.sep)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


source_manifest = os.path.join(model_dir, source_manifest_name)
try:
    with open(source_manifest, "r", encoding="utf-8") as handle:
        source = json.load(handle)
except (OSError, ValueError) as error:
    fail(f"cannot read manifest source {source_manifest}: {error}")

try:
    core_ml = source["coreML"]
    declared_version = str(source["version"])
    source_files = core_ml["files"]
    declared_directory = core_ml["directory"]
    declared_labels = core_ml["labels"]
    declared_filterbank = core_ml["filterbank"]
except (KeyError, TypeError):
    fail("manifest source is missing coreML file metadata")

if declared_version != version:
    fail(
        f"requested version {version} does not match manifest source "
        f"version {declared_version}"
    )
if declared_directory != model_directory_name:
    fail(f"manifest source declares an unexpected Core ML directory: {declared_directory}")
if declared_labels != labels_name:
    fail(f"manifest source declares unexpected labels: {declared_labels}")
if declared_filterbank != filterbank_name:
    fail(f"manifest source declares an unexpected filterbank: {declared_filterbank}")
if not isinstance(source_files, dict) or not source_files:
    fail("manifest source has no coreML.files entries")

def is_release_path(path):
    return (
        path in (labels_name, filterbank_name)
        or path.startswith(model_directory_name + "/")
    )


core_model_dir = os.path.join(model_dir, model_directory_name)
if os.path.islink(core_model_dir):
    fail(f"symbolic link in model source: {core_model_dir}")
if not os.path.isdir(core_model_dir):
    fail(f"missing Core ML directory: {core_model_dir}")

for root, directories, files in os.walk(core_model_dir, followlinks=False):
    for name in directories + files:
        path = os.path.join(root, name)
        if os.path.islink(path):
            fail(f"symbolic link in model source: {path}")
        if name in (".DS_Store", "__MACOSX", "__pycache__", ".cache") or name.startswith("._"):
            fail(f"cache or metadata entry in model source: {path}")
        if not contained(path, core_model_dir):
            fail(f"path escapes Core ML directory: {path}")
        if not os.path.isdir(path) and not os.path.isfile(path):
            fail(f"unsupported file type in model source: {path}")

for path in source_files:
    if not safe_relative_path(path):
        fail(f"unsafe path in manifest source: {path!r}")
    if not is_release_path(path):
        fail(f"manifest source references a non-release file: {path}")
    absolute = os.path.join(model_dir, *path.split("/"))
    if not contained(absolute, model_dir):
        fail(f"path escapes model directory: {path}")
    if os.path.islink(absolute):
        fail(f"symbolic link in model source: {absolute}")
    if not os.path.isfile(absolute):
        fail(f"missing model file: {path}")
    metadata = source_files[path]
    try:
        expected_size = int(metadata["size"])
        expected_hash = str(metadata["sha256"]).lower()
    except (KeyError, TypeError, ValueError):
        fail(f"invalid manifest source metadata for {path}")
    actual_size = os.path.getsize(absolute)
    if actual_size != expected_size:
        fail(
            f"size mismatch for {path}: expected {expected_size}, "
            f"got {actual_size}"
        )
    actual_hash = sha256(absolute)
    if actual_hash != expected_hash:
        fail(f"SHA-256 mismatch for {path}")
if labels_name not in source_files or filterbank_name not in source_files:
    fail("manifest source does not describe labels and filterbank files")
if not any(
    path.startswith(model_directory_name + "/")
    for path in source_files
):
    fail("manifest source does not describe Core ML files")
PY

OUTPUT_PARENT="$(dirname "$OUTPUT")"
OUTPUT_NAME="$(basename "$OUTPUT")"
[[ "$OUTPUT_NAME" != "." && "$OUTPUT_NAME" != ".." && -n "$OUTPUT_NAME" ]] \
    || die "invalid output directory: $OUTPUT"
mkdir -p "$OUTPUT_PARENT"
OUTPUT_PARENT="$(cd "$OUTPUT_PARENT" && pwd)"
OUTPUT="$OUTPUT_PARENT/$OUTPUT_NAME"
[[ ! -e "$OUTPUT" ]] \
    || die "output already exists: $OUTPUT"

WORK_ROOT="$(mktemp -d "$OUTPUT_PARENT/.lyrid-model-release.XXXXXX")"
cleanup() {
    rm -rf -- "$WORK_ROOT"
}
trap cleanup EXIT

STAGING_ROOT="$WORK_ROOT/staging"
RELEASE_ROOT="$WORK_ROOT/release"
ARCHIVE_NAME="maest-10s-fp16-$VERSION.zip"
ARCHIVE_PATH="$RELEASE_ROOT/$ARCHIVE_NAME"
MANIFEST_PATH="$RELEASE_ROOT/manifest.json"
mkdir -p "$STAGING_ROOT" "$RELEASE_ROOT"

ditto --norsrc --noextattr --noqtn \
    "$CORE_MODEL_DIR" "$STAGING_ROOT/$MODEL_DIRECTORY_NAME"
cp -p "$LABELS_FILE" "$STAGING_ROOT/$LABELS_FILE_NAME"
cp -p "$FILTERBANK_FILE" "$STAGING_ROOT/$FILTERBANK_FILE_NAME"

ditto -c -k --norsrc --noextattr --noqtn \
    "$STAGING_ROOT/" "$ARCHIVE_PATH"

python3 - "$MODEL_DIR" "$STAGING_ROOT" "$ARCHIVE_PATH" "$MANIFEST_PATH" \
    "$VERSION" "$ARCHIVE_NAME" "$MODEL_DIRECTORY_NAME" "$LABELS_FILE_NAME" \
    "$FILTERBANK_FILE_NAME" "$SOURCE_MANIFEST_NAME" \
    "$DEFAULT_LICENSE_URL" "$DEFAULT_ATTRIBUTION" <<'PY'
import hashlib
import json
import os
import stat
import sys
import zipfile

(
    model_dir,
    staging_root,
    archive_path,
    manifest_path,
    version,
    archive_name,
    model_directory_name,
    labels_name,
    filterbank_name,
    source_manifest_name,
    default_license_url,
    default_attribution,
) = sys.argv[1:]


def fail(message):
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def safe_relative_path(value):
    if (
        not value
        or value.startswith("/")
        or value.startswith("~")
        or "\\" in value
        or "\0" in value
    ):
        return False
    return all(part not in ("", ".", "..") for part in value.split("/"))


def contained(path, root):
    root = os.path.realpath(root)
    path = os.path.realpath(path)
    return path == root or path.startswith(root + os.sep)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


inventory = {}


def is_release_path(path):
    return (
        path in (labels_name, filterbank_name)
        or path.startswith(model_directory_name + "/")
    )


for root, directories, files in os.walk(staging_root, followlinks=False):
    for name in directories + files:
        path = os.path.join(root, name)
        if os.path.islink(path):
            fail(f"symbolic link in staging directory: {path}")
        if not contained(path, staging_root):
            fail(f"path escapes staging directory: {path}")
        if not os.path.isdir(path) and not os.path.isfile(path):
            fail(f"unsupported file type in staging directory: {path}")

for root, _, files in os.walk(staging_root, followlinks=False):
    for name in files:
        path = os.path.join(root, name)
        relative = os.path.relpath(path, staging_root).replace(os.sep, "/")
        if not safe_relative_path(relative):
            fail(f"unsafe staged path: {relative!r}")
        if not is_release_path(relative):
            fail(f"unexpected file in staging directory: {relative}")
        if os.path.islink(path) or not os.path.isfile(path):
            fail(f"staged entry is not a regular file: {relative}")
        inventory[relative] = {
            "size": os.path.getsize(path),
            "sha256": sha256(path),
        }

if labels_name not in inventory:
    fail(f"staging directory is missing {labels_name}")
if filterbank_name not in inventory:
    fail(f"staging directory is missing {filterbank_name}")
if not any(path.startswith(model_directory_name + "/") for path in inventory):
    fail("staging directory is missing Core ML files")

try:
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive_entries = set()
        for info in archive.infolist():
            name = info.filename
            if name.endswith("/"):
                name = name[:-1]
            if not name or not safe_relative_path(name):
                fail(f"unsafe archive entry: {info.filename!r}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                fail(f"symbolic link in archive: {info.filename}")
            if info.is_dir():
                if not (
                    name == model_directory_name
                    or name.startswith(model_directory_name + "/")
                ):
                    fail(f"unexpected directory in archive: {name}")
                continue
            if name in archive_entries:
                fail(f"duplicate archive entry: {name}")
            if name not in inventory:
                fail(f"unexpected file in archive: {name}")
            if info.file_size != inventory[name]["size"]:
                fail(f"archive size mismatch for {name}")
            archive_entries.add(name)
except (OSError, zipfile.BadZipFile) as error:
    fail(f"cannot validate archive: {error}")

if archive_entries != set(inventory):
    missing = sorted(set(inventory) - archive_entries)
    extra = sorted(archive_entries - set(inventory))
    fail(f"archive inventory mismatch; missing={missing}, extra={extra}")

with open(os.path.join(model_dir, source_manifest_name), "r", encoding="utf-8") as handle:
    source = json.load(handle)

model_key = str(source.get("model", "")).strip()
license_name = str(source.get("license", "")).strip()
source_url = str(source.get("source", "")).strip()
commercial_use = str(source.get("commercialUse", "")).strip()
if not model_key:
    fail("manifest source is missing model")
if not license_name:
    fail("manifest source is missing license")
if not source_url:
    fail("manifest source is missing source")
if not commercial_use:
    fail("manifest source is missing commercialUse")

attribution = str(source.get("attribution", "")).strip()
if not attribution:
    model_card = os.path.join(model_dir, "MODEL_CARD.md")
    if os.path.isfile(model_card) and not os.path.islink(model_card):
        with open(model_card, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("- Attribution:"):
                    attribution = stripped.split(":", 1)[1].strip()
                    break
if not attribution:
    attribution = default_attribution

license_url = str(source.get("licenseURL", "")).strip() or default_license_url
try:
    analyzer_version = int(source.get("analyzerVersion", 1))
except (TypeError, ValueError):
    fail("manifest source has an invalid analyzerVersion")
if analyzer_version <= 0:
    fail("manifest source analyzerVersion must be positive")
minimum_app_version = str(source.get("minimumAppVersion", "1.0.0"))
minimum_macos_version = str(source.get("minimumMacOSVersion", "14.0"))
provider = str(source.get("provider", "MAEST"))
display_name = str(source.get("displayName", "MAEST · Discogs 10s"))

manifest = {
    "schemaVersion": 1,
    "provider": provider,
    "modelKey": model_key,
    "version": version,
    "displayName": display_name,
    "minimumAppVersion": minimum_app_version,
    "minimumMacOSVersion": minimum_macos_version,
    "license": license_name,
    "licenseURL": license_url,
    "commercialUse": commercial_use,
    "attribution": attribution,
    "sourceURL": source_url,
    "analyzerVersion": analyzer_version,
    "archive": {
        "name": archive_name,
        "size": os.path.getsize(archive_path),
        "sha256": sha256(archive_path),
    },
    "files": {
        path: inventory[path]
        for path in sorted(inventory)
    },
}

with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
with open(manifest_path, "r", encoding="utf-8") as handle:
    json.load(handle)
PY

chmod 0644 "$ARCHIVE_PATH" "$MANIFEST_PATH"
chmod 0755 "$RELEASE_ROOT"
mv "$RELEASE_ROOT" "$OUTPUT"
trap - EXIT
cleanup

printf 'Manifest: %s\n' "$OUTPUT/manifest.json"
printf 'Archive:  %s\n' "$OUTPUT/$ARCHIVE_NAME"
