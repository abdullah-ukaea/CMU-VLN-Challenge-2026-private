#!/usr/bin/env python3
"""Fetch only small metadata members from the official VLA-3D Unity ZIP."""

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import struct
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zlib


ARCHIVE_URL = (
    'https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/'
    'AUTH_ac8533a83cff4d48bc8c608ad222d330/vla/Unity.zip'
)
EOCD_SIGNATURE = b'PK\x05\x06'
CENTRAL_SIGNATURE = b'PK\x01\x02'
LOCAL_SIGNATURE = b'PK\x03\x04'
METADATA_SUFFIXES = (
    '_object_result.csv',
    '_region_result.csv',
    '_scene_graph.json',
)


@dataclass(frozen=True)
class RemoteZipEntry:
    """Central-directory fields needed for safe range extraction."""

    name: str
    compression: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int


class DownloadError(RuntimeError):
    """Raised when ranged metadata retrieval fails validation."""


def _request(request: Request):
    try:
        return urlopen(request, timeout=60.0)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise DownloadError(f'archive request failed: {exc}') from exc


def _archive_size(url: str) -> int:
    request = Request(url, method='HEAD', headers={'User-Agent': 'Q-MapNav/0.1'})
    with _request(request) as response:
        raw_size = response.headers.get('Content-Length')
    if raw_size is None:
        raise DownloadError('archive HEAD response has no Content-Length')
    try:
        size = int(raw_size)
    except ValueError as exc:
        raise DownloadError('archive Content-Length is not an integer') from exc
    if size <= 0:
        raise DownloadError('archive Content-Length must be positive')
    return size


def _read_range(url: str, start: int, end: int) -> bytes:
    if start < 0 or end < start:
        raise ValueError('invalid byte range')
    request = Request(
        url,
        headers={
            'Range': f'bytes={start}-{end}',
            'User-Agent': 'Q-MapNav/0.1',
        },
    )
    with _request(request) as response:
        status = getattr(response, 'status', None)
        content_range = response.headers.get('Content-Range', '')
        data = response.read()
    if status != 206:
        raise DownloadError(
            f'server ignored byte range {start}-{end}; HTTP status was {status}'
        )
    expected_prefix = f'bytes {start}-{end}/'
    if not content_range.startswith(expected_prefix):
        raise DownloadError(f'unexpected Content-Range: {content_range!r}')
    expected_size = end - start + 1
    if len(data) != expected_size:
        raise DownloadError(
            f'range {start}-{end} returned {len(data)} bytes, '
            f'expected {expected_size}'
        )
    return data


def _central_directory(url: str, archive_size: int) -> list[RemoteZipEntry]:
    tail_size = min(archive_size, 65557)
    tail_offset = archive_size - tail_size
    tail = _read_range(url, tail_offset, archive_size - 1)
    eocd_offset = tail.rfind(EOCD_SIGNATURE)
    if eocd_offset < 0:
        raise DownloadError('ZIP end-of-central-directory record was not found')
    values = struct.unpack_from('<4s4H2LH', tail, eocd_offset)
    _, disk, central_disk, disk_entries, total_entries, size, offset, comment = values
    if disk != 0 or central_disk != 0 or disk_entries != total_entries:
        raise DownloadError('multi-disk ZIP archives are unsupported')
    if comment != len(tail) - eocd_offset - 22:
        raise DownloadError('ZIP comment length is inconsistent')
    central = _read_range(url, offset, offset + size - 1)

    entries: list[RemoteZipEntry] = []
    position = 0
    for _ in range(total_entries):
        if central[position:position + 4] != CENTRAL_SIGNATURE:
            raise DownloadError('invalid central-directory entry signature')
        fields = struct.unpack_from('<4s6H3L5H2L', central, position)
        compression = fields[4]
        crc32_value = fields[7]
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        name_length = fields[10]
        extra_length = fields[11]
        comment_length = fields[12]
        local_offset = fields[16]
        name_start = position + 46
        name_end = name_start + name_length
        try:
            name = central[name_start:name_end].decode('utf-8')
        except UnicodeDecodeError as exc:
            raise DownloadError('ZIP member name is not UTF-8') from exc
        entries.append(
            RemoteZipEntry(
                name=name,
                compression=compression,
                crc32=crc32_value,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_offset=local_offset,
            )
        )
        position = name_end + extra_length + comment_length
    if position != len(central):
        raise DownloadError('central-directory size is inconsistent')
    return entries


def _safe_destination(root: Path, member_name: str) -> Path:
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or '..' in relative.parts:
        raise DownloadError(f'unsafe ZIP member path: {member_name!r}')
    if not relative.parts or relative.parts[0] != 'Unity':
        raise DownloadError(f'unexpected ZIP member root: {member_name!r}')
    destination = root.joinpath(*relative.parts)
    resolved_root = root.resolve()
    if resolved_root not in destination.resolve().parents:
        raise DownloadError(f'ZIP member escapes output root: {member_name!r}')
    return destination


def _extract_member(
    url: str,
    entry: RemoteZipEntry,
    next_offset: int,
) -> bytes:
    member = _read_range(url, entry.local_offset, next_offset - 1)
    if member[:4] != LOCAL_SIGNATURE:
        raise DownloadError(f'{entry.name}: invalid local-header signature')
    fields = struct.unpack_from('<4s5H3L2H', member, 0)
    compression = fields[3]
    name_length = fields[9]
    extra_length = fields[10]
    data_start = 30 + name_length + extra_length
    data_end = data_start + entry.compressed_size
    compressed = member[data_start:data_end]
    if len(compressed) != entry.compressed_size:
        raise DownloadError(f'{entry.name}: compressed data is truncated')
    if compression != entry.compression:
        raise DownloadError(f'{entry.name}: compression method mismatch')
    if compression == 0:
        data = compressed
    elif compression == 8:
        try:
            data = zlib.decompress(compressed, -15)
        except zlib.error as exc:
            raise DownloadError(f'{entry.name}: deflate failed: {exc}') from exc
    else:
        raise DownloadError(
            f'{entry.name}: unsupported ZIP compression method {compression}'
        )
    if len(data) != entry.uncompressed_size:
        raise DownloadError(f'{entry.name}: uncompressed size mismatch')
    if zlib.crc32(data) & 0xFFFFFFFF != entry.crc32:
        raise DownloadError(f'{entry.name}: CRC-32 mismatch')
    return data


def download_metadata(
    output_root: Path,
    *,
    url: str = ARCHIVE_URL,
    force: bool = False,
) -> tuple[Path, ...]:
    """Download and verify only object, region, and relation metadata."""
    archive_size = _archive_size(url)
    entries = _central_directory(url, archive_size)
    offsets = sorted({entry.local_offset for entry in entries} | {archive_size})
    next_offsets = {
        offset: offsets[index + 1]
        for index, offset in enumerate(offsets[:-1])
    }
    selected = [
        entry
        for entry in entries
        if entry.name.endswith(METADATA_SUFFIXES)
    ]
    if len(selected) != 45:
        raise DownloadError(
            f'expected 45 Unity metadata files, found {len(selected)}'
        )

    written: list[Path] = []
    for index, entry in enumerate(selected, start=1):
        destination = _safe_destination(Path(output_root), entry.name)
        if destination.is_file() and not force:
            print(f'[{index:02d}/{len(selected)}] exists {destination}')
            written.append(destination)
            continue
        data = _extract_member(url, entry, next_offsets[entry.local_offset])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        print(
            f'[{index:02d}/{len(selected)}] wrote {destination} '
            f'({len(data)} bytes)'
        )
        written.append(destination)
    return tuple(written)


def main() -> None:
    """Run the selective VLA-3D metadata downloader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-root',
        type=Path,
        required=True,
        help='Directory under which the Unity/<scene> metadata tree is written.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Replace already present metadata after revalidating the archive.',
    )
    args = parser.parse_args()
    files = download_metadata(args.output_root, force=args.force)
    print(f'Verified {len(files)} VLA-3D metadata files.')


if __name__ == '__main__':
    main()
