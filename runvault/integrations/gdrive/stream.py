"""Bounded streaming reader for Google Drive checkpoint artifacts."""

from __future__ import annotations

import io
import os
from typing import Any, BinaryIO


class SingleChunkWriter:
    """Bounded single-chunk memory buffer for MediaIoBaseDownload.

    MediaIoBaseDownload writes downloaded chunks via .write(content).
    By capturing only the latest chunk and clearing it once consumed,
    memory usage remains strictly bounded to O(chunk_size), preventing
    unbounded memory growth during multi-gigabyte checkpoint downloads.
    """

    def __init__(self) -> None:
        self.chunk: bytes = b""

    def write(self, b: bytes) -> int:
        self.chunk = b
        return len(b)

    def pop_chunk(self) -> bytes:
        data = self.chunk
        self.chunk = b""
        return data


class GoogleDriveDownloadStream(io.RawIOBase, BinaryIO):
    """Streaming binary reader that downloads Google Drive artifacts in bounded chunks.

    Prevents loading entire remote checkpoints into memory during verification or recovery.
    Adheres strictly to the Python BinaryIO and io.RawIOBase contracts.
    """

    def __init__(
        self,
        service: Any,
        file_id: str,
        total_size: int | None = None,
        chunk_size: int = 1024 * 1024 * 2,  # 2 MB buffer
    ) -> None:
        super().__init__()
        self._service = service
        self._file_id = file_id
        self._chunk_size = chunk_size
        self._closed = False
        self._pos = 0

        # Discover total size if not provided
        if total_size is None:
            meta = service.files().get(fileId=file_id, fields="size").execute()
            self._total_size = int(meta.get("size", 0))
        else:
            self._total_size = total_size

        self._buffer = b""
        self._buffer_offset = 0
        self._downloader: Any = None
        self._chunk_writer = SingleChunkWriter()
        self._init_downloader()

    def _init_downloader(self) -> None:
        from googleapiclient.http import MediaIoBaseDownload  # type: ignore[reportMissingImports]

        request = self._service.files().get_media(fileId=self._file_id)
        self._chunk_writer = SingleChunkWriter()
        self._downloader = MediaIoBaseDownload(
            self._chunk_writer,
            request,
            chunksize=self._chunk_size,
        )
        self._buffer = b""
        self._buffer_offset = 0

    def readable(self) -> bool:
        return not self._closed

    def seekable(self) -> bool:
        return not self._closed

    def writable(self) -> bool:
        return False

    def flush(self) -> None:
        pass

    def tell(self) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed file.")
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed file.")
        if whence == os.SEEK_SET:
            new_pos = offset
        elif whence == os.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == os.SEEK_END:
            new_pos = self._total_size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")

        new_pos = max(0, min(new_pos, self._total_size))
        if new_pos != self._pos:
            if new_pos < self._buffer_offset or new_pos > (self._buffer_offset + len(self._buffer)):
                self._init_downloader()
                to_skip = new_pos
                while to_skip > 0 and not getattr(self._downloader, "_done", True):
                    chunk = self._fetch_next_chunk()
                    if not chunk:
                        break
                    if len(chunk) <= to_skip:
                        to_skip -= len(chunk)
                    else:
                        self._buffer = chunk[to_skip:]
                        self._buffer_offset = new_pos
                        to_skip = 0
            self._pos = new_pos
        return self._pos

    def _fetch_next_chunk(self) -> bytes:
        if self._downloader is None or getattr(self._downloader, "_done", True):
            return b""
        status, done = self._downloader.next_chunk()
        _ = (status, done)
        return self._chunk_writer.pop_chunk()

    def read(self, size: int | None = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on closed file.")
        if size == 0 or self._pos >= self._total_size:
            return b""

        bytes_to_read = (
            self._total_size - self._pos
            if size is None or size < 0
            else min(size, self._total_size - self._pos)
        )
        chunks: list[bytes] = []
        collected = 0

        while collected < bytes_to_read:
            rel_offset = self._pos - self._buffer_offset
            if 0 <= rel_offset < len(self._buffer):
                available = len(self._buffer) - rel_offset
                take = min(available, bytes_to_read - collected)
                chunks.append(self._buffer[rel_offset : rel_offset + take])
                self._pos += take
                collected += take
            else:
                next_chunk = self._fetch_next_chunk()
                if not next_chunk:
                    break
                self._buffer_offset = self._pos
                self._buffer = next_chunk

        return b"".join(chunks)

    def readinto(self, b: Any) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed file.")
        if len(b) == 0:
            return 0
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    @property
    def buffered_chunk_size(self) -> int:
        """Return the size of the currently buffered chunk in bytes."""
        return len(self._chunk_writer.chunk)

    def readline(self, size: int | None = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on closed file.")
        max_size = -1 if size is None else size
        res = bytearray()
        while max_size < 0 or len(res) < max_size:
            ch = self.read(1)
            if not ch:
                break
            res.extend(ch)
            if ch == b"\n":
                break
        return bytes(res)

    def readlines(self, hint: int = -1) -> list[bytes]:
        if self._closed:
            raise ValueError("I/O operation on closed file.")
        lines: list[bytes] = []
        total = 0
        while line := self.readline():
            lines.append(line)
            total += len(line)
            if hint > 0 and total >= hint:
                break
        return lines

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._buffer = b""
            self._chunk_writer.chunk = b""
            super().close()

    def __enter__(self) -> GoogleDriveDownloadStream:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
