"""Connectivity manager - auto-detect online/offline status."""
import asyncio
import socket
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class ConnectivityManager:
    """Manages online/offline state with auto-detection."""

    def __init__(self, check_hosts: Optional[list[str]] = None, timeout: float = 3.0):
        self.check_hosts = check_hosts or [
            ("8.8.8.8", 53),      # Google DNS
            ("1.1.1.1", 53),      # Cloudflare DNS
            ("208.67.222.222", 53), # OpenDNS
        ]
        self.timeout = timeout
        self._is_online: Optional[bool] = None
        self._last_check: float = 0
        self._check_interval: float = 30.0  # Re-check every 30 seconds

    async def check_online(self) -> bool:
        """Check if internet is available."""
        import time
        now = time.time()
        if self._is_online is not None and (now - self._last_check) < self._check_interval:
            return self._is_online

        for host, port in self.check_hosts:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=self.timeout
                )
                writer.close()
                await writer.wait_closed()
                self._is_online = True
                self._last_check = now
                return True
            except Exception:
                continue

        self._is_online = False
        self._last_check = now
        return False

    def check_online_sync(self) -> bool:
        """Synchronous version for non-async contexts."""
        try:
            for host, port in self.check_hosts:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                try:
                    result = sock.connect_ex((host, port))
                    if result == 0:
                        self._is_online = True
                        return True
                finally:
                    sock.close()
            self._is_online = False
            return False
        except Exception:
            self._is_online = False
            return False

    @property
    def is_online(self) -> bool:
        """Return cached online status (check_sync if unknown)."""
        if self._is_online is None:
            return self.check_online_sync()
        return self._is_online

    def reset(self) -> None:
        """Force re-check on next call."""
        self._is_online = None
        self._last_check = 0

