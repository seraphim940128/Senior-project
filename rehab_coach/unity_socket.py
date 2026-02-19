from __future__ import annotations

import socket


class UnitySocketClient:
    def __init__(self, host: str, port: int, enabled: bool = False, timeout_s: float = 0.4):
        self.host = host
        self.port = port
        self.enabled = enabled
        self.timeout_s = timeout_s

    def send_command(self, command: str) -> bool:
        if not self.enabled:
            return False
        if not isinstance(command, str) or len(command) != 1:
            return False
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout_s)
                sock.connect((self.host, self.port))
                sock.sendall(command.encode("utf-8"))
            return True
        except OSError:
            return False
