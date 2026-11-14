"""
Signing in with ORCID, and the requests of an account of those who have
none yet (``neurolang-repl-serve --users FILE --orcid ORCID``).

A user whose name is an ORCID iD (``0000-0002-1825-0097``) signs in with
ORCID: the server sends them to ORCID's page, which sends them back with a
code, that the server exchanges for their iD and their name
(:class:`OrcidClient`, OAuth 2.0 with the scope ``/authenticate``, which
tells who they are and gives nothing else of their record). An iD of no
user may ask for an account: the request waits in a file of its own next
to the users file (:class:`AccountRequests`) until an admin adds them to
groups, or declines it.
"""

import json
import os
import re
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ORCID_ID = re.compile(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]\Z")
# Where ORCID is, and its sandbox, for testing a client of its own.
ORCID_URL = "https://orcid.org"
SANDBOX_URL = "https://sandbox.orcid.org"
# The longest message of a request, and the most requests that wait.
LONGEST_MESSAGE = 1000
MOST_REQUESTS = 500


def is_orcid(text: str) -> bool:
    """Whether *text* is an ORCID iD: four groups of four digits, the last
    character its check digit (ISO 7064 MOD 11-2), 0 to 9 or X."""
    if not isinstance(text, str) or not ORCID_ID.match(text):
        return False
    digits = text.replace("-", "")
    total = 0
    for digit in digits[:-1]:
        total = (total + int(digit)) * 2
    check = (12 - total % 11) % 11
    return digits[-1] == ("X" if check == 10 else str(check))


class OrcidError(ValueError):
    """ORCID did not tell who signed in."""


class OrcidClient:
    """
    The client of this server at ORCID: its id and secret, from ORCID's
    developer tools (https://orcid.org/developer-tools), whose redirect URI
    is this server's ``/login/orcid/callback``.

    The file *path* is YAML::

        client_id: APP-XXXXXXXXXXXXXXXX
        client_secret: ...
        # if the server is reached by another address than the one its
        # requests name (a proxy):
        redirect_uri: https://repl.example.org/login/orcid/callback
        # ORCID's sandbox, for a client of the sandbox:
        sandbox: false
    """

    def __init__(self, client_id: str, client_secret: str,
                 redirect_uri: str = "", sandbox: bool = False):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.base = SANDBOX_URL if sandbox else ORCID_URL

    @classmethod
    def from_file(cls, path: Path) -> "OrcidClient":
        """The client of the YAML file *path*; ValueError, telling why, if
        it is not one."""
        import yaml

        try:
            data = yaml.safe_load(Path(path).read_text()) or {}
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"The ORCID file {path} cannot be read: "
                             f"{error}") from None
        return cls.from_mapping(data, f"The ORCID file {path}")

    @classmethod
    def from_mapping(cls, data, where: str) -> "OrcidClient":
        """The client of the mapping *data* (a file's, or the ``orcid`` of
        the secrets), *where* naming it; ValueError if it is not one."""
        if not isinstance(data, dict) or not data.get("client_id") \
                or not data.get("client_secret"):
            raise ValueError(f"{where} has no client_id and client_secret "
                             "(ORCID's developer tools)")
        return cls(str(data["client_id"]), str(data["client_secret"]),
                   str(data.get("redirect_uri") or ""),
                   bool(data.get("sandbox")))

    def authorize_url(self, redirect_uri: str, state: str) -> str:
        """The address of ORCID's page where the visitor signs in, which
        sends them back to *redirect_uri* with *state*."""
        return f"{self.base}/oauth/authorize?" + urllib.parse.urlencode({
            "client_id": self.client_id, "response_type": "code",
            "scope": "/authenticate", "redirect_uri": redirect_uri,
            "state": state,
        })

    def exchange(self, code: str, redirect_uri: str) -> tuple:
        """The iD and the name of who signed in, from the *code* ORCID
        sent back; OrcidError if ORCID does not tell them."""
        request = urllib.request.Request(
            f"{self.base}/oauth/token", method="POST",
            headers={"Accept": "application/json",
                     "Content-Type": "application/x-www-form-urlencoded"},
            data=urllib.parse.urlencode({
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "authorization_code", "code": code,
                "redirect_uri": redirect_uri,
            }).encode(),
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                answer = json.load(response)
        except Exception as error:  # refused, or unreachable
            raise OrcidError(f"ORCID did not answer: {error}") from None
        orcid = str(answer.get("orcid") or "")
        if not is_orcid(orcid):
            raise OrcidError("ORCID did not tell who signed in")
        return orcid, str(answer.get("name") or "").strip()


class AccountRequests:
    """
    The requests of an account, by ORCID iD, in the JSON file *path*
    (readable by its owner only): the name ORCID gave, the message of the
    request and when it was made. At most :data:`MOST_REQUESTS` wait.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def current(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def list(self) -> list:
        """The requests, the oldest first."""
        return [{"orcid": orcid, **entry} for orcid, entry in sorted(
            self.current().items(), key=lambda item: item[1].get("when", 0))]

    def add(self, orcid: str, name: str, message: str) -> None:
        if not is_orcid(orcid):
            raise ValueError(f"{orcid!r} is not an ORCID iD")
        with self._lock:
            requests = self.current()
            if orcid not in requests and len(requests) >= MOST_REQUESTS:
                raise ValueError("Too many requests wait: ask an admin")
            requests[orcid] = {
                "name": str(name)[:200],
                "message": str(message).strip()[:LONGEST_MESSAGE],
                "when": time.time(),
            }
            self._write(requests)

    def remove(self, orcid: str) -> dict:
        """Take the request of *orcid* away and return it (None: none)."""
        with self._lock:
            requests = self.current()
            entry = requests.pop(orcid, None)
            if entry is not None:
                self._write(requests)
            return entry

    def _write(self, requests: dict) -> None:
        handle, temporary = tempfile.mkstemp(dir=self.path.parent,
                                             prefix=".requests-")
        try:
            with os.fdopen(handle, "w") as file:  # mkstemp's mode: 600
                json.dump(requests, file, indent=1)
            os.replace(temporary, self.path)
        except BaseException:
            os.unlink(temporary)
            raise


def requests_path(users_path: Path) -> Path:
    """The file of the requests of an account, next to the users file."""
    users_path = Path(users_path)
    return users_path.with_name(users_path.name + ".requests.json")
