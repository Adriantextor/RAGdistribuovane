"""Create local secrets without printing them or overwriting an existing configuration."""
import secrets
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / ".env.mac"
token = secrets.token_urlsafe(32)
password = secrets.token_hex(24)
content = (root / ".env.mac.example").read_text()
content = content.replace("replace-with-a-random-token", token)
content = content.replace("replace-with-a-different-random-password", password)
with target.open("x") as output:
    output.write(content)
target.chmod(0o600)
print("Created .env.mac with generated secrets. Keep this file private.")
