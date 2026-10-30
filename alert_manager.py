"""Enterprise SQLite-backed alert engine with severity tiers, cooldowns, and CSV export."""

from __future__ import annotations

import csv
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

STATUSES = {"New", "Acknowledged", "Resolved"}
SEVERITIES = {"CRITICAL", "HIGH", "WARNING", "INFO"}


class AlertManager:
    def __init__(self, database_path: Path | str, evidence_dir: Path | str, cooldown_seconds: int = 30):
        self.database_path = Path(database_path)
        self.evidence_dir = Path(evidence_dir)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown_seconds = max(int(cooldown_seconds), 0)
        self._initialize()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.database_path, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize(self):
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    confidence REAL,
                    track_id TEXT,
                    region TEXT,
                    status TEXT NOT NULL DEFAULT 'New',
                    snapshot_path TEXT,
                    clip_path TEXT,
                    dedupe_key TEXT NOT NULL
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")

    def create_alert(
        self,
        *,
        source: str,
        alert_type: str,
        message: str,
        severity: str = "WARNING",
        confidence: float | None = None,
        track_id: str | int | None = None,
        region: str | None = None,
        snapshot_path: str | None = None,
        clip_path: str | None = None,
        dedupe_key: str | None = None,
        cooldown_seconds: int | None = None,
    ) -> dict | None:
        now = datetime.now(timezone.utc)
        key = dedupe_key or f"{source}:{alert_type}:{track_id or ''}"
        cooldown = self.cooldown_seconds if cooldown_seconds is None else max(int(cooldown_seconds), 0)

        with self._connect() as db:
            if cooldown > 0:
                prev = db.execute(
                    "SELECT created_at FROM alerts WHERE dedupe_key=? ORDER BY created_at DESC LIMIT 1",
                    (key,),
                ).fetchone()
                if prev:
                    try:
                        prev_time = datetime.fromisoformat(prev["created_at"])
                        if now - prev_time < timedelta(seconds=cooldown):
                            return None
                    except ValueError:
                        pass

            alert_id = uuid.uuid4().hex
            created_str = now.isoformat(timespec="seconds")
            sev = severity.upper() if severity.upper() in SEVERITIES else "WARNING"

            db.execute(
                """INSERT INTO alerts 
                (id, created_at, updated_at, source, alert_type, severity, message, 
                 confidence, track_id, region, status, snapshot_path, clip_path, dedupe_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'New', ?, ?, ?)""",
                (
                    alert_id, created_str, created_str, str(source), str(alert_type),
                    sev, str(message), confidence,
                    str(track_id) if track_id is not None else None,
                    region, snapshot_path, clip_path, key
                ),
            )
            row = db.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
            return self._public(dict(row))

    @staticmethod
    def _public(row: dict | None) -> dict | None:
        if not row:
            return None
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "source": row["source"],
            "alert_type": row["alert_type"],
            "severity": row["severity"],
            "message": row["message"],
            "confidence": round(row["confidence"], 3) if row["confidence"] is not None else None,
            "track_id": row["track_id"],
            "region": row["region"],
            "status": row["status"],
            "snapshot_url": f"/api/alerts/{row['id']}/snapshot" if row.get("snapshot_path") else None,
            "clip_url": f"/api/alerts/{row['id']}/clip" if row.get("clip_path") else None,
        }

    # CRITICAL FIX: Retrieves a single alert for snapshot delivery
    def get_alert(self, alert_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        return dict(row) if row else None

    def list_alerts(
        self,
        *,
        status: str | None = None,
        severity: str | None = None,
        alert_type: str | None = None,
        source: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        clauses, params = [], []
        if status:
            clauses.append("status=?")
            params.append(status)
        if severity:
            clauses.append("severity=?")
            params.append(severity.upper())
        if alert_type:
            clauses.append("alert_type=?")
            params.append(alert_type)
        if source:
            clauses.append("source LIKE ?")
            params.append(f"%{source}%")
        if start_date:
            clauses.append("created_at >= ?")
            params.append(f"{start_date}T00:00:00")
        if end_date:
            clauses.append("created_at <= ?")
            params.append(f"{end_date}T23:59:59")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as db:
            rows = db.execute(
                f"SELECT * FROM alerts {where} ORDER BY created_at DESC LIMIT ?",
                (*params, min(max(int(limit), 1), 500)),
            ).fetchall()
        return [self._public(dict(r)) for r in rows]

    def update_status(self, alert_id: str, new_status: str) -> dict | None:
        if new_status not in STATUSES:
            raise ValueError(f"Invalid status: {new_status}. Allowed: {STATUSES}")
        updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as db:
            db.execute("UPDATE alerts SET status=?, updated_at=? WHERE id=?", (new_status, updated, alert_id))
            row = db.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        return self._public(dict(row)) if row else None

    def export_csv(self, target_path: Path | str):
        alerts = self.list_alerts(limit=1000)
        fields = [
            "id", "created_at", "source", "alert_type", "severity",
            "message", "confidence", "track_id", "region", "status"
        ]
        with Path(target_path).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(alerts)