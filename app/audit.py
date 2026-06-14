"""감사 추적(audit trail) · 사람 검토 큐 · 드리프트 모니터링.

규제 산업(ISO 13485 등)에서 AI를 운영하려면 "언제, 무엇을, 어떤 확신으로
판정했고, 사람이 어떻게 검증했는가"가 추적 가능해야 한다. 모든 추론을
SQLite에 남기고, 확신이 낮은 건은 사람 검토로 돌리며, 입력 분포가 흔들리면
경보한다. 외부 의존성 없이 표준 라이브러리만 사용한다.
"""
import hashlib
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from math import log
from pathlib import Path

from . import config

DB_PATH = Path(__file__).parent.parent / "data" / "audit" / "inspections.db"
# 검토 필요(needs_review) 샘플의 이미지를 저장하는 곳. 재라벨링·재학습에 쓰인다.
REVIEW_IMG_DIR = DB_PATH.parent / "images"

# FastAPI는 동기 엔드포인트를 스레드풀에서 돌리므로 쓰기를 직렬화한다.
_write_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT    NOT NULL,   -- ISO8601 UTC
    image_sha256   TEXT    NOT NULL,   -- 입력 추적용 해시 (원본은 저장 안 함)
    predicted_type TEXT,
    severity       TEXT,
    confidence     REAL    NOT NULL,
    latency_ms     REAL,
    model_version  TEXT,
    review_status  TEXT    NOT NULL,   -- auto_accepted | needs_review | corrected
    true_type      TEXT,               -- 사람이 확정한 정답 (검토 후)
    reviewer       TEXT,
    reviewed_at    TEXT,
    raw_output     TEXT
);
CREATE INDEX IF NOT EXISTS idx_ts ON inspections(ts);
CREATE INDEX IF NOT EXISTS idx_status ON inspections(review_status);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _session():
    """트랜잭션 커밋/롤백 후 연결을 반드시 닫는다.

    sqlite3의 `with conn`은 트랜잭션만 관리하고 연결은 닫지 않으므로
    (특히 Windows에서) 파일 핸들이 남는다. 여기서 명시적으로 닫는다.
    """
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    with _write_lock, _session() as conn:
        conn.executescript(_SCHEMA)


def image_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def decide_review(confidence: float, predicted_type: str | None) -> str:
    """확신도/파싱 결과로 자동 승인 여부를 정한다."""
    if predicted_type is None:
        return "needs_review"  # 파싱 실패나 미지 클래스는 무조건 사람에게
    if confidence < config.CONFIDENCE_THRESHOLD:
        return "needs_review"
    return "auto_accepted"


def log_inference(
    *,
    image_sha256: str,
    predicted_type: str | None,
    severity: str | None,
    confidence: float,
    latency_ms: float,
    model_version: str,
    review_status: str,
    raw_output: str,
) -> int:
    """추론 1건 기록 → record id 반환."""
    with _write_lock, _session() as conn:
        cur = conn.execute(
            """INSERT INTO inspections
               (ts, image_sha256, predicted_type, severity, confidence,
                latency_ms, model_version, review_status, raw_output)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (_now(), image_sha256, predicted_type, severity, confidence,
             latency_ms, model_version, review_status, raw_output),
        )
        return cur.lastrowid


def get_record(record_id: int) -> dict | None:
    with _session() as conn:
        row = conn.execute(
            "SELECT * FROM inspections WHERE id = ?", (record_id,)
        ).fetchone()
        return dict(row) if row else None


def list_review_queue(limit: int = 50) -> list[dict]:
    """사람 검토가 필요한 건 목록 (오래된 것부터)."""
    with _session() as conn:
        rows = conn.execute(
            """SELECT * FROM inspections
               WHERE review_status = 'needs_review'
               ORDER BY ts ASC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def submit_correction(record_id: int, true_type: str, reviewer: str) -> dict | None:
    """사람이 정답을 확정 → 기록 갱신. 갱신된 레코드 반환."""
    with _write_lock, _session() as conn:
        cur = conn.execute(
            """UPDATE inspections
               SET true_type = ?, reviewer = ?, reviewed_at = ?,
                   review_status = 'corrected'
               WHERE id = ?""",
            (true_type, reviewer, _now(), record_id),
        )
        if cur.rowcount == 0:
            return None
        row = conn.execute(
            "SELECT * FROM inspections WHERE id = ?", (record_id,)
        ).fetchone()
        return dict(row)


def stats() -> dict:
    """운영 현황 요약."""
    with _session() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM inspections").fetchone()["c"]
        by_status = {
            r["review_status"]: r["c"]
            for r in conn.execute(
                "SELECT review_status, COUNT(*) AS c FROM inspections GROUP BY review_status"
            ).fetchall()
        }
        avg_conf = conn.execute(
            "SELECT AVG(confidence) AS a FROM inspections"
        ).fetchone()["a"]

        # 사람이 검토한 건에서의 실측 정확도 (true_type vs predicted_type)
        reviewed = conn.execute(
            """SELECT predicted_type, true_type FROM inspections
               WHERE review_status = 'corrected' AND true_type IS NOT NULL"""
        ).fetchall()
        observed_acc = None
        if reviewed:
            correct = sum(1 for r in reviewed if r["predicted_type"] == r["true_type"])
            observed_acc = round(correct / len(reviewed), 4)

    review_rate = round(
        (by_status.get("needs_review", 0)) / total, 4
    ) if total else 0.0
    return {
        "total_inspections": total,
        "by_status": by_status,
        "avg_confidence": round(avg_conf, 4) if avg_conf is not None else None,
        "needs_review_rate": review_rate,
        "observed_accuracy_on_reviewed": observed_acc,
        "reviewed_count": len(reviewed),
    }


def _psi(reference: dict, recent: dict, classes: list[str]) -> float:
    """Population Stability Index — 두 분포의 차이. 0 안정 / 0.25+ 큰 변화."""
    eps = 1e-6
    ref_total = sum(reference.values()) or 1
    rec_total = sum(recent.values()) or 1
    psi = 0.0
    for c in classes:
        ref_p = (reference.get(c, 0) / ref_total) or eps
        rec_p = (recent.get(c, 0) / rec_total) or eps
        psi += (rec_p - ref_p) * log(rec_p / ref_p)
    return psi


def drift_report(window: int | None = None) -> dict:
    """최근 window건 vs 그 이전 전체를 비교해 입력/출력 드리프트를 진단."""
    window = window or config.DRIFT_WINDOW
    classes = config.DEFECT_CLASSES
    with _session() as conn:
        rows = conn.execute(
            "SELECT predicted_type, confidence FROM inspections ORDER BY ts ASC"
        ).fetchall()

    total = len(rows)
    if total < window * 2:
        return {
            "status": "insufficient_data",
            "detail": f"드리프트 판정에는 최소 {window * 2}건 필요 (현재 {total}건)",
            "total": total,
        }

    reference, recent = rows[:-window], rows[-window:]

    def _dist(rs):
        d = {c: 0 for c in classes}
        for r in rs:
            if r["predicted_type"] in d:
                d[r["predicted_type"]] += 1
        return d

    def _mean_conf(rs):
        return sum(r["confidence"] for r in rs) / len(rs)

    psi = _psi(_dist(reference), _dist(recent), classes)
    ref_conf, rec_conf = _mean_conf(reference), _mean_conf(recent)
    conf_drop = ref_conf - rec_conf

    if psi >= config.DRIFT_PSI_ALERT or conf_drop >= config.DRIFT_CONF_DROP:
        status = "alert"
    elif psi >= config.DRIFT_PSI_WARN:
        status = "warn"
    else:
        status = "stable"

    return {
        "status": status,
        "window": window,
        "class_distribution_psi": round(psi, 4),
        "reference_mean_confidence": round(ref_conf, 4),
        "recent_mean_confidence": round(rec_conf, 4),
        "confidence_drop": round(conf_drop, 4),
        "thresholds": {
            "psi_warn": config.DRIFT_PSI_WARN,
            "psi_alert": config.DRIFT_PSI_ALERT,
            "confidence_drop": config.DRIFT_CONF_DROP,
        },
    }


# ── 자가개선 루프 지원 (이미지 보관 · 메타 · 교정 라벨) ──
def save_review_image(image_sha256: str, img) -> str:
    """검토 필요 샘플의 이미지를 해시 이름으로 저장(재학습용). 경로 반환."""
    REVIEW_IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = REVIEW_IMG_DIR / f"{image_sha256}.png"
    if not path.exists():
        img.save(path)
    return str(path)


def get_meta(key: str, default: str | None = None) -> str | None:
    with _session() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default


def set_meta(key: str, value: str) -> None:
    with _write_lock, _session() as conn:
        conn.execute(
            """INSERT INTO meta (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (key, str(value)),
        )


def count_corrected_since(ts: str | None = None) -> int:
    """ts 이후(없으면 전체) 사람이 교정한 라벨 수 — 재학습 트리거 판단용."""
    with _session() as conn:
        if ts:
            row = conn.execute(
                """SELECT COUNT(*) AS c FROM inspections
                   WHERE review_status = 'corrected' AND reviewed_at > ?""",
                (ts,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM inspections WHERE review_status = 'corrected'"
            ).fetchone()
        return row["c"]


def corrected_for_training() -> list[dict]:
    """교정 라벨 + 저장 이미지 경로 목록 (재학습 데이터 구성용, D2)."""
    with _session() as conn:
        rows = conn.execute(
            """SELECT image_sha256, true_type, reviewed_at FROM inspections
               WHERE review_status = 'corrected' AND true_type IS NOT NULL
               ORDER BY reviewed_at ASC"""
        ).fetchall()
    out = []
    for r in rows:
        p = REVIEW_IMG_DIR / f"{r['image_sha256']}.png"
        out.append({
            "image_sha256": r["image_sha256"],
            "true_type": r["true_type"],
            "image_path": str(p),
            "image_exists": p.exists(),
            "reviewed_at": r["reviewed_at"],
        })
    return out
