"""Build the local SQLite database consumed by the Streamlit dashboard."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROPOSTAS = ROOT / "output" / "vrum_propostas_flags.csv"
DEFAULT_TIMELINE = ROOT / "output" / "vrum_timeline_cronologica.csv"
DEFAULT_DATABASE = Path(__file__).resolve().parent / "data" / "vrum_dashboard.sqlite"
SCHEMA = Path(__file__).resolve().parent / "schema.sql"

PROPOSTA_COLUMNS = (
    "id_proposta",
    "data_hora_proposta",
    "chassi_id_sintetico",
    "if_id_sintetico",
    "cpf_cnpj_proponente_sintetico",
    "tipo_proponente",
    "canal",
    "uf_proposta",
    "safra",
    "score_modelo",
    "target_observado",
    "target_risco_90d",
    "valor_financiado",
    "valor_entrada",
    "prazo_meses",
    "valor_fipe_referencia",
    "ltv_fipe",
    "dias_desde_ultima_proposta",
    "qtd_propostas_historicas",
    "qtd_sinais_mesa",
    "zona_decisao",
)

EVENTO_COLUMNS = (
    "id_proposta",
    "chassi_id_sintetico",
    "dt_evento",
    "tipo_evento",
    "origem_registro",
    "valor_financiado",
    "canal",
    "uf_proposta",
    "marca",
    "flag_evento_risco",
)

PROPOSTA_SQL = """INSERT INTO propostas VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
EVENTO_SQL = """INSERT INTO eventos_risco (
    id_proposta, chassi_id_sintetico, dt_evento, tipo_evento,
    origem_registro, valor_financiado, canal, uf_proposta, marca,
    flag_evento_risco
) VALUES (?,?,?,?,?,?,?,?,?,?)"""


def number(value: str) -> float | None:
    return float(value) if value not in (None, "") else None


def integer(value: str) -> int | None:
    return int(float(value)) if value not in (None, "") else None


def boolean(value: str) -> int | None:
    if value in (None, ""):
        return None
    return int(value.lower() in {"1", "true", "sim", "yes"})


def load_propostas(
    connection: sqlite3.Connection,
    path: Path,
    batch_size: int = 5000,
    progress_every: int = 100_000,
) -> int:
    count = 0
    last_report = 0
    started = time.monotonic()
    batch: list[tuple[object, ...]] = []
    print(f"[1/2] Carregando propostas: {path}", flush=True)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            batch.append(
                (
                    row["id_proposta"],
                    row["data_hora_proposta"],
                    row["chassi_id_sintetico"],
                    row["if_id_sintetico"],
                    row["cpf_cnpj_proponente_sintetico"],
                    row["tipo_proponente"],
                    row["canal"],
                    row["uf_proposta"],
                    row["safra"],
                    number(row["score_modelo"]),
                    integer(row["target_observado"]),
                    integer(row["target_risco_90d"]),
                    number(row["valor_financiado"]),
                    number(row["valor_entrada"]),
                    integer(row["prazo_meses"]),
                    number(row["valor_fipe_referencia"]),
                    number(row["ltv_fipe"]),
                    number(row["dias_desde_ultima_proposta"]),
                    integer(row["qtd_propostas_historicas"]),
                    integer(row["qtd_sinais_mesa"]),
                    row["zona_decisao"],
                )
            )
            if len(batch) >= batch_size:
                connection.executemany(PROPOSTA_SQL, batch)
                count += len(batch)
                batch.clear()
                if count - last_report >= progress_every:
                    elapsed = time.monotonic() - started
                    print(f"      propostas: {count:,} ({elapsed:.1f}s)", flush=True)
                    last_report = count
        if batch:
            connection.executemany(PROPOSTA_SQL, batch)
            count += len(batch)
    print(f"      propostas concluídas: {count:,} ({time.monotonic() - started:.1f}s)", flush=True)
    return count


def load_eventos(
    connection: sqlite3.Connection,
    path: Path,
    batch_size: int = 5000,
    progress_every: int = 10_000,
) -> int:
    count = 0
    last_report = 0
    started = time.monotonic()
    batch: list[tuple[object, ...]] = []
    print(f"[2/2] Carregando eventos de risco: {path}", flush=True)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            if row["tipo_evento"] == "PROPOSTA_FINANCIAMENTO":
                continue
            batch.append(
                (
                    row["id_proposta"],
                    row["chassi_id_sintetico"],
                    row["dt_evento"],
                    row["tipo_evento"],
                    row["origem_registro"],
                    number(row["valor_financiado"]),
                    row["canal"],
                    row["uf_proposta"],
                    row["marca"],
                    integer(row["flag_evento_risco"]),
                )
            )
            if len(batch) >= batch_size:
                connection.executemany(EVENTO_SQL, batch)
                count += len(batch)
                batch.clear()
                if count - last_report >= progress_every:
                    elapsed = time.monotonic() - started
                    print(f"      eventos: {count:,} ({elapsed:.1f}s)", flush=True)
                    last_report = count
        if batch:
            connection.executemany(EVENTO_SQL, batch)
            count += len(batch)
    print(f"      eventos concluídos: {count:,} ({time.monotonic() - started:.1f}s)", flush=True)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--propostas", type=Path, default=DEFAULT_PROPOSTAS)
    parser.add_argument("--timeline", type=Path, default=DEFAULT_TIMELINE)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    print("Iniciando construção do banco SQLite", flush=True)
    for path in (args.propostas, args.timeline, SCHEMA):
        if not path.exists():
            raise FileNotFoundError(path)

    args.database.parent.mkdir(parents=True, exist_ok=True)
    if args.database.exists():
        print(f"Removendo banco anterior: {args.database}", flush=True)
        args.database.unlink()

    print(f"Criando banco: {args.database}", flush=True)
    connection = sqlite3.connect(args.database)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.executescript(SCHEMA.read_text(encoding="utf-8"))
        print("Schema inicializado", flush=True)
        with connection:
            proposta_count = load_propostas(connection, args.propostas)
            evento_count = load_eventos(connection, args.timeline)
            connection.execute(
                "INSERT INTO carga_metadata VALUES (?, ?)",
                ("propostas_arquivo", str(args.propostas)),
            )
            connection.execute(
                "INSERT INTO carga_metadata VALUES (?, ?)",
                ("timeline_arquivo", str(args.timeline)),
            )
            connection.execute(
                "INSERT INTO carga_metadata VALUES (?, ?)",
                ("gerado_em", datetime.now().isoformat(timespec="seconds")),
            )
            connection.execute(
                "INSERT INTO carga_metadata VALUES (?, ?)",
                ("propostas_lidas", str(proposta_count)),
            )
            connection.execute(
                "INSERT INTO carga_metadata VALUES (?, ?)",
                ("eventos_lidos", str(evento_count)),
            )
            print("Atualizando agregações de fraude", flush=True)
            connection.executescript(
                """
                DROP TABLE IF EXISTS fraude_mensal;
                DROP TABLE IF EXISTS fraude_canal;
                DROP TABLE IF EXISTS fraude_uf;
                DROP TABLE IF EXISTS fraude_marca;

                CREATE TABLE fraude_mensal AS
                SELECT substr(dt_evento, 1, 7) AS mes,
                       COUNT(DISTINCT id_proposta) AS casos,
                       COALESCE(SUM(valor_financiado), 0) AS exposicao
                FROM eventos_risco
                WHERE tipo_evento = 'FRAUDE_CONFIRMADA'
                GROUP BY substr(dt_evento, 1, 7);

                CREATE TABLE fraude_canal AS
                SELECT COALESCE(e.canal, p.canal) AS canal,
                       COUNT(DISTINCT e.id_proposta) AS casos,
                       COALESCE(SUM(e.valor_financiado), 0) AS exposicao
                FROM eventos_risco e
                LEFT JOIN propostas p ON p.id_proposta = e.id_proposta
                WHERE e.tipo_evento = 'FRAUDE_CONFIRMADA'
                GROUP BY COALESCE(e.canal, p.canal);

                CREATE TABLE fraude_uf AS
                SELECT COALESCE(e.uf_proposta, p.uf_proposta) AS uf_proposta,
                       COUNT(DISTINCT e.id_proposta) AS casos,
                       COALESCE(SUM(e.valor_financiado), 0) AS exposicao
                FROM eventos_risco e
                LEFT JOIN propostas p ON p.id_proposta = e.id_proposta
                WHERE e.tipo_evento = 'FRAUDE_CONFIRMADA'
                GROUP BY COALESCE(e.uf_proposta, p.uf_proposta);

                CREATE TABLE fraude_marca AS
                SELECT marca,
                       COUNT(DISTINCT id_proposta) AS casos,
                       COALESCE(SUM(valor_financiado), 0) AS exposicao
                FROM eventos_risco
                WHERE tipo_evento = 'FRAUDE_CONFIRMADA'
                GROUP BY marca;
                """
            )
            print("Agregações concluídas", flush=True)
        connection.execute("ANALYZE")
        connection.commit()
        print("Índices e estatísticas atualizados", flush=True)
    finally:
        connection.close()

    print(f"Banco criado: {args.database}", flush=True)
    print(f"Propostas: {proposta_count:,}", flush=True)
    print(f"Eventos de risco: {evento_count:,}", flush=True)
    print(f"Tempo total: {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
