import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_text_file(path, write_content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            newline="",
        ) as file:
            temporary_path = Path(file.name)
            write_content(file)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_json(path, value):
    def write(file):
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")

    _atomic_text_file(path, write)


def _csv_value(value):
    if value is None:
        return ""

    if isinstance(value, float):
        return f"{value:.6f}"

    return value


def atomic_write_csv(path, columns, rows):
    def write(file):
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: _csv_value(row.get(column, ""))
                    for column in columns
                }
            )

    _atomic_text_file(path, write)


def load_csv_rows(path, expected_columns):
    path = Path(path)

    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        actual_columns = tuple(reader.fieldnames or ())

        if actual_columns != tuple(expected_columns):
            raise ValueError(
                f"CSV schema mismatch for {path}: expected {tuple(expected_columns)}, "
                f"found {actual_columns}"
            )

        return list(reader)


def validate_unique_rows(rows, key_fields):
    seen = set()

    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)

        if key in seen:
            raise ValueError(f"duplicate CSV result key: {key}")

        seen.add(key)


def rows_for_job(rows, job_key, job_key_fields):
    return [
        row
        for row in rows
        if tuple(row.get(field, "") for field in job_key_fields) == job_key
    ]


def job_checkpoint_status(
    rows,
    job_key,
    job_key_fields,
    expected_modes,
):
    job_rows = rows_for_job(rows, job_key, job_key_fields)

    if not job_rows:
        return "pending"

    modes = [row.get("mode", "") for row in job_rows]

    if len(job_rows) != len(expected_modes) or set(modes) != set(expected_modes):
        return "incomplete"

    if all(row.get("status") == "ok" and not row.get("error") for row in job_rows):
        return "complete"

    return "failed"


def replace_job_rows(rows, new_rows, job_key_fields, row_key_fields, sort_key):
    if not new_rows:
        raise ValueError("new_rows must contain one complete job checkpoint")

    job_keys = {
        tuple(row.get(field, "") for field in job_key_fields)
        for row in new_rows
    }

    if len(job_keys) != 1:
        raise ValueError("all checkpoint rows must belong to the same job")

    job_key = next(iter(job_keys))
    retained = [
        row
        for row in rows
        if tuple(row.get(field, "") for field in job_key_fields) != job_key
    ]
    updated = retained + list(new_rows)
    validate_unique_rows(updated, row_key_fields)
    updated.sort(key=sort_key)
    return updated


class CsvCheckpoint:
    def __init__(
        self,
        output_path,
        metadata_path,
        columns,
        row_key_fields,
        job_key_fields,
        expected_modes,
        sort_key,
        schema_version,
        config_hash,
        rows,
        metadata,
    ):
        self.output_path = Path(output_path)
        self.metadata_path = Path(metadata_path)
        self.columns = tuple(columns)
        self.row_key_fields = tuple(row_key_fields)
        self.job_key_fields = tuple(job_key_fields)
        self.expected_modes = tuple(expected_modes)
        self.sort_key = sort_key
        self.schema_version = schema_version
        self.config_hash = config_hash
        self.rows = rows
        self.metadata = metadata

    @classmethod
    def open(
        cls,
        *,
        output_path,
        metadata_path,
        columns,
        row_key_fields,
        job_key_fields,
        expected_modes,
        sort_key,
        schema_version,
        configuration,
        config_hash,
        resume=False,
        overwrite=False,
    ):
        output_path = Path(output_path)
        metadata_path = Path(metadata_path)

        if output_path.resolve() == metadata_path.resolve():
            raise ValueError("CSV and metadata paths must be different")

        output_exists = output_path.exists()
        metadata_exists = metadata_path.exists()

        if overwrite:
            rows = []
            metadata = cls._new_metadata(
                schema_version,
                configuration,
                config_hash,
            )
        elif output_exists or metadata_exists:
            if not resume:
                raise FileExistsError(
                    "Output checkpoint already exists. Use --resume or "
                    f"--overwrite: {output_path}"
                )

            if not output_exists or not metadata_exists:
                raise ValueError(
                    "Cannot safely resume because the CSV and metadata "
                    "sidecar are not both present."
                )

            rows = load_csv_rows(output_path, columns)
            validate_unique_rows(rows, row_key_fields)

            with metadata_path.open() as file:
                metadata = json.load(file)

            if metadata.get("schema_version") != schema_version:
                raise ValueError(
                    "checkpoint metadata schema version is incompatible"
                )

            if metadata.get("configuration_hash") != config_hash:
                raise ValueError(
                    "checkpoint configuration does not match the current "
                    "code, parameters, dataset, or evaluator backend"
                )

            if any(row.get("config_hash", "") != config_hash for row in rows):
                raise ValueError(
                    "CSV contains rows from a different configuration"
                )
        else:
            rows = []
            metadata = cls._new_metadata(
                schema_version,
                configuration,
                config_hash,
            )

        checkpoint = cls(
            output_path,
            metadata_path,
            columns,
            row_key_fields,
            job_key_fields,
            expected_modes,
            sort_key,
            schema_version,
            config_hash,
            rows,
            metadata,
        )
        atomic_write_csv(output_path, columns, rows)
        checkpoint.metadata["status"] = "running"
        checkpoint.save_metadata()
        return checkpoint

    @staticmethod
    def _new_metadata(schema_version, configuration, config_hash):
        now = utc_timestamp()
        return {
            "schema_version": schema_version,
            "configuration_hash": config_hash,
            "configuration": configuration,
            "created_at": now,
            "updated_at": now,
            "status": "running",
            "progress": {
                "jobs": 0,
                "complete_jobs": 0,
                "failed_jobs": 0,
                "incomplete_jobs": 0,
                "rows": 0,
            },
            "invocations": [],
        }

    def job_status(self, job_key):
        return job_checkpoint_status(
            self.rows,
            job_key,
            self.job_key_fields,
            self.expected_modes,
        )

    def progress(self):
        grouped = {}

        for row in self.rows:
            key = tuple(
                row.get(field, "")
                for field in self.job_key_fields
            )
            grouped.setdefault(key, []).append(row)

        counts = {
            "jobs": len(grouped),
            "complete_jobs": 0,
            "failed_jobs": 0,
            "incomplete_jobs": 0,
            "rows": len(self.rows),
        }

        for key in grouped:
            status = self.job_status(key)
            counts[f"{status}_jobs"] += 1

        return counts

    def save_metadata(self):
        self.metadata["updated_at"] = utc_timestamp()
        self.metadata["progress"] = self.progress()
        atomic_write_json(self.metadata_path, self.metadata)

    def checkpoint_job(self, job_rows):
        self.rows = replace_job_rows(
            self.rows,
            job_rows,
            self.job_key_fields,
            self.row_key_fields,
            self.sort_key,
        )
        atomic_write_csv(self.output_path, self.columns, self.rows)
        self.save_metadata()
