import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any
import hashlib


class DocumentDatabase:
    def __init__(self, db_path: str = "data/documents.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Documents metadata table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                file_type TEXT,
                content_hash TEXT,
                metadata TEXT,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        """
        )

        # Document chunks table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT,
                chunk_index INTEGER,
                content TEXT,
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """
        )

        # Update log for tracking modifications
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS update_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                operation TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def add_document(
        self, filename: str, filepath: str, content: str, metadata: dict = None
    ) -> str:
        """Add a document to the database"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        doc_id = f"doc_{content_hash[:16]}"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if document already exists
        cursor.execute(
            "SELECT id FROM documents WHERE content_hash = ?", (content_hash,)
        )
        existing = cursor.fetchone()

        if existing:
            doc_id = existing[0]
            # Update last modified time
            cursor.execute(
                "UPDATE documents SET last_modified = ? WHERE id = ?",
                (datetime.now(), doc_id),
            )
        else:
            # Insert new document
            cursor.execute(
                """
                INSERT INTO documents (id, filename, filepath, content_hash, metadata, upload_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    doc_id,
                    filename,
                    filepath,
                    content_hash,
                    json.dumps(metadata or {}),
                    datetime.now(),
                ),
            )

        # Log the operation
        cursor.execute(
            """
            INSERT INTO update_log (document_id, operation, details)
            VALUES (?, ?, ?)
        """,
            (
                doc_id,
                "ADD" if not existing else "UPDATE",
                json.dumps({"filename": filename}),
            ),
        )

        conn.commit()
        conn.close()
        return doc_id

    def update_document_chunks(self, doc_id: str, chunks: List[Dict[str, Any]]):
        """Update chunks for a document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete existing chunks
        cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))

        # Insert new chunks
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{idx}"
            cursor.execute(
                """
                INSERT INTO document_chunks (chunk_id, document_id, chunk_index, content, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    chunk_id,
                    doc_id,
                    idx,
                    chunk["content"],
                    json.dumps(chunk.get("metadata", {})),
                ),
            )

        conn.commit()
        conn.close()

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Retrieve document metadata"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all active documents"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM documents WHERE is_active = 1 ORDER BY upload_time DESC"
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def delete_document(self, doc_id: str):
        """Soft delete a document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("UPDATE documents SET is_active = 0 WHERE id = ?", (doc_id,))

        cursor.execute(
            """
            INSERT INTO update_log (document_id, operation, details)
            VALUES (?, ?, ?)
        """,
            (doc_id, "DELETE", "Soft delete"),
        )

        conn.commit()
        conn.close()
