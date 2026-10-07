"""Committed entry receipts from independent read-only SQLite snapshots.

Receipts prove validated storage, not permission to submit an order. Entry
release still requires journal reconciliation and independent daily accounting.
As elsewhere in this process, the registry enforces implementation integrity;
it is not isolation from hostile code sharing the interpreter.
"""

import json
import weakref
from dataclasses import dataclass
from datetime import datetime

import aiosqlite

from .paper_entry_settlement import PaperEntryTerminalRecord
from .paper_entry_storage_replay import read_entry_settlement
from .safety.models import ValidationError, canonical_json, sha256_text
from .safety.sqlite_identity import SQLitePathBinding


def _receipt_runtime():
    token = object()
    owned = weakref.WeakKeyDictionary()

    @dataclass(frozen=True, eq=False)
    class PaperEntrySettlementReceipt:
        settlement_id: str
        record: PaperEntryTerminalRecord
        trade_id: int | None
        database_path: str
        database_identity: str
        database_device: int
        database_inode: int
        committed_at: datetime
        _producer_token: object

        def __post_init__(self):
            if self._producer_token is not token:
                raise ValidationError("entry receipt requires the committed ledger producer")

        def canonical_payload(self):
            return canonical_json(
                dict(
                    settlement_id=self.settlement_id,
                    request_fingerprint=self.record.fingerprint,
                    trade_id=self.trade_id,
                    database_path=self.database_path,
                    database_identity=self.database_identity,
                    database_device=self.database_device,
                    database_inode=self.database_inode,
                    committed_at=self.committed_at,
                    schema_version=1,
                    settlement_kind="ENTRY",
                )
            )

        def fingerprint(self):
            return sha256_text(self.canonical_payload())

        def __copy__(self):
            raise ValidationError("entry receipts cannot copy")

        def __deepcopy__(self, memo):
            raise ValidationError("entry receipts cannot copy")

        def __reduce__(self):
            raise ValidationError("entry receipts cannot serialize")

    def assert_owned(receipt, *, database, runtime_contract):
        from .database_async import AsyncTradingDatabase

        if (
            type(receipt) is not PaperEntrySettlementReceipt
            or type(database) is not AsyncTradingDatabase
        ):
            raise ValidationError("entry receipt ownership is invalid")
        if type(receipt.record) is not PaperEntryTerminalRecord:
            raise ValidationError("entry receipt record type changed")
        original = owned.get(receipt)
        if original is None or original != receipt.canonical_payload():
            raise ValidationError("entry receipt is unregistered or changed")
        path, identity = database._expected_safety_database(runtime_contract=runtime_contract)
        claim = json.loads(receipt.record.payload_json)["claim"]
        if (
            str(path) != receipt.database_path
            or identity != receipt.database_identity
            or runtime_contract.execution_mode != "paper"
            or runtime_contract.ibkr_readonly is not True
            or runtime_contract.safety_account_scope != claim["account_scope"]
            or runtime_contract.safety_execution_domain_scope != claim["execution_domain_scope"]
            or database._expected_database_file_identity
            != (receipt.database_device, receipt.database_inode)
        ):
            raise ValidationError("entry receipt runtime identity differs")
        binding = SQLitePathBinding.open_readonly(path)
        try:
            if (binding.device, binding.inode) != (receipt.database_device, receipt.database_inode):
                raise ValidationError("entry receipt file identity differs")
        finally:
            binding.close()
        return receipt

    async def recover(record, *, database, runtime_contract, journal):
        from .database_async import AsyncTradingDatabase

        if (
            type(database) is not AsyncTradingDatabase
            or type(record) is not PaperEntryTerminalRecord
        ):
            raise ValidationError("entry receipt recovery requires exact database and record types")
        # Capture the payload before yielding; later caller mutation must not
        # change the record bound to this committed snapshot.
        record = PaperEntryTerminalRecord(record.payload_json)
        path, identity = database._expected_safety_database(runtime_contract=runtime_contract)
        binding = SQLitePathBinding.open_readonly(path)
        try:
            # A separate mode=ro connection cannot see another connection's
            # uncommitted staging, nor repair a missing row. Never use the pool.
            async with aiosqlite.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
                descriptor = await database._sqlite_descriptor_identity(connection)
                binding = binding.bind_sqlite_connection(descriptor)
                await connection.execute("PRAGMA foreign_keys=ON")
                await connection.execute("BEGIN")
                try:
                    row = await read_entry_settlement(
                        connection,
                        record,
                        database=database,
                        runtime_contract=runtime_contract,
                        journal=journal,
                    )
                finally:
                    await connection.rollback()
                binding.assert_connection_identity(
                    await database._sqlite_descriptor_identity(connection)
                )
            binding.assert_connection_identity(descriptor)
            receipt = PaperEntrySettlementReceipt(
                row.settlement_id,
                record,
                row.trade_id,
                str(path),
                identity,
                descriptor.device,
                descriptor.inode,
                row.committed_at,
                token,
            )
            if receipt.fingerprint() != row.fingerprint:
                raise ValidationError("entry receipt differs from committed fingerprint")
            owned[receipt] = receipt.canonical_payload()
            return assert_owned(receipt, database=database, runtime_contract=runtime_contract)
        finally:
            binding.close()

    return PaperEntrySettlementReceipt, recover, assert_owned


PaperEntrySettlementReceipt, recover_committed_entry_receipt, assert_owned_entry_receipt = (
    _receipt_runtime()
)
del _receipt_runtime
