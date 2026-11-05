"""
Blockchain Security Logging Module for EV Charging Station Intrusion Detection.
Implements a tamper-evident, SHA-256-hashed distributed ledger prototype for auditability.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from config import LEDGER_FILE


class Block:
    """
    Represents an immutable audit block in the intrusion detection security ledger.
    """

    def __init__(
        self,
        index: int,
        timestamp: str,
        dataset_filename: str,
        number_of_analysed_records: int,
        normal_traffic_count: int,
        malicious_traffic_count: int,
        attack_wise_counts: Dict[str, int],
        previous_block_hash: str,
        current_block_hash: Optional[str] = None,
        nonce: int = 0
    ):
        self.index = index
        self.timestamp = timestamp
        self.dataset_filename = dataset_filename
        self.number_of_analysed_records = number_of_analysed_records
        self.normal_traffic_count = normal_traffic_count
        self.malicious_traffic_count = malicious_traffic_count
        self.attack_wise_counts = attack_wise_counts
        self.previous_block_hash = previous_block_hash
        self.nonce = nonce
        self.current_block_hash = current_block_hash or self.calculate_hash()

    def calculate_hash(self) -> str:
        """
        Computes SHA-256 cryptographic hash of canonical block contents.
        """
        block_content = {
            "index": self.index,
            "timestamp": self.timestamp,
            "dataset_filename": self.dataset_filename,
            "number_of_analysed_records": self.number_of_analysed_records,
            "normal_traffic_count": self.normal_traffic_count,
            "malicious_traffic_count": self.malicious_traffic_count,
            "attack_wise_counts": self.attack_wise_counts,
            "previous_block_hash": self.previous_block_hash,
            "nonce": self.nonce
        }
        encoded_string = json.dumps(block_content, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded_string).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Converts block to dictionary for JSON persistence."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "dataset_filename": self.dataset_filename,
            "number_of_analysed_records": self.number_of_analysed_records,
            "normal_traffic_count": self.normal_traffic_count,
            "malicious_traffic_count": self.malicious_traffic_count,
            "attack_wise_counts": self.attack_wise_counts,
            "previous_block_hash": self.previous_block_hash,
            "current_block_hash": self.current_block_hash,
            "nonce": self.nonce
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Block":
        """Instantiates a Block from dictionary data."""
        return cls(
            index=data["index"],
            timestamp=data["timestamp"],
            dataset_filename=data["dataset_filename"],
            number_of_analysed_records=data["number_of_analysed_records"],
            normal_traffic_count=data["normal_traffic_count"],
            malicious_traffic_count=data["malicious_traffic_count"],
            attack_wise_counts=data.get("attack_wise_counts", {}),
            previous_block_hash=data["previous_block_hash"],
            current_block_hash=data.get("current_block_hash"),
            nonce=data.get("nonce", 0)
        )


class Blockchain:
    """
    Permissioned Audit Blockchain Prototype.
    Guarantees tamper-evident traceability for intrusion detection events in EV charging networks.
    """

    GENESIS_PREV_HASH = "0" * 64

    def __init__(self, ledger_file: Path = LEDGER_FILE):
        self.ledger_file = ledger_file
        self.chain: List[Block] = []
        self._load_or_initialize_chain()

    def _load_or_initialize_chain(self):
        """Loads existing blockchain from JSON ledger or creates Genesis block."""
        if self.ledger_file.exists():
            try:
                with open(self.ledger_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chain = [Block.from_dict(b) for b in data]
                if self.chain:
                    return
            except Exception:
                pass

        # Create Genesis Block
        self.chain = [self._create_genesis_block()]
        self._save_chain()

    def _create_genesis_block(self) -> Block:
        """Initializes the Root Genesis Block for the EVSE Intrusion Detection Network."""
        return Block(
            index=0,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            dataset_filename="GENESIS_ROOT_EVSE_SECURITY_ANCHOR",
            number_of_analysed_records=0,
            normal_traffic_count=0,
            malicious_traffic_count=0,
            attack_wise_counts={},
            previous_block_hash=self.GENESIS_PREV_HASH,
            nonce=100
        )

    def add_detection_block(
        self,
        dataset_filename: str,
        number_of_analysed_records: int,
        normal_traffic_count: int,
        malicious_traffic_count: int,
        attack_wise_counts: Dict[str, int]
    ) -> Block:
        """
        Creates and links a new audit block for a completed detection event.
        """
        latest_block = self.get_latest_block()
        new_index = latest_block.index + 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        new_block = Block(
            index=new_index,
            timestamp=timestamp,
            dataset_filename=dataset_filename,
            number_of_analysed_records=number_of_analysed_records,
            normal_traffic_count=normal_traffic_count,
            malicious_traffic_count=malicious_traffic_count,
            attack_wise_counts=attack_wise_counts,
            previous_block_hash=latest_block.current_block_hash,
            nonce=0
        )

        self.chain.append(new_block)
        self._save_chain()
        return new_block

    def get_latest_block(self) -> Block:
        """Returns the most recent block in the chain."""
        return self.chain[-1]

    def verify_integrity(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validates the integrity of the entire blockchain:
        1. Checks Genesis block integrity
        2. Recalculates current block hash against canonical contents
        3. Validates cryptographic linkage (previous_block_hash == chain[i-1].current_block_hash)
        Returns:
            is_valid: bool
            block_diagnostics: List of dicts describing status of each block
        """
        is_chain_valid = True
        diagnostics = []

        for i, block in enumerate(self.chain):
            recalculated_hash = block.calculate_hash()
            hash_match = (block.current_block_hash == recalculated_hash)

            if i == 0:
                link_valid = (block.previous_block_hash == self.GENESIS_PREV_HASH)
            else:
                prev_block = self.chain[i - 1]
                link_valid = (block.previous_block_hash == prev_block.current_block_hash)

            block_status = hash_match and link_valid
            if not block_status:
                is_chain_valid = False

            diagnostics.append({
                "block_index": block.index,
                "dataset_filename": block.dataset_filename,
                "recorded_hash": block.current_block_hash,
                "recomputed_hash": recalculated_hash,
                "hash_valid": hash_match,
                "link_valid": link_valid,
                "is_intact": block_status,
                "status_message": "VERIFIED & UNTAMPERED" if block_status else "INTEGRITY VIOLATION DETECTED"
            })

        return is_chain_valid, diagnostics

    def simulate_tamper(self, block_index: int, malicious_record_delta: int = 500) -> bool:
        """
        Simulates an unauthorized database alteration to demonstrate the tamper-evident
        property of the cryptographic hash chain.
        """
        if 0 <= block_index < len(self.chain):
            target = self.chain[block_index]
            # Tamper the data without updating the hash
            target.malicious_traffic_count += malicious_record_delta
            target.normal_traffic_count = max(0, target.normal_traffic_count - malicious_record_delta)
            self._save_chain()
            return True
        return False

    def reset_chain(self):
        """Resets the chain to the genesis block."""
        self.chain = [self._create_genesis_block()]
        self._save_chain()

    def _save_chain(self):
        """Persists chain to local JSON storage."""
        data = [b.to_dict() for b in self.chain]
        with open(self.ledger_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

