"""
data_loader.py  –  Parse the bocdata 1 PostgreSQL dump file into pandas DataFrames.
All processing is done in-memory from the raw .sql dump.
"""

import re
import io
import pandas as pd
import streamlit as st
from pathlib import Path

# ── Path to the dump file (same folder as this script's parent) ──────────────
DATA_FILE = Path(__file__).parent.parent / "bocdata 1"


def _extract_table(raw: str, table_name: str, columns: list[str]) -> pd.DataFrame:
    """
    Pull COPY … FROM stdin block for *table_name* out of the dump text and
    return it as a DataFrame.
    """
    # Match the COPY block header
    pattern = (
        rf"COPY public\.\"{re.escape(table_name)}\""  # quoted table
        r"[^\n]*\n"                                   # rest of header line
        r"(.*?)\n\\\.(?:\n|$)"                        # data rows → group 1
    )
    # Try quoted form first, then unquoted
    m = re.search(pattern, raw, re.DOTALL)
    if not m:
        pattern2 = (
            rf"COPY public\.{re.escape(table_name)}"
            r"[^\n]*\n"
            r"(.*?)\n\\\.(?:\n|$)"
        )
        m = re.search(pattern2, raw, re.DOTALL)
    if not m:
        return pd.DataFrame(columns=columns)

    block = m.group(1).strip()
    if not block:
        return pd.DataFrame(columns=columns)

    rows = []
    for line in block.splitlines():
        parts = line.split("\t")
        # Pad / truncate to expected column count
        while len(parts) < len(columns):
            parts.append(None)
        parts = parts[: len(columns)]
        # Replace PostgreSQL null markers
        parts = [None if p == "\\N" else p for p in parts]
        rows.append(parts)

    df = pd.DataFrame(rows, columns=columns)
    return df


@st.cache_resource(show_spinner="Loading BOC database dump…", ttl=3600)
def load_all() -> dict[str, pd.DataFrame]:
    """Load and return all relevant tables as DataFrames."""
    raw = DATA_FILE.read_text(encoding="utf-8", errors="replace")

    # ── users ────────────────────────────────────────────────────────────────
    users = _extract_table(
        raw, "user",
        ["id","email","emailVerified","name","image","createdAt","updatedAt",
         "did","didStatus","didCreationTxId","rewardBalance","role",
         "firstName","lastName","phone","dateOfBirth","address",
         "mainWalletAddress","mainWalletSavedAt","referralBonusGranted",
         "referralCode","referredBy","termsAccepted","lifetimeRewardPoints"]
    )
    for col in ["createdAt","updatedAt","dateOfBirth","mainWalletSavedAt"]:
        if col in users.columns:
            users[col] = pd.to_datetime(users[col], errors="coerce")
    for col in ["rewardBalance","lifetimeRewardPoints"]:
        if col in users.columns:
            users[col] = pd.to_numeric(users[col], errors="coerce").fillna(0)

    # ── bills ────────────────────────────────────────────────────────────────
    bills = _extract_table(
        raw, "bill",
        ["id","userId","s3Key","originalFilename","contentType","fileSizeBytes",
         "sha256","phash","category","status","failureReason","s3DeletedAt",
         "createdAt","updatedAt"]
    )
    for col in ["createdAt","updatedAt","s3DeletedAt"]:
        if col in bills.columns:
            bills[col] = pd.to_datetime(bills[col], errors="coerce")
    if "fileSizeBytes" in bills.columns:
        bills["fileSizeBytes"] = pd.to_numeric(bills["fileSizeBytes"], errors="coerce")

    # ── bill_extraction ──────────────────────────────────────────────────────
    extractions = _extract_table(
        raw, "bill_extraction",
        ["id","billId","merchantName","invoiceNumber","invoiceDate","currency",
         "totalAmount","taxAmount","category","taxRegNumber","lineItems",
         "confidenceScores","modelUsed","rawResponse","textHash","createdAt"]
    )
    for col in ["invoiceDate","createdAt"]:
        if col in extractions.columns:
            extractions[col] = pd.to_datetime(extractions[col], errors="coerce")
    for col in ["totalAmount","taxAmount"]:
        if col in extractions.columns:
            extractions[col] = pd.to_numeric(extractions[col], errors="coerce")

    # ── fraud_check ──────────────────────────────────────────────────────────
    fraud = _extract_table(
        raw, "fraud_check",
        ["id","billId","vendor","truthscanJobId","score","finalResult",
         "decision","previewUrl","flags","rawResponse","createdAt"]
    )
    if "createdAt" in fraud.columns:
        fraud["createdAt"] = pd.to_datetime(fraud["createdAt"], errors="coerce")
    if "score" in fraud.columns:
        fraud["score"] = pd.to_numeric(fraud["score"], errors="coerce")

    # ── nft ──────────────────────────────────────────────────────────────────
    nfts = _extract_table(
        raw, "nft",
        ["id","billId","userId","tokenId","serialNumber","artworkUri",
         "metadataUri","mintTxId","transferredAt","transferTxId",
         "recipientAccountId","createdAt"]
    )
    for col in ["transferredAt","createdAt"]:
        if col in nfts.columns:
            nfts[col] = pd.to_datetime(nfts[col], errors="coerce")
    if "serialNumber" in nfts.columns:
        nfts["serialNumber"] = pd.to_numeric(nfts["serialNumber"], errors="coerce")

    # ── reward_credit ────────────────────────────────────────────────────────
    reward_credits = _extract_table(
        raw, "reward_credit",
        ["id","billId","userId","amount","createdAt"]
    )
    if "createdAt" in reward_credits.columns:
        reward_credits["createdAt"] = pd.to_datetime(reward_credits["createdAt"], errors="coerce")
    if "amount" in reward_credits.columns:
        reward_credits["amount"] = pd.to_numeric(reward_credits["amount"], errors="coerce")

    # ── reward_claim ─────────────────────────────────────────────────────────
    reward_claims = _extract_table(
        raw, "reward_claim",
        ["id","userId","walletAddress","amount","status","createdAt",
         "transactionHash","rejectionReason"]
    )
    if "createdAt" in reward_claims.columns:
        reward_claims["createdAt"] = pd.to_datetime(reward_claims["createdAt"], errors="coerce")
    if "amount" in reward_claims.columns:
        reward_claims["amount"] = pd.to_numeric(reward_claims["amount"], errors="coerce")

    # ── reward_transaction ───────────────────────────────────────────────────
    reward_txns = _extract_table(
        raw, "reward_transaction",
        ["id","userId","walletAccountId","amount","transferTxId",
         "triggeredBy","createdAt"]
    )
    if "createdAt" in reward_txns.columns:
        reward_txns["createdAt"] = pd.to_datetime(reward_txns["createdAt"], errors="coerce")
    if "amount" in reward_txns.columns:
        reward_txns["amount"] = pd.to_numeric(reward_txns["amount"], errors="coerce")

    # ── mint_attempt ─────────────────────────────────────────────────────────
    mint_attempts = _extract_table(
        raw, "mint_attempt",
        ["id","billId","txId","status","createdAt","serialNumber"]
    )
    if "createdAt" in mint_attempts.columns:
        mint_attempts["createdAt"] = pd.to_datetime(mint_attempts["createdAt"], errors="coerce")

    # ── leaderboard_snapshot_entry ───────────────────────────────────────────
    leaderboard = _extract_table(
        raw, "leaderboard_snapshot_entry",
        ["id","snapshotId","rank","userId","userName","userEmail","points"]
    )
    if "rank" in leaderboard.columns:
        leaderboard["rank"] = pd.to_numeric(leaderboard["rank"], errors="coerce")
    if "points" in leaderboard.columns:
        leaderboard["points"] = pd.to_numeric(leaderboard["points"], errors="coerce")

    # ── wallet_connection ────────────────────────────────────────────────────
    wallets = _extract_table(
        raw, "wallet_connection",
        ["id","userId","accountId","publicKey","nftTokenAssociated",
         "rewardTokenAssociated","status","verifiedAt","createdAt","updatedAt"]
    )
    for col in ["verifiedAt","createdAt","updatedAt"]:
        if col in wallets.columns:
            wallets[col] = pd.to_datetime(wallets[col], errors="coerce")

    # ── audit_log ────────────────────────────────────────────────────────────
    audit = _extract_table(
        raw, "audit_log",
        ["id","userId","eventType","metadata","ipAddress","createdAt"]
    )
    if "createdAt" in audit.columns:
        audit["createdAt"] = pd.to_datetime(audit["createdAt"], errors="coerce")

    return {
        "users": users,
        "bills": bills,
        "extractions": extractions,
        "fraud": fraud,
        "nfts": nfts,
        "reward_credits": reward_credits,
        "reward_claims": reward_claims,
        "reward_txns": reward_txns,
        "mint_attempts": mint_attempts,
        "leaderboard": leaderboard,
        "wallets": wallets,
        "audit": audit,
    }
