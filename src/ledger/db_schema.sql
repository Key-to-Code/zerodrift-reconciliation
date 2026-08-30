-- Layer 3: PostgreSQL double-entry ledger schema.
-- Accounts are classified from the MERCHANT's books, not the gateway's or the bank's.

CREATE TYPE entry_status AS ENUM ('posted', 'reversed');
CREATE TYPE account_type AS ENUM ('asset', 'liability', 'revenue', 'expense', 'suspense');
CREATE TYPE match_status AS ENUM ('fast_path', 'agent_resolved', 'honest_exception');

CREATE TABLE accounts (
    account_id SERIAL PRIMARY KEY,
    account_code VARCHAR(50) UNIQUE NOT NULL,
    account_name VARCHAR(200) NOT NULL,
    account_type account_type NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE journal_entries (
    entry_id BIGSERIAL PRIMARY KEY,
    batch_run_id UUID NOT NULL,
    idempotency_key VARCHAR(120) UNIQUE NOT NULL,
    reference_id VARCHAR(100) NOT NULL,
    description TEXT,
    status entry_status NOT NULL DEFAULT 'posted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE journal_lines (
    line_id BIGSERIAL PRIMARY KEY,
    entry_id BIGINT NOT NULL REFERENCES journal_entries(entry_id),
    account_id INT NOT NULL REFERENCES accounts(account_id),
    direction CHAR(1) NOT NULL CHECK (direction IN ('D','C')),
    amount NUMERIC(14,2) NOT NULL CHECK (amount > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE reconciliation_matches (
    match_id BIGSERIAL PRIMARY KEY,
    batch_run_id UUID NOT NULL,
    order_id VARCHAR(100),
    payment_id VARCHAR(100),
    utr VARCHAR(100),
    status match_status NOT NULL,
    confidence_note TEXT,
    journal_entry_id BIGINT REFERENCES journal_entries(entry_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (batch_run_id, order_id)
);

CREATE INDEX idx_journal_lines_entry ON journal_lines(entry_id);
CREATE INDEX idx_journal_entries_reference ON journal_entries(reference_id);
CREATE INDEX idx_journal_entries_run ON journal_entries(batch_run_id);
CREATE INDEX idx_recon_order ON reconciliation_matches(order_id);
CREATE INDEX idx_recon_utr ON reconciliation_matches(utr);
CREATE INDEX idx_recon_run ON reconciliation_matches(batch_run_id);

CREATE OR REPLACE FUNCTION check_entry_balances() RETURNS TRIGGER AS $$
DECLARE
    target_entry_id BIGINT;
    debit_total NUMERIC(14,2);
    credit_total NUMERIC(14,2);
BEGIN
    target_entry_id := COALESCE(NEW.entry_id, OLD.entry_id);
    SELECT COALESCE(SUM(amount) FILTER (WHERE direction='D'), 0),
           COALESCE(SUM(amount) FILTER (WHERE direction='C'), 0)
    INTO debit_total, credit_total
    FROM journal_lines WHERE entry_id = target_entry_id;

    IF debit_total <= 0 OR credit_total <= 0 THEN
        RAISE EXCEPTION 'Journal entry % has non-positive balance totals (Debits: %, Credits: %)',
            target_entry_id, debit_total, credit_total;
    END IF;
    IF debit_total != credit_total THEN
        RAISE EXCEPTION 'Unbalanced journal entry %: debits % != credits %',
            target_entry_id, debit_total, credit_total;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE CONSTRAINT TRIGGER trg_check_balance
    AFTER INSERT OR UPDATE OR DELETE ON journal_lines
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION check_entry_balances();

-- Accounts are classified from the MERCHANT's books, not Razorpay's.
-- GST paid on the MDR fee is Input Tax Credit (an asset) -- not a liability.
-- TDS withheld under Section 194-O (0.1%) is an advance tax asset -- not a liability.
INSERT INTO accounts (account_code, account_name, account_type) VALUES
    ('CASH', 'Cash / Bank (Nodal Escrow)', 'asset'),
    ('CASH_IN_TRANSIT_UTR', 'Cash In Transit -- UTR Batch Clearing', 'asset'),
    ('AR_GATEWAY_CLEARING', 'Gateway Clearing Receivable', 'asset'),
    ('REVENUE_GROSS', 'Merchant Gross Sales', 'revenue'),
    ('MDR_EXPENSE', 'Payment Processing Fee (MDR)', 'expense'),
    ('GST_ITC_RECEIVABLE', 'GST Input Tax Credit on MDR', 'asset'),
    ('TDS_194O_CREDIT', 'TDS Withheld Asset (Section 194-O, 0.1%)', 'asset'),
    ('ROUNDING_DIFFERENCE', 'UTR Batch Allocation Rounding', 'expense'),
    ('SUSPENSE_UNRESOLVED', 'Unresolved Reconciliation Suspense', 'suspense');
