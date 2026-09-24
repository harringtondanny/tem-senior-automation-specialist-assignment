import os
import re
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# 1. MODULAR INGESTION LAYER
# ==========================================
class CSVConnector:
    """Modular data reader easily replaceable by a Warehouse Connector."""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def read_table(self, filename):
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Target data file '{filename}' missing from '{self.data_dir}'")
        return pd.read_csv(path)

# ==========================================
# 2. SANITIZATION HELPERS
# ==========================================
def clean_name(name):
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    name = re.sub(r'\b(ltd|limited|plc|co|corp|inc|gmbh|uk|holding|holdings)\b', '', name)
    name = re.sub(r'[^a-z0-9\s]', '', name)
    return " ".join(name.split())

def normalize_email(email):
    if pd.isna(email):
        return ""
    email = str(email).strip().lower()
    if email.endswith('.co.uk'):
        email = email[:-6] + '.com'
    return email

# ==========================================
# 3. CORE RUN EXECUTION
# ==========================================
def run_pipeline(input_dir, output_dir, as_of_date_str, run_id):
    start_time = datetime.now()
    as_of_date = pd.to_datetime(as_of_date_str)

    os.makedirs(output_dir, exist_ok=True)
    connector = CSVConnector(input_dir)

    # Ingest source tables
    cust_df = connector.read_table('customers.csv')
    sites_df = connector.read_table('sites_contracts.csv')
    quotes_df = connector.read_table('renewal_quotes.csv')
    service_df = connector.read_table('service_contacts.csv')
    nps_df = connector.read_table('nps_responses.csv')

    # Standardize data quality validations & metrics
    sites_df['mpan_clean'] = sites_df['mpan'].astype(str).str.replace(r'\D', '', regex=True)
    sites_df['is_mpan_valid'] = sites_df['mpan_clean'].str.len() == 13
    quotes_df['mpan_clean'] = quotes_df['mpan'].astype(str).str.replace(r'\D', '', regex=True)

    total_mpans = len(sites_df)
    invalid_mpan_count = int((~sites_df['is_mpan_valid']).sum())

    # Extract clean keys for waterfalls
    cust_df['contact_email_clean'] = cust_df['contact_email'].astype(str).str.strip().str.lower()
    cust_df['contact_email_norm'] = cust_df['contact_email_clean'].apply(normalize_email)
    cust_df['company_name_clean'] = cust_df['company_name'].astype(str).apply(clean_name)

    email_map = cust_df.set_index('contact_email_norm')['customer_id'].to_dict()
    name_map = cust_df.set_index('company_name_clean')['customer_id'].to_dict()

    # Map service contacts
    service_df['raised_by_clean'] = service_df['raised_by'].astype(str).str.strip().str.lower()
    service_df['raised_by_norm_email'] = service_df['raised_by_clean'].apply(normalize_email)
    service_df['raised_by_norm_name'] = service_df['raised_by_clean'].apply(clean_name)

    service_df['customer_id'] = np.nan
    service_df['customer_id'] = service_df['customer_id'].fillna(service_df['raised_by_norm_email'].map(email_map))
    service_df['customer_id'] = service_df['customer_id'].fillna(service_df['raised_by_norm_name'].map(name_map))

    cust_service_stats = service_df.groupby('customer_id').agg(
        avg_csat=('csat_1_5', 'mean'),
        ticket_count=('ticket_id', 'count')
    ).reset_index()

    # Map NPS surveys
    nps_df['respondent_email_norm'] = nps_df['respondent_email'].astype(str).str.strip().str.lower().apply(normalize_email)
    nps_df['customer_id'] = nps_df['respondent_email_norm'].map(email_map)

    cust_nps_stats = nps_df.groupby('customer_id').agg(
        avg_nps=('score_0_10', 'mean')
    ).reset_index()

    # Match Quote rate hikes
    sites_quotes_temp = sites_df.merge(quotes_df, on='mpan_clean', how='inner')
    sites_quotes_temp['rate_increase_pct'] = ((sites_quotes_temp['quote_rate_p_per_kwh'] - sites_quotes_temp['current_unit_rate_p_per_kwh']) / sites_quotes_temp['current_unit_rate_p_per_kwh']) * 100
    cust_pricing_risk = sites_quotes_temp.groupby('account_ref')['rate_increase_pct'].mean().reset_index()

    # Active customer risk calculation
    predictive_df = cust_df[cust_df['status'] == 'Active'].copy()
    predictive_df = predictive_df.merge(cust_nps_stats, on='customer_id', how='left')
    predictive_df = predictive_df.merge(cust_service_stats, on='customer_id', how='left')
    predictive_df['ticket_count'] = predictive_df['ticket_count'].fillna(0)
    predictive_df = predictive_df.merge(cust_pricing_risk, on='account_ref', how='left')
    predictive_df['rate_increase_pct'] = predictive_df['rate_increase_pct'].fillna(0)

    predictive_df['nps_risk_points'] = np.where(predictive_df['avg_nps'] <= 6, 35, 0)
    predictive_df['csat_risk_points'] = np.where(predictive_df['avg_csat'] <= 2, 25, 0)
    predictive_df['ticket_risk_points'] = np.where(predictive_df['ticket_count'] > 2, 15, 0)
    predictive_df['pricing_risk_points'] = np.where(predictive_df['rate_increase_pct'] > 20, 25, 0)

    predictive_df['churn_risk_score'] = (
        predictive_df['nps_risk_points'] +
        predictive_df['csat_risk_points'] +
        predictive_df['ticket_risk_points'] +
        predictive_df['pricing_risk_points']
    )
    predictive_df['is_high_risk'] = predictive_df['churn_risk_score'] >= 50

    # Operational renewal windowing
    sites_df['contract_end_dt'] = pd.to_datetime(sites_df['contract_end'], errors='coerce')
    sites_df['days_to_renewal'] = (sites_df['contract_end_dt'] - as_of_date).dt.days

    cust_renewal_info = sites_df.groupby('account_ref').agg(
        min_days_to_renewal=('days_to_renewal', 'min'),
        has_invalid_mpan=('is_mpan_valid', lambda x: (~x).any())
    ).reset_index()

    final_decision_register = predictive_df.merge(cust_renewal_info, on='account_ref', how='left')

    # Incorporate "Likely to Sign" promotion logic
    final_decision_register['standard_renewal_scope'] = final_decision_register['min_days_to_renewal'] <= 90
    final_decision_register['avg_nps'] = final_decision_register['avg_nps'].fillna(0)

    final_decision_register['likely_to_sign_trigger'] = (
        (final_decision_register['avg_nps'] >= 8) &
        (final_decision_register['churn_risk_score'] < 50) &
        (~final_decision_register['has_invalid_mpan'])
    )

    # Define operational 'in_renewal_scope' using standard window or promoted promoters
    final_decision_register['in_renewal_scope'] = final_decision_register['standard_renewal_scope'] | final_decision_register['likely_to_sign_trigger']

    def determine_action(row):
        # 1. If NOT in renewal scope (standard or promoted), route appropriately
        if not row['in_renewal_scope']:
            return 'nurture' if row['is_high_risk'] else 'no action'
        # 2. If inside renewal scope, verify data completeness & risk thresholds
        if row['is_high_risk'] or row['has_invalid_mpan']:
            return 'human review'
        return 'auto-attempt'

    final_decision_register['target_action'] = final_decision_register.apply(determine_action, axis=1)

    # File 1: Save decision register
    output_cols = [
        'customer_id', 'company_name', 'account_ref', 'industry',
        'churn_risk_score', 'is_high_risk', 'has_invalid_mpan',
        'min_days_to_renewal', 'in_renewal_scope', 'target_action'
    ]
    final_decision_register[output_cols].to_csv(os.path.join(output_dir, 'decision_register.csv'), index=False)

    # File 2: Capture Match Exceptions
    unmatched_service = service_df[service_df['customer_id'].isna()]
    unmatched_nps = nps_df[nps_df['customer_id'].isna()]

    exceptions_list = []
    for _, row in unmatched_service.iterrows():
        exceptions_list.append({
            'entity_type': 'service_ticket',
            'entity_id': row['ticket_id'],
            'raw_reference': row['raised_by'],
            'failure_reason': 'Could not match email or business name to active customer record'
        })
    for _, row in unmatched_nps.iterrows():
        exceptions_list.append({
            'entity_type': 'nps_survey',
            'entity_id': row['response_id'],
            'raw_reference': row['respondent_email'],
            'failure_reason': 'Email mismatch or missing customer reference'
        })

    match_exceptions_df = pd.DataFrame(exceptions_list if exceptions_list else [
        {'entity_type': 'None', 'entity_id': 'None', 'raw_reference': 'None', 'failure_reason': 'None'}
    ])
    match_exceptions_df.to_csv(os.path.join(output_dir, 'match_exceptions.csv'), index=False)

    # File 3: Data Quality Report JSON
    data_quality_report = {
        "mpan_completeness_pct": round(((total_mpans - invalid_mpan_count) / total_mpans) * 100, 2) if total_mpans > 0 else 0,
        "invalid_mpans_count": invalid_mpan_count,
        "total_mpans_processed": total_mpans,
        "unmatched_service_tickets": len(unmatched_service),
        "unmatched_nps_surveys": len(unmatched_nps)
    }
    with open(os.path.join(output_dir, 'data_quality_report.json'), 'w') as f:
        json.dump(data_quality_report, f, indent=2)

    # File 4: Run Summary JSON
    duration = (datetime.now() - start_time).total_seconds()
    run_summary = {
        "run_id": run_id,
        "as_of_date": as_of_date_str,
        "execution_duration_seconds": round(duration, 2),
        "total_customers_processed": len(predictive_df),
        "target_action_breakdown": final_decision_register['target_action'].value_counts().to_dict()
    }
    with open(os.path.join(output_dir, 'run_summary.json'), 'w') as f:
        json.dump(run_summary, f, indent=2)

    print(f"Pipeline execution completed successfully for Run {run_id}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Matching and Renewal CLI Automation")
    parser.add_argument('--input-dir', default='./raw_data', help='Directory containing the raw CSV files')
    parser.add_argument('--output-dir', default='.', help='Directory where output files will be written')
    parser.add_argument('--as-of-date', default='2026-09-01', help='Reference date to determine contract end window scope')
    parser.add_argument('--run-id', default='RUN-MOCK-001', help='Unique Identifier for the workflow execution run')

    args = parser.parse_args()
    run_pipeline(args.input_dir, args.output_dir, args.as_of_date, args.run_id)
