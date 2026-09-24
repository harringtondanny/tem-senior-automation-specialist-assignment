import os
import pandas as pd
import numpy as np

def create_downstream_outputs(register_path, raw_data_dir, output_dir):
    # 1. Read existing decision register from production location
    df = pd.read_csv(register_path)

    # 2. Ingest necessary raw tables to resolve descriptive fields
    cust_df = pd.read_csv(os.path.join(raw_data_dir, 'customers.csv'))
    brokers_df = pd.read_csv(os.path.join(raw_data_dir, 'brokers.csv'))
    sites_df = pd.read_csv(os.path.join(raw_data_dir, 'sites_contracts.csv'))
    quotes_df = pd.read_csv(os.path.join(raw_data_dir, 'renewal_quotes.csv'))

    # Pre-clean MPAN to link quotes to customers via sites
    sites_df['mpan_clean'] = sites_df['mpan'].astype(str).str.replace(r'\D', '', regex=True)
    quotes_df['mpan_clean'] = quotes_df['mpan'].astype(str).str.replace(r'\D', '', regex=True)

    # Establish maps
    broker_map = brokers_df.set_index('broker_id')['broker_name'].to_dict()
    cust_broker_id_map = cust_df.set_index('customer_id')['broker_id'].to_dict()

    # Resolve quote reference
    sites_quotes = sites_df.merge(quotes_df, on='mpan_clean', how='inner')
    cust_quote_map = sites_quotes.groupby('account_ref')['quote_id'].first().to_dict()

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 3. Write human_review_queue.csv
    # Target: Customers in-scope but marked for manual intervention
    hr_mask = (df['in_renewal_scope'] == True) & (df['target_action'] == 'human review')
    hr_df = df[hr_mask].copy()

    hr_output = pd.DataFrame()
    if not hr_df.empty:
        hr_output['customer'] = hr_df['company_name']
        hr_output['trigger'] = np.where(hr_df['min_days_to_renewal'] <= 90, 'Standard 90-day Renewal Window', 'Likely to Sign Promoter Target')

        # Explain reason
        reasons = []
        for _, r in hr_df.iterrows():
            reasons_list = []
            if r['is_high_risk']:
                reasons_list.append('High Predictive Churn Risk')
            if r['has_invalid_mpan']:
                reasons_list.append('Grid Registration Failure (Invalid MPAN Format)')
            reasons.append(' & '.join(reasons_list))
        hr_output['reason'] = reasons

        hr_output['risk_factors'] = 'Churn Risk Score: ' + hr_df['churn_risk_score'].astype(str) + '/100'
        hr_output['broker'] = hr_df['customer_id'].map(cust_broker_id_map).map(broker_map).fillna('Direct (No Broker)')
        hr_output['quote_reference'] = hr_df['account_ref'].map(cust_quote_map).fillna('Needs Recalculation')
        hr_output['next_step'] = np.where(hr_df['has_invalid_mpan'], 'Route to Data Operations for Grid Clean-up', 'Route to Dedicated Account Manager for Bespoke Negotiation')
    else:
        hr_output = pd.DataFrame(columns=['customer', 'trigger', 'reason', 'risk_factors', 'broker', 'quote_reference', 'next_step'])

    hr_output.to_csv(os.path.join(output_dir, 'human_review_queue.csv'), index=False)

    # 4. Write renewal_attempts.csv
    # Target: Rows already marked as auto-attempt
    auto_df = df[df['target_action'] == 'auto-attempt'].copy()
    auto_output = pd.DataFrame()
    if not auto_df.empty:
        auto_output['customer'] = auto_df['company_name']
        auto_output['quote_reference'] = auto_df['account_ref'].map(cust_quote_map).fillna('Automatic Quote Generated')
        auto_output['pricing_hike_status'] = 'Risk Score: ' + auto_df['churn_risk_score'].astype(str)
        auto_output['status'] = 'Scheduled for Automatic Renewal Execution'
    else:
        auto_output = pd.DataFrame(columns=['customer', 'quote_reference', 'pricing_hike_status', 'status'])

    auto_output.to_csv(os.path.join(output_dir, 'renewal_attempts.csv'), index=False)
    print('Downstream action queues populated successfully in production output directory.')

if __name__ == "__main__":
    create_downstream_outputs(
        register_path='output/decision_register.csv',
        raw_data_dir='./raw_data',
        output_dir='output'
    )
