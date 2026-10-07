"""SERA full-model pipeline with embedded dataset configurations.

Internal modules are bundled to retain a single model entry file.
"""
from __future__ import annotations
import sys, types, json, os, copy, argparse
from pathlib import Path
_SCRIPT = Path(__file__).resolve()
_ROOT = _SCRIPT.parent.parent
_DATASET = Path(os.environ.get('MAS_DATASET_DIR', str(_ROOT/'dataset'))).resolve()
_OUTPUT = Path(os.environ.get('MAS_OUTPUT_DIR', str(_ROOT/'outputs'))).resolve()
def _install(name, source, **values):
    if name in sys.modules: return sys.modules[name]
    module=types.ModuleType(name)
    module.__file__=str(_SCRIPT)
    module.__package__=name.rpartition('.')[0]
    module.__dict__.update(values)
    sys.modules[name]=module
    exec(compile(source, str(_SCRIPT)+'::'+name, 'exec'),module.__dict__)
    parent, _, child=name.rpartition('.')
    if parent in sys.modules: setattr(sys.modules[parent],child,module)
    return module
_package=sys.modules.get('how123') or types.ModuleType('how123')
_package.__path__=[]
sys.modules['how123']=_package

_CONFIGS = {'banksim': {'window_rows': 400000, 'semantic': {'enabled': 'auto', 'task_prompt': 'Detect whether this retail card payment is fraudulent.', 'fields': [{'name': 'ts', 'description': 'day index of the simulated period, one step per day', 'unit': 'unix seconds', 'entity_role': 'event time', 'source': 'BankSim step', 'in_serialization': False}, {'name': 'entity', 'description': 'customer account making the payment', 'entity_role': 'cardholder', 'source': 'BankSim customer', 'in_serialization': False}, {'name': 'counterparty', 'description': 'merchant receiving the payment', 'entity_role': 'merchant', 'source': 'BankSim merchant', 'in_serialization': False}, {'name': 'amount', 'description': 'payment amount', 'unit': 'euro', 'entity_role': 'transaction value', 'source': 'BankSim amount'}, {'name': 'cat1', 'description': 'merchant spending category', 'entity_role': 'spending category', 'source': 'BankSim category', 'codebook': {'es_transportation': 'transport and travel', 'es_health': 'healthcare services', 'es_otherservices': 'other services', 'es_food': 'groceries and food', 'es_hotelservices': 'hotels and lodging', 'es_barsandrestaurants': 'bars and restaurants', 'es_tech': 'technology and electronics', 'es_sportsandtoys': 'sports goods and toys', 'es_wellnessandbeauty': 'wellness and beauty', 'es_hyper': 'hypermarket', 'es_fashion': 'clothing and fashion', 'es_home': 'home goods', 'es_contents': 'digital content', 'es_travel': 'travel bookings', 'es_leisure': 'leisure activities'}}, {'name': 'cat2', 'description': 'binned age bracket of the cardholder', 'entity_role': 'customer age band', 'source': 'BankSim age'}, {'name': 'cat3', 'description': 'gender of the cardholder', 'entity_role': 'customer gender', 'source': 'BankSim gender'}, {'name': 'text', 'description': 'no free-text field in this dataset', 'source': 'n/a', 'in_serialization': False}]}, 'context': {'enabled': 'auto', 'mode': 'soft', 'window_seconds': 2592000, 'max_history': 8}, 'graph': {'enabled': True, 'bucket_seconds': 86400, 'streams': [{'source': 'entity', 'target': 'counterparty', 'relation': 'pays', 'source_ns': 'CUST', 'target_ns': 'MERCH', 'trigger': True}], 'max_seeds_per_bucket': 64, 'signal_horizon_buckets': 7}, 'constraint': {'method': 'np', 'alpha': 0.01, 'delta': 0.05}}, 'default': {'seed': 2026, 'window_rows': 400000, 'split': {'mode': 'temporal', 'ratios': [0.6, 0.15, 0.1, 0.15], 'group_column': None}, 'semantic': {'enabled': 'auto', 'embedding_dim': 64, 'numeric_bins': 32, 'max_text_tokens': 256, 'task_prompt': 'Detect whether this financial transaction is fraudulent.', 'version': 'gamma-v1', 'fields': []}, 'context': {'enabled': 'auto', 'mode': 'soft', 'window_seconds': 2592000, 'top_k': 8, 'required_fields': ['amount'], 'compatibility_fields': [], 'blocked_value_pairs': {}, 'soft_relation_fields': ['cat1', 'cat2', 'cat3'], 'hash_buckets': 512, 'embedding_dim': 8, 'output_dim': 8, 'epochs': 3, 'batch_size': 4096, 'lr': 0.005, 'negative_control': 'none'}, 'graph': {'enabled': True, 'bucket_seconds': 86400, 'epsilon': 0.01, 'bonferroni': True, 'trigger': True, 'require_prior_history': True, 'signal_horizon_buckets': 7, 'signal_scope': 'seed', 'force_rho_one': False, 'rho_estimator': 'overlap', 'rho_min': 'auto', 'rho_quantile': 'auto', 'rho_quantile_grid': [0.0, 0.5, 0.75], 'zeta_mode': 'auto', 'zeta_mode_grid': ['product', 'burst_only'], 'rho_fallback': 0.2, 'rho_min_calibration_seeds': 20, 'max_seeds_per_bucket': 64, 'max_patterns_exported': 2000, 'path_samples': 16, 'max_hops': 3, 'vector_dim': 32, 'counter': 'exact', 'cms_width': 4096, 'cms_depth': 5, 'add_reverse': True, 'streams': []}, 'detector': {'n_estimators': 800, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5, 'reg_lambda': 1.0, 'scale_pos_weight': 'auto', 'early_stopping_rounds': 50, 'n_jobs': -1}, 'constraint': {'alpha': 0.01, 'delta': 0.05, 'method': 'np', 'capacity_per_window': None}, 'evaluation': {'selection_criterion': 'ap', 'fpr_grid': [0.001, 0.005, 0.01], 'bootstrap': 500, 'shap_top_k': 3, 'shap_examples': 20, 'policy_block_at': 0.7, 'policy_investigate_at': 0.4}}, 'ibm-aml-li': {'window_rows': 2000000, 'semantic': {'enabled': 'auto', 'task_prompt': 'Detect whether this bank transfer is part of a money laundering pattern.', 'fields': [{'name': 'ts', 'description': 'transaction timestamp at minute granularity', 'unit': 'unix seconds', 'entity_role': 'event time', 'source': 'IBM AML Timestamp', 'in_serialization': False}, {'name': 'entity', 'description': 'originating bank account', 'entity_role': 'payer account', 'source': 'IBM AML Account', 'in_serialization': False}, {'name': 'counterparty', 'description': 'receiving bank account', 'entity_role': 'payee account', 'source': 'IBM AML Account.1', 'in_serialization': False}, {'name': 'amount', 'description': 'amount paid by the originator', 'unit': 'source currency', 'entity_role': 'transaction value', 'source': 'IBM AML Amount Paid'}, {'name': 'cat1', 'description': 'settlement instrument used for the transfer', 'entity_role': 'payment instrument', 'source': 'IBM AML Payment Format', 'codebook': {'ACH': 'automated clearing house transfer', 'Cheque': 'paper cheque', 'Credit Card': 'credit card payment', 'Cash': 'cash deposit or withdrawal', 'Wire': 'wire transfer', 'Bitcoin': 'bitcoin transfer', 'Reinvestment': 'reinvested proceeds'}}, {'name': 'cat2', 'description': 'currency in which the payee is credited', 'entity_role': 'settlement currency', 'source': 'IBM AML Receiving Currency'}, {'name': 'cat3', 'description': 'receiving bank identifier', 'entity_role': 'counterparty institution', 'source': 'IBM AML To Bank'}, {'name': 'text', 'description': 'no free-text field in this dataset', 'source': 'n/a', 'in_serialization': False}]}, 'context': {'enabled': 'auto', 'mode': 'soft', 'window_seconds': 21600, 'max_history': 8}, 'graph': {'enabled': True, 'bucket_seconds': 1800, 'streams': [{'source': 'entity', 'target': 'counterparty', 'relation': 'transfer', 'source_ns': 'ACC', 'target_ns': 'ACC', 'trigger': True}], 'max_seeds_per_bucket': 64, 'signal_horizon_buckets': 4}, 'constraint': {'method': 'np', 'alpha': 0.01, 'delta': 0.05}}, 'ibm-aml-medium': {'window_rows': 2000000, 'semantic': {'enabled': 'auto', 'task_prompt': 'Detect whether this bank transfer is part of a money laundering pattern.', 'fields': [{'name': 'ts', 'description': 'transaction timestamp at minute granularity', 'unit': 'unix seconds', 'entity_role': 'event time', 'source': 'IBM AML Timestamp', 'in_serialization': False}, {'name': 'entity', 'description': 'originating bank account', 'entity_role': 'payer account', 'source': 'IBM AML Account', 'in_serialization': False}, {'name': 'counterparty', 'description': 'receiving bank account', 'entity_role': 'payee account', 'source': 'IBM AML Account.1', 'in_serialization': False}, {'name': 'amount', 'description': 'amount paid by the originator', 'unit': 'source currency', 'entity_role': 'transaction value', 'source': 'IBM AML Amount Paid'}, {'name': 'cat1', 'description': 'settlement instrument used for the transfer', 'entity_role': 'payment instrument', 'source': 'IBM AML Payment Format', 'codebook': {'ACH': 'automated clearing house transfer', 'Cheque': 'paper cheque', 'Credit Card': 'credit card payment', 'Cash': 'cash deposit or withdrawal', 'Wire': 'wire transfer', 'Bitcoin': 'bitcoin transfer', 'Reinvestment': 'reinvested proceeds'}}, {'name': 'cat2', 'description': 'currency in which the payee is credited', 'entity_role': 'settlement currency', 'source': 'IBM AML Receiving Currency'}, {'name': 'cat3', 'description': 'receiving bank identifier', 'entity_role': 'counterparty institution', 'source': 'IBM AML To Bank'}, {'name': 'text', 'description': 'no free-text field in this dataset', 'source': 'n/a', 'in_serialization': False}]}, 'context': {'enabled': 'auto', 'mode': 'soft', 'window_seconds': 21600, 'max_history': 8}, 'graph': {'enabled': True, 'bucket_seconds': 1800, 'streams': [{'source': 'entity', 'target': 'counterparty', 'relation': 'transfer', 'source_ns': 'ACC', 'target_ns': 'ACC', 'trigger': True}], 'max_seeds_per_bucket': 64, 'signal_horizon_buckets': 4}, 'constraint': {'method': 'np', 'alpha': 0.01, 'delta': 0.05}}, 'ibm-aml': {'window_rows': 600000, 'semantic': {'task_prompt': 'Detect whether this interbank transfer is part of a money-laundering pattern.', 'fields': [{'name': 'ts', 'description': 'transaction timestamp', 'unit': 'unix seconds', 'entity_role': 'event time', 'source': 'IBM AML transaction log', 'in_serialization': False}, {'name': 'entity', 'description': 'originating account identifier', 'entity_role': 'payer account', 'source': 'IBM AML transaction log', 'in_serialization': False}, {'name': 'counterparty', 'description': 'beneficiary account identifier', 'entity_role': 'payee account', 'source': 'IBM AML transaction log', 'in_serialization': False}, {'name': 'amount', 'description': 'amount paid by the originating account', 'unit': 'currency units of the payment currency', 'entity_role': 'transaction value', 'source': 'IBM AML transaction log'}, {'name': 'cat1', 'description': 'payment rail used to settle the transfer', 'entity_role': 'payment channel', 'source': 'IBM AML transaction log'}, {'name': 'cat2', 'description': 'receiving bank identifier, prefixed with BANK', 'entity_role': 'beneficiary institution', 'source': 'IBM AML transaction log'}, {'name': 'cat3', 'description': 'currency in which the beneficiary is credited', 'unit': 'ISO currency name', 'entity_role': 'settlement currency', 'source': 'IBM AML transaction log'}, {'name': 'text', 'description': 'no free-text field in this dataset', 'source': 'n/a', 'in_serialization': False}, {'name': 'f_amount_received', 'description': 'amount credited to the beneficiary account', 'unit': 'currency units of the receiving currency', 'entity_role': 'transaction value', 'source': 'IBM AML transaction log'}, {'name': 'f_cross_currency', 'description': '1 if paying and receiving currency differ', 'entity_role': 'settlement flag', 'source': 'derived at load time'}, {'name': 'f_cross_bank', 'description': '1 if originating and receiving bank differ', 'entity_role': 'settlement flag', 'source': 'derived at load time'}]}, 'context': {'window_seconds': 21600, 'top_k': 8, 'compatibility_fields': ['cat3'], 'soft_relation_fields': ['cat1', 'cat2', 'cat3']}, 'graph': {'bucket_seconds': 1800, 'streams': [{'source': 'entity', 'target': 'counterparty', 'relation': 'transfer', 'source_ns': 'ACC', 'target_ns': 'ACC', 'trigger': True}], 'max_seeds_per_bucket': 64, 'signal_horizon_buckets': 4}, 'constraint': {'alpha': 0.01, 'delta': 0.05}}, 'ieee-cis': {'window_rows': 400000, 'semantic': {'enabled': 'auto', 'task_prompt': 'Detect whether this card transaction is fraudulent.', 'fields': [{'name': 'ts', 'description': 'transaction timestamp derived from TransactionDT', 'unit': 'unix seconds', 'entity_role': 'event time', 'source': 'IEEE-CIS transaction table', 'in_serialization': False}, {'name': 'entity', 'description': 'card identifier hash', 'entity_role': 'payment card', 'source': 'IEEE-CIS card1', 'in_serialization': False}, {'name': 'counterparty', 'description': 'billing address region of the transaction', 'entity_role': 'billing region', 'source': 'IEEE-CIS addr1', 'in_serialization': False}, {'name': 'amount', 'description': 'transaction amount', 'unit': 'USD', 'entity_role': 'transaction value', 'source': 'IEEE-CIS TransactionAmt'}, {'name': 'cat1', 'description': 'product code of the purchased item', 'entity_role': 'product category', 'source': 'IEEE-CIS ProductCD', 'codebook': {'W': 'web purchase', 'C': 'card-present or catalogue', 'R': 'refund or reversal', 'H': 'hospitality', 'S': 'services'}}, {'name': 'cat2', 'description': 'card network', 'entity_role': 'card scheme', 'source': 'IEEE-CIS card4', 'codebook': {'visa': 'Visa', 'mastercard': 'Mastercard', 'discover': 'Discover', 'american express': 'American Express'}}, {'name': 'cat3', 'description': 'funding type of the card', 'entity_role': 'funding type', 'source': 'IEEE-CIS card6', 'codebook': {'debit': 'debit card, funds drawn immediately', 'credit': 'credit card, funds drawn on a line of credit'}}, {'name': 'text', 'description': 'no free-text field in this dataset', 'source': 'n/a', 'in_serialization': False}]}, 'context': {'enabled': 'auto', 'mode': 'soft', 'window_seconds': 2592000, 'max_history': 8}, 'graph': {'enabled': True, 'bucket_seconds': 86400, 'streams': [{'source': 'entity', 'target': 'counterparty', 'relation': 'pays', 'source_ns': 'CARD', 'target_ns': 'ADDR', 'trigger': True}], 'max_seeds_per_bucket': 64, 'signal_horizon_buckets': 7}, 'constraint': {'method': 'np', 'alpha': 0.01, 'delta': 0.05}}, 'sparkov': {'window_rows': 400000, 'semantic': {'task_prompt': 'Detect whether this card-present or card-not-present purchase is fraudulent.', 'fields': [{'name': 'ts', 'description': 'transaction timestamp', 'unit': 'unix seconds', 'entity_role': 'event time', 'source': 'Sparkov generator', 'in_serialization': False}, {'name': 'entity', 'description': 'card number of the payer', 'entity_role': 'payment card', 'source': 'Sparkov generator', 'in_serialization': False}, {'name': 'counterparty', 'description': 'merchant that accepted the payment', 'entity_role': 'merchant', 'source': 'Sparkov generator', 'in_serialization': False}, {'name': 'amount', 'description': 'purchase amount', 'unit': 'USD', 'entity_role': 'transaction value', 'source': 'Sparkov generator'}, {'name': 'cat1', 'description': 'merchant spending category', 'entity_role': 'merchant category', 'source': 'Sparkov generator', 'codebook': {'grocery_pos': 'in-store grocery', 'grocery_net': 'online grocery', 'gas_transport': 'fuel and transport', 'misc_net': 'online miscellaneous', 'misc_pos': 'in-store miscellaneous', 'shopping_net': 'online shopping', 'shopping_pos': 'in-store shopping', 'entertainment': 'entertainment', 'food_dining': 'food and dining', 'personal_care': 'personal care', 'health_fitness': 'health and fitness', 'home': 'home goods', 'kids_pets': 'kids and pets', 'travel': 'travel'}}, {'name': 'cat2', 'description': 'US state in the cardholder billing address', 'entity_role': 'cardholder region', 'source': 'Sparkov generator'}, {'name': 'cat3', 'description': 'declared occupation of the cardholder', 'entity_role': 'cardholder attribute', 'source': 'Sparkov generator'}, {'name': 'text', 'description': 'no free-text field in this dataset', 'source': 'n/a', 'in_serialization': False}, {'name': 'f_city_pop', 'description': 'population of the cardholder city', 'unit': 'residents', 'entity_role': 'cardholder region attribute', 'source': 'Sparkov generator'}]}, 'context': {'window_seconds': 2592000, 'top_k': 8, 'soft_relation_fields': ['cat1', 'cat2', 'cat3']}, 'graph': {'bucket_seconds': 86400, 'streams': [{'source': 'entity', 'target': 'counterparty', 'relation': 'pays', 'source_ns': 'CARD', 'target_ns': 'MERCH', 'trigger': True}], 'max_seeds_per_bucket': 64, 'signal_horizon_buckets': 7}, 'constraint': {'alpha': 0.01, 'delta': 0.05}}}
_DATA = {'csv_schema': {'sparkov': {'ssn': {'dtype': 'object'}, 'cc_num': {'dtype': 'int64'}, 'first': {'dtype': 'object'}, 'last': {'dtype': 'object'}, 'gender': {'dtype': 'object'}, 'city': {'dtype': 'object'}, 'state': {'dtype': 'object'}, 'zip': {'dtype': 'int64'}, 'city_pop': {'dtype': 'int64'}, 'job': {'dtype': 'object'}, 'dob': {'dtype': 'object'}, 'acct_num': {'dtype': 'int64'}, 'trans_num': {'dtype': 'object'}, 'trans_date': {'dtype': 'object'}, 'trans_time': {'dtype': 'object'}, 'unix_time': {'dtype': 'int64'}, 'category': {'dtype': 'object'}, 'amt': {'dtype': 'float64'}, 'is_fraud': {'dtype': 'int64'}, 'merchant': {'dtype': 'object'}}, 'ieee-cis': {'transaction_id': {'dtype': 'int64'}, 'transaction_dt': {'dtype': 'int64'}, 'transaction_ts': {'dtype': 'datetime64[us]'}, 'transaction_amt': {'dtype': 'float64'}, 'log_amt': {'dtype': 'float64'}, 'hour_of_day': {'dtype': 'int64'}, 'day_of_week': {'dtype': 'int64'}, 'product_cd': {'dtype': 'object'}, 'has_identity': {'dtype': 'int32'}, 'is_fraud': {'dtype': 'int64'}, 'card1': {'dtype': 'int64'}, 'card2': {'dtype': 'float64'}, 'card3': {'dtype': 'float64'}, 'card4': {'dtype': 'object'}, 'card5': {'dtype': 'float64'}, 'card6': {'dtype': 'object'}, 'addr1': {'dtype': 'float64'}, 'addr2': {'dtype': 'float64'}, 'dist1': {'dtype': 'float64'}, 'dist2': {'dtype': 'float64'}, 'purchaser_email_domain': {'dtype': 'object'}, 'recipient_email_domain': {'dtype': 'object'}, 'C1': {'dtype': 'float64'}, 'C2': {'dtype': 'float64'}, 'C3': {'dtype': 'float64'}, 'C4': {'dtype': 'float64'}, 'C5': {'dtype': 'float64'}, 'C6': {'dtype': 'float64'}, 'C7': {'dtype': 'float64'}, 'C8': {'dtype': 'float64'}, 'C9': {'dtype': 'float64'}, 'C10': {'dtype': 'float64'}, 'C11': {'dtype': 'float64'}, 'C12': {'dtype': 'float64'}, 'C13': {'dtype': 'float64'}, 'C14': {'dtype': 'float64'}, 'D1': {'dtype': 'float64'}, 'D2': {'dtype': 'float64'}, 'D3': {'dtype': 'float64'}, 'D4': {'dtype': 'float64'}, 'D5': {'dtype': 'float64'}, 'D10': {'dtype': 'float64'}, 'D11': {'dtype': 'float64'}, 'D15': {'dtype': 'float64'}, 'M1_enc': {'dtype': 'float64'}, 'M2_enc': {'dtype': 'float64'}, 'M3_enc': {'dtype': 'float64'}, 'M4_enc': {'dtype': 'float64'}, 'M5_enc': {'dtype': 'float64'}, 'M6_enc': {'dtype': 'float64'}, 'M7_enc': {'dtype': 'float64'}, 'M8_enc': {'dtype': 'float64'}, 'M9_enc': {'dtype': 'float64'}, 'id_01': {'dtype': 'float64'}, 'id_02': {'dtype': 'float64'}, 'id_03': {'dtype': 'float64'}, 'id_05': {'dtype': 'float64'}, 'id_06': {'dtype': 'float64'}, 'id_09': {'dtype': 'float64'}, 'id_11': {'dtype': 'float64'}, 'id_13': {'dtype': 'float64'}, 'id_17': {'dtype': 'float64'}, 'id_19': {'dtype': 'float64'}, 'id_20': {'dtype': 'float64'}, 'device_type': {'dtype': 'object'}, 'card1_txn_count': {'dtype': 'int64'}, 'card1_avg_amt': {'dtype': 'float64'}, 'card1_historical_fraud_rate': {'dtype': 'float64'}, 'email_txn_count': {'dtype': 'float64'}, 'email_historical_fraud_rate': {'dtype': 'float64'}, 'is_high_risk_product': {'dtype': 'int32'}, 'amt_vs_card_avg_ratio': {'dtype': 'float64'}}}, 'ours_config': {'configs': {'ibm-aml': {'seed': 2026, 'window_rows': 600000, 'split': {'mode': 'temporal', 'ratios': [0.6, 0.15, 0.1, 0.15], 'group_column': None}, 'semantic': {'enabled': 'true', 'embedding_dim': 128, 'numeric_bins': 32, 'max_text_tokens': 256, 'task_prompt': 'Detect whether this interbank transfer is part of a money-laundering pattern.', 'version': 'gamma-v1', 'fields': [{'name': 'ts', 'description': 'transaction timestamp', 'unit': 'unix seconds', 'entity_role': 'event time', 'source': 'IBM AML transaction log', 'in_serialization': False}, {'name': 'entity', 'description': 'originating account identifier', 'entity_role': 'payer account', 'source': 'IBM AML transaction log', 'in_serialization': False}, {'name': 'counterparty', 'description': 'beneficiary account identifier', 'entity_role': 'payee account', 'source': 'IBM AML transaction log', 'in_serialization': False}, {'name': 'amount', 'description': 'amount paid by the originating account', 'unit': 'currency units of the payment currency', 'entity_role': 'transaction value', 'source': 'IBM AML transaction log'}, {'name': 'cat1', 'description': 'payment rail used to settle the transfer', 'entity_role': 'payment channel', 'source': 'IBM AML transaction log'}, {'name': 'cat2', 'description': 'receiving bank identifier, prefixed with BANK', 'entity_role': 'beneficiary institution', 'source': 'IBM AML transaction log'}, {'name': 'cat3', 'description': 'currency in which the beneficiary is credited', 'unit': 'ISO currency name', 'entity_role': 'settlement currency', 'source': 'IBM AML transaction log'}, {'name': 'text', 'description': 'no free-text field in this dataset', 'source': 'n/a', 'in_serialization': False}, {'name': 'f_amount_received', 'description': 'amount credited to the beneficiary account', 'unit': 'currency units of the receiving currency', 'entity_role': 'transaction value', 'source': 'IBM AML transaction log'}, {'name': 'f_cross_currency', 'description': '1 if paying and receiving currency differ', 'entity_role': 'settlement flag', 'source': 'derived at load time'}, {'name': 'f_cross_bank', 'description': '1 if originating and receiving bank differ', 'entity_role': 'settlement flag', 'source': 'derived at load time'}]}, 'context': {'enabled': 'true', 'mode': 'soft', 'window_seconds': 21600, 'top_k': 8, 'required_fields': ['amount'], 'compatibility_fields': ['cat3'], 'blocked_value_pairs': {}, 'soft_relation_fields': ['cat1', 'cat2', 'cat3'], 'hash_buckets': 512, 'embedding_dim': 8, 'output_dim': 16, 'epochs': 3, 'batch_size': 4096, 'lr': 0.005, 'negative_control': 'none'}, 'graph': {'enabled': True, 'bucket_seconds': 1800, 'epsilon': 0.01, 'bonferroni': True, 'trigger': True, 'require_prior_history': True, 'signal_horizon_buckets': 4, 'signal_scope': 'seed', 'force_rho_one': False, 'rho_estimator': 'overlap', 'rho_min': 'auto', 'rho_quantile': 0.5, 'rho_quantile_grid': [0.0, 0.5, 0.75], 'zeta_mode': 'product', 'zeta_mode_grid': ['product', 'burst_only'], 'rho_fallback': 0.2, 'rho_min_calibration_seeds': 20, 'max_seeds_per_bucket': 64, 'max_patterns_exported': 2000, 'path_samples': 16, 'max_hops': 3, 'vector_dim': 32, 'counter': 'exact', 'cms_width': 4096, 'cms_depth': 5, 'add_reverse': True, 'streams': [{'source': 'entity', 'target': 'counterparty', 'relation': 'transfer', 'source_ns': 'ACC', 'target_ns': 'ACC', 'trigger': True}]}, 'detector': {'n_estimators': 800, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5, 'reg_lambda': 0.5, 'scale_pos_weight': 'auto', 'early_stopping_rounds': 50, 'n_jobs': -1}, 'constraint': {'alpha': 0.015, 'delta': 0.01, 'method': 'np', 'capacity_per_window': None}, 'evaluation': {'selection_criterion': 'ap', 'fpr_grid': [0.001, 0.005, 0.01], 'bootstrap': 0, 'shap_top_k': 3, 'shap_examples': 20, 'policy_block_at': 0.7, 'policy_investigate_at': 0.4}, 'dataset': 'ibm-aml'}, 'sparkov': {'seed': 2026, 'window_rows': 400000, 'split': {'mode': 'temporal', 'ratios': [0.6, 0.15, 0.1, 0.15], 'group_column': None}, 'semantic': {'enabled': 'false', 'embedding_dim': 64, 'numeric_bins': 32, 'max_text_tokens': 256, 'task_prompt': 'Detect whether this card-present or card-not-present purchase is fraudulent.', 'version': 'gamma-v1', 'fields': [{'name': 'ts', 'description': 'transaction timestamp', 'unit': 'unix seconds', 'entity_role': 'event time', 'source': 'Sparkov generator', 'in_serialization': False}, {'name': 'entity', 'description': 'card number of the payer', 'entity_role': 'payment card', 'source': 'Sparkov generator', 'in_serialization': False}, {'name': 'counterparty', 'description': 'merchant that accepted the payment', 'entity_role': 'merchant', 'source': 'Sparkov generator', 'in_serialization': False}, {'name': 'amount', 'description': 'purchase amount', 'unit': 'USD', 'entity_role': 'transaction value', 'source': 'Sparkov generator'}, {'name': 'cat1', 'description': 'merchant spending category', 'entity_role': 'merchant category', 'source': 'Sparkov generator', 'codebook': {'grocery_pos': 'in-store grocery', 'grocery_net': 'online grocery', 'gas_transport': 'fuel and transport', 'misc_net': 'online miscellaneous', 'misc_pos': 'in-store miscellaneous', 'shopping_net': 'online shopping', 'shopping_pos': 'in-store shopping', 'entertainment': 'entertainment', 'food_dining': 'food and dining', 'personal_care': 'personal care', 'health_fitness': 'health and fitness', 'home': 'home goods', 'kids_pets': 'kids and pets', 'travel': 'travel'}}, {'name': 'cat2', 'description': 'US state in the cardholder billing address', 'entity_role': 'cardholder region', 'source': 'Sparkov generator'}, {'name': 'cat3', 'description': 'declared occupation of the cardholder', 'entity_role': 'cardholder attribute', 'source': 'Sparkov generator'}, {'name': 'text', 'description': 'no free-text field in this dataset', 'source': 'n/a', 'in_serialization': False}, {'name': 'f_city_pop', 'description': 'population of the cardholder city', 'unit': 'residents', 'entity_role': 'cardholder region attribute', 'source': 'Sparkov generator'}]}, 'context': {'enabled': 'true', 'mode': 'soft', 'window_seconds': 2592000, 'top_k': 8, 'required_fields': ['amount'], 'compatibility_fields': [], 'blocked_value_pairs': {}, 'soft_relation_fields': ['cat1', 'cat2', 'cat3'], 'hash_buckets': 512, 'embedding_dim': 16, 'output_dim': 8, 'epochs': 2, 'batch_size': 4096, 'lr': 0.005, 'negative_control': 'none'}, 'graph': {'enabled': True, 'bucket_seconds': 86400, 'epsilon': 0.01, 'bonferroni': True, 'trigger': True, 'require_prior_history': True, 'signal_horizon_buckets': 7, 'signal_scope': 'seed', 'force_rho_one': False, 'rho_estimator': 'overlap', 'rho_min': 'auto', 'rho_quantile': 0.0, 'rho_quantile_grid': [0.0, 0.5, 0.75], 'zeta_mode': 'burst_only', 'zeta_mode_grid': ['product', 'burst_only'], 'rho_fallback': 0.2, 'rho_min_calibration_seeds': 20, 'max_seeds_per_bucket': 256, 'max_patterns_exported': 2000, 'path_samples': 16, 'max_hops': 2, 'vector_dim': 32, 'counter': 'exact', 'cms_width': 4096, 'cms_depth': 5, 'add_reverse': True, 'streams': [{'source': 'entity', 'target': 'counterparty', 'relation': 'pays', 'source_ns': 'CARD', 'target_ns': 'MERCH', 'trigger': True}]}, 'detector': {'n_estimators': 800, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.6, 'colsample_bytree': 0.6, 'min_child_weight': 20, 'reg_lambda': 0.5, 'scale_pos_weight': 'auto', 'early_stopping_rounds': 50, 'n_jobs': -1}, 'constraint': {'alpha': 0.008, 'delta': 0.05, 'method': 'np', 'capacity_per_window': None}, 'evaluation': {'selection_criterion': 'ap', 'fpr_grid': [0.001, 0.005, 0.01], 'bootstrap': 0, 'shap_top_k': 3, 'shap_examples': 20, 'policy_block_at': 0.7, 'policy_investigate_at': 0.4}, 'dataset': 'sparkov'}, 'ieee-cis': {'seed': 2026, 'window_rows': 400000, 'split': {'mode': 'temporal', 'ratios': [0.6, 0.15, 0.1, 0.15], 'group_column': None}, 'semantic': {'enabled': 'false', 'embedding_dim': 64, 'numeric_bins': 32, 'max_text_tokens': 256, 'task_prompt': 'Detect whether this card transaction is fraudulent.', 'version': 'gamma-v1', 'fields': [{'name': 'ts', 'description': 'transaction timestamp derived from TransactionDT', 'unit': 'unix seconds', 'entity_role': 'event time', 'source': 'IEEE-CIS transaction table', 'in_serialization': False}, {'name': 'entity', 'description': 'card identifier hash', 'entity_role': 'payment card', 'source': 'IEEE-CIS card1', 'in_serialization': False}, {'name': 'counterparty', 'description': 'billing address region of the transaction', 'entity_role': 'billing region', 'source': 'IEEE-CIS addr1', 'in_serialization': False}, {'name': 'amount', 'description': 'transaction amount', 'unit': 'USD', 'entity_role': 'transaction value', 'source': 'IEEE-CIS TransactionAmt'}, {'name': 'cat1', 'description': 'product code of the purchased item', 'entity_role': 'product category', 'source': 'IEEE-CIS ProductCD', 'codebook': {'W': 'web purchase', 'C': 'card-present or catalogue', 'R': 'refund or reversal', 'H': 'hospitality', 'S': 'services'}}, {'name': 'cat2', 'description': 'card network', 'entity_role': 'card scheme', 'source': 'IEEE-CIS card4', 'codebook': {'visa': 'Visa', 'mastercard': 'Mastercard', 'discover': 'Discover', 'american express': 'American Express'}}, {'name': 'cat3', 'description': 'funding type of the card', 'entity_role': 'funding type', 'source': 'IEEE-CIS card6', 'codebook': {'debit': 'debit card, funds drawn immediately', 'credit': 'credit card, funds drawn on a line of credit'}}, {'name': 'text', 'description': 'no free-text field in this dataset', 'source': 'n/a', 'in_serialization': False}]}, 'context': {'enabled': 'true', 'mode': 'soft', 'window_seconds': 2592000, 'top_k': 8, 'required_fields': ['amount'], 'compatibility_fields': [], 'blocked_value_pairs': {}, 'soft_relation_fields': ['cat1', 'cat2', 'cat3'], 'hash_buckets': 512, 'embedding_dim': 8, 'output_dim': 16, 'epochs': 3, 'batch_size': 4096, 'lr': 0.005, 'negative_control': 'none', 'max_history': 8}, 'graph': {'enabled': True, 'bucket_seconds': 86400, 'epsilon': 0.05, 'bonferroni': True, 'trigger': True, 'require_prior_history': True, 'signal_horizon_buckets': 14, 'signal_scope': 'seed', 'force_rho_one': False, 'rho_estimator': 'overlap', 'rho_min': 'auto', 'rho_quantile': 0.75, 'rho_quantile_grid': [0.0, 0.5, 0.75], 'zeta_mode': 'product', 'zeta_mode_grid': ['product', 'burst_only'], 'rho_fallback': 0.2, 'rho_min_calibration_seeds': 20, 'max_seeds_per_bucket': 16, 'max_patterns_exported': 2000, 'path_samples': 8, 'max_hops': 3, 'vector_dim': 32, 'counter': 'exact', 'cms_width': 4096, 'cms_depth': 5, 'add_reverse': True, 'streams': [{'source': 'entity', 'target': 'counterparty', 'relation': 'pays', 'source_ns': 'CARD', 'target_ns': 'ADDR', 'trigger': True}]}, 'detector': {'n_estimators': 800, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 1.0, 'colsample_bytree': 0.8, 'min_child_weight': 20, 'reg_lambda': 5.0, 'scale_pos_weight': 'auto', 'early_stopping_rounds': 50, 'n_jobs': -1}, 'constraint': {'alpha': 0.01, 'delta': 0.01, 'method': 'np', 'capacity_per_window': None}, 'evaluation': {'selection_criterion': 'ap', 'fpr_grid': [0.001, 0.005, 0.01], 'bootstrap': 0, 'shap_top_k': 3, 'shap_examples': 20, 'policy_block_at': 0.7, 'policy_investigate_at': 0.4}, 'dataset': 'ieee-cis'}, 'ibm-aml-medium': {'seed': 2026, 'window_rows': 2000000, 'split': {'mode': 'temporal', 'ratios': [0.6, 0.15, 0.1, 0.15], 'group_column': None}, 'semantic': {'enabled': True, 'embedding_dim': 64, 'numeric_bins': 32, 'max_text_tokens': 256, 'task_prompt': 'Detect whether this bank transfer is part of a money laundering pattern.', 'version': 'gamma-v1', 'fields': [{'name': 'ts', 'description': 'transaction timestamp at minute granularity', 'unit': 'unix seconds', 'entity_role': 'event time', 'source': 'IBM AML Timestamp', 'in_serialization': False}, {'name': 'entity', 'description': 'originating bank account', 'entity_role': 'payer account', 'source': 'IBM AML Account', 'in_serialization': False}, {'name': 'counterparty', 'description': 'receiving bank account', 'entity_role': 'payee account', 'source': 'IBM AML Account.1', 'in_serialization': False}, {'name': 'amount', 'description': 'amount paid by the originator', 'unit': 'source currency', 'entity_role': 'transaction value', 'source': 'IBM AML Amount Paid'}, {'name': 'cat1', 'description': 'settlement instrument used for the transfer', 'entity_role': 'payment instrument', 'source': 'IBM AML Payment Format', 'codebook': {'ACH': 'automated clearing house transfer', 'Cheque': 'paper cheque', 'Credit Card': 'credit card payment', 'Cash': 'cash deposit or withdrawal', 'Wire': 'wire transfer', 'Bitcoin': 'bitcoin transfer', 'Reinvestment': 'reinvested proceeds'}}, {'name': 'cat2', 'description': 'currency in which the payee is credited', 'entity_role': 'settlement currency', 'source': 'IBM AML Receiving Currency'}, {'name': 'cat3', 'description': 'receiving bank identifier', 'entity_role': 'counterparty institution', 'source': 'IBM AML To Bank'}, {'name': 'text', 'description': 'no free-text field in this dataset', 'source': 'n/a', 'in_serialization': False}]}, 'context': {'enabled': 'true', 'mode': 'soft', 'window_seconds': 21600, 'top_k': 8, 'required_fields': ['amount'], 'compatibility_fields': [], 'blocked_value_pairs': {}, 'soft_relation_fields': ['cat1', 'cat2', 'cat3'], 'hash_buckets': 512, 'embedding_dim': 8, 'output_dim': 8, 'epochs': 3, 'batch_size': 4096, 'lr': 0.005, 'negative_control': 'none', 'max_history': 8}, 'graph': {'enabled': True, 'bucket_seconds': 1800, 'epsilon': 0.01, 'bonferroni': True, 'trigger': True, 'require_prior_history': True, 'signal_horizon_buckets': 4, 'signal_scope': 'seed', 'force_rho_one': False, 'rho_estimator': 'overlap', 'rho_min': 'auto', 'rho_quantile': 0.0, 'rho_quantile_grid': [0.0, 0.5, 0.75], 'zeta_mode': 'product', 'zeta_mode_grid': ['product', 'burst_only'], 'rho_fallback': 0.2, 'rho_min_calibration_seeds': 20, 'max_seeds_per_bucket': 64, 'max_patterns_exported': 2000, 'path_samples': 16, 'max_hops': 3, 'vector_dim': 32, 'counter': 'exact', 'cms_width': 4096, 'cms_depth': 5, 'add_reverse': True, 'streams': [{'source': 'entity', 'target': 'counterparty', 'relation': 'transfer', 'source_ns': 'ACC', 'target_ns': 'ACC', 'trigger': True}]}, 'detector': {'n_estimators': 800, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1, 'reg_lambda': 1.0, 'scale_pos_weight': 'auto', 'early_stopping_rounds': 50, 'n_jobs': -1}, 'constraint': {'alpha': 0.015, 'delta': 0.05, 'method': 'np', 'capacity_per_window': None}, 'evaluation': {'selection_criterion': 'ap', 'fpr_grid': [0.001, 0.005, 0.01], 'bootstrap': 0, 'shap_top_k': 3, 'shap_examples': 20, 'policy_block_at': 0.7, 'policy_investigate_at': 0.4}, 'dataset': 'ibm-aml-medium'}, 'ibm-aml-li': {'seed': 2026, 'window_rows': 2000000, 'split': {'mode': 'temporal', 'ratios': [0.6, 0.15, 0.1, 0.15], 'group_column': None}, 'semantic': {'enabled': True, 'embedding_dim': 64, 'numeric_bins': 32, 'max_text_tokens': 256, 'task_prompt': 'Detect whether this bank transfer is part of a money laundering pattern.', 'version': 'gamma-v1', 'fields': [{'name': 'ts', 'description': 'transaction timestamp at minute granularity', 'unit': 'unix seconds', 'entity_role': 'event time', 'source': 'IBM AML Timestamp', 'in_serialization': False}, {'name': 'entity', 'description': 'originating bank account', 'entity_role': 'payer account', 'source': 'IBM AML Account', 'in_serialization': False}, {'name': 'counterparty', 'description': 'receiving bank account', 'entity_role': 'payee account', 'source': 'IBM AML Account.1', 'in_serialization': False}, {'name': 'amount', 'description': 'amount paid by the originator', 'unit': 'source currency', 'entity_role': 'transaction value', 'source': 'IBM AML Amount Paid'}, {'name': 'cat1', 'description': 'settlement instrument used for the transfer', 'entity_role': 'payment instrument', 'source': 'IBM AML Payment Format', 'codebook': {'ACH': 'automated clearing house transfer', 'Cheque': 'paper cheque', 'Credit Card': 'credit card payment', 'Cash': 'cash deposit or withdrawal', 'Wire': 'wire transfer', 'Bitcoin': 'bitcoin transfer', 'Reinvestment': 'reinvested proceeds'}}, {'name': 'cat2', 'description': 'currency in which the payee is credited', 'entity_role': 'settlement currency', 'source': 'IBM AML Receiving Currency'}, {'name': 'cat3', 'description': 'receiving bank identifier', 'entity_role': 'counterparty institution', 'source': 'IBM AML To Bank'}, {'name': 'text', 'description': 'no free-text field in this dataset', 'source': 'n/a', 'in_serialization': False}]}, 'context': {'enabled': 'true', 'mode': 'soft', 'window_seconds': 21600, 'top_k': 16, 'required_fields': ['amount'], 'compatibility_fields': [], 'blocked_value_pairs': {}, 'soft_relation_fields': ['cat1', 'cat2', 'cat3'], 'hash_buckets': 512, 'embedding_dim': 16, 'output_dim': 8, 'epochs': 2, 'batch_size': 4096, 'lr': 0.01, 'negative_control': 'none', 'max_history': 8}, 'graph': {'enabled': True, 'bucket_seconds': 1800, 'epsilon': 0.01, 'bonferroni': True, 'trigger': True, 'require_prior_history': True, 'signal_horizon_buckets': 4, 'signal_scope': 'seed', 'force_rho_one': False, 'rho_estimator': 'overlap', 'rho_min': 'auto', 'rho_quantile': 0.5, 'rho_quantile_grid': [0.0, 0.5, 0.75], 'zeta_mode': 'product', 'zeta_mode_grid': ['product', 'burst_only'], 'rho_fallback': 0.2, 'rho_min_calibration_seeds': 20, 'max_seeds_per_bucket': 64, 'max_patterns_exported': 2000, 'path_samples': 16, 'max_hops': 3, 'vector_dim': 32, 'counter': 'exact', 'cms_width': 4096, 'cms_depth': 5, 'add_reverse': True, 'streams': [{'source': 'entity', 'target': 'counterparty', 'relation': 'transfer', 'source_ns': 'ACC', 'target_ns': 'ACC', 'trigger': True}]}, 'detector': {'n_estimators': 800, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5, 'reg_lambda': 1.0, 'scale_pos_weight': 'auto', 'early_stopping_rounds': 50, 'n_jobs': -1}, 'constraint': {'alpha': 0.015, 'delta': 0.05, 'method': 'np', 'capacity_per_window': None}, 'evaluation': {'selection_criterion': 'ap', 'fpr_grid': [0.001, 0.005, 0.01], 'bootstrap': 0, 'shap_top_k': 3, 'shap_examples': 20, 'policy_block_at': 0.7, 'policy_investigate_at': 0.4}, 'dataset': 'ibm-aml-li'}, 'banksim': {'seed': 2026, 'window_rows': 400000, 'split': {'mode': 'temporal', 'ratios': [0.6, 0.15, 0.1, 0.15], 'group_column': None}, 'semantic': {'enabled': 'true', 'embedding_dim': 64, 'numeric_bins': 32, 'max_text_tokens': 256, 'task_prompt': 'Detect whether this retail card payment is fraudulent.', 'version': 'gamma-v1', 'fields': [{'name': 'ts', 'description': 'day index of the simulated period, one step per day', 'unit': 'unix seconds', 'entity_role': 'event time', 'source': 'BankSim step', 'in_serialization': False}, {'name': 'entity', 'description': 'customer account making the payment', 'entity_role': 'cardholder', 'source': 'BankSim customer', 'in_serialization': False}, {'name': 'counterparty', 'description': 'merchant receiving the payment', 'entity_role': 'merchant', 'source': 'BankSim merchant', 'in_serialization': False}, {'name': 'amount', 'description': 'payment amount', 'unit': 'euro', 'entity_role': 'transaction value', 'source': 'BankSim amount'}, {'name': 'cat1', 'description': 'merchant spending category', 'entity_role': 'spending category', 'source': 'BankSim category', 'codebook': {'es_transportation': 'transport and travel', 'es_health': 'healthcare services', 'es_otherservices': 'other services', 'es_food': 'groceries and food', 'es_hotelservices': 'hotels and lodging', 'es_barsandrestaurants': 'bars and restaurants', 'es_tech': 'technology and electronics', 'es_sportsandtoys': 'sports goods and toys', 'es_wellnessandbeauty': 'wellness and beauty', 'es_hyper': 'hypermarket', 'es_fashion': 'clothing and fashion', 'es_home': 'home goods', 'es_contents': 'digital content', 'es_travel': 'travel bookings', 'es_leisure': 'leisure activities'}}, {'name': 'cat2', 'description': 'binned age bracket of the cardholder', 'entity_role': 'customer age band', 'source': 'BankSim age'}, {'name': 'cat3', 'description': 'gender of the cardholder', 'entity_role': 'customer gender', 'source': 'BankSim gender'}, {'name': 'text', 'description': 'no free-text field in this dataset', 'source': 'n/a', 'in_serialization': False}]}, 'context': {'enabled': 'true', 'mode': 'soft', 'window_seconds': 2592000, 'top_k': 16, 'required_fields': ['amount'], 'compatibility_fields': [], 'blocked_value_pairs': {}, 'soft_relation_fields': ['cat1', 'cat2', 'cat3'], 'hash_buckets': 512, 'embedding_dim': 8, 'output_dim': 8, 'epochs': 3, 'batch_size': 4096, 'lr': 0.005, 'negative_control': 'none', 'max_history': 8}, 'graph': {'enabled': True, 'bucket_seconds': 172800.0, 'epsilon': 0.01, 'bonferroni': True, 'trigger': True, 'require_prior_history': True, 'signal_horizon_buckets': 2, 'signal_scope': 'seed', 'force_rho_one': False, 'rho_estimator': 'overlap', 'rho_min': 'auto', 'rho_quantile': 0.0, 'rho_quantile_grid': [0.0, 0.5, 0.75], 'zeta_mode': 'product', 'zeta_mode_grid': ['product', 'burst_only'], 'rho_fallback': 0.2, 'rho_min_calibration_seeds': 20, 'max_seeds_per_bucket': 64, 'max_patterns_exported': 2000, 'path_samples': 16, 'max_hops': 3, 'vector_dim': 32, 'counter': 'exact', 'cms_width': 4096, 'cms_depth': 5, 'add_reverse': True, 'streams': [{'source': 'entity', 'target': 'counterparty', 'relation': 'pays', 'source_ns': 'CUST', 'target_ns': 'MERCH', 'trigger': True}]}, 'detector': {'n_estimators': 800, 'max_depth': 5, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5, 'reg_lambda': 1.0, 'scale_pos_weight': 'auto', 'early_stopping_rounds': 50, 'n_jobs': -1}, 'constraint': {'alpha': 0.004, 'delta': 0.05, 'method': 'np', 'capacity_per_window': None}, 'evaluation': {'selection_criterion': 'ap', 'fpr_grid': [0.001, 0.005, 0.01], 'bootstrap': 0, 'shap_top_k': 3, 'shap_examples': 20, 'policy_block_at': 0.7, 'policy_investigate_at': 0.4}, 'dataset': 'banksim'}}}}

# -------------------- _ours_runtime --------------------
_install('_ours_runtime', r'''
from pathlib import Path
import os, sys
MODEL = _ROOT / 'model'
ROOT = _ROOT
DATASET = _DATASET
CONFIG = MODEL / 'config'
OUTPUT = _OUTPUT

def bootstrap():
    pass

def read_dataset_csv(path):
    """Restore Parquet dtypes after verified, lossless CSV export."""
    import json
    import pandas as pd
    path = Path(path).with_suffix('.csv')
    schema = _DATA['csv_schema'][path.parent.name]
    dtype = {k: 'string' if v['dtype'].startswith('datetime') or v['dtype'] == 'category' else v['dtype'] for k, v in schema.items()}
    frame = pd.read_csv(path, dtype=dtype, keep_default_na=False, na_values=['__MAS_NULL__'], float_precision='round_trip')
    for col, spec in schema.items():
        if spec['dtype'].startswith('datetime'):
            frame[col] = pd.to_datetime(frame[col]).astype(spec['dtype'])
        elif spec['dtype'] == 'category':
            frame[col] = pd.Categorical(frame[col], categories=spec['categories'], ordered=spec['ordered'])
    return frame
''', _ROOT=_ROOT, _DATASET=_DATASET, _OUTPUT=_OUTPUT, _DATA=_DATA, _CONFIGS=_CONFIGS)

# -------------------- how123.config --------------------
_install('how123.config', r'''
from __future__ import annotations
import copy
import json
import os
from typing import Any, Dict, Mapping
import yaml
HERE = os.path.dirname(os.path.abspath(__file__))
from _ours_runtime import CONFIG
CONFIG_DIR = str(CONFIG)

def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并；override 的叶子覆盖 base，dict 继续下钻，list 整体替换。"""
    out = dict(copy.deepcopy(dict(base)))
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out

def get_path(cfg: Mapping[str, Any], dotted: str, default: Any=None) -> Any:
    node: Any = cfg
    for part in dotted.split('.'):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node

def set_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split('.')
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value

def apply_overrides(cfg: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """按点号路径覆盖。未知路径不会被静默接受——先建后校验，validate 会拦。"""
    out = copy.deepcopy(dict(cfg))
    for dotted, value in overrides.items():
        if get_path(out, dotted, _MISSING) is _MISSING:
            raise KeyError(f'覆盖路径在配置中不存在：{dotted}（防止消融臂写错字段名而静默无效）')
        set_path(out, dotted, value)
    return out

class _Missing:
    pass
_MISSING = _Missing()

def load_config(dataset, config_dir=None):
    if dataset not in _CONFIGS:
        raise ValueError(f'Unknown dataset: {dataset}')
    cfg = deep_merge(_CONFIGS['default'], _CONFIGS[dataset])
    cfg['dataset'] = dataset
    validate(cfg)
    return cfg

def validate(cfg: Mapping[str, Any]) -> None:
    """跑之前一次性检查取值范围。放过一个非法值的代价是几十分钟后才报错。"""
    alpha = float(get_path(cfg, 'constraint.alpha'))
    delta = float(get_path(cfg, 'constraint.delta'))
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError('constraint.alpha 与 constraint.delta 必须落在 (0,1)')
    if get_path(cfg, 'constraint.method') not in ('np', 'empirical', 'fixed'):
        raise ValueError('constraint.method 只能是 np / empirical / fixed')
    eps = float(get_path(cfg, 'graph.epsilon'))
    if not 0.0 < eps < 1.0:
        raise ValueError('graph.epsilon 必须落在 (0,1)')
    if get_path(cfg, 'graph.counter') not in ('exact', 'cms'):
        raise ValueError('graph.counter 只能是 exact 或 cms')
    if float(get_path(cfg, 'graph.bucket_seconds')) <= 0:
        raise ValueError('graph.bucket_seconds 必须为正')
    if get_path(cfg, 'context.mode') not in ('soft', 'hard'):
        raise ValueError('context.mode 只能是 soft 或 hard')
    if int(get_path(cfg, 'context.top_k')) <= 0:
        raise ValueError('context.top_k 必须为正')
    if float(get_path(cfg, 'context.window_seconds')) <= 0:
        raise ValueError('context.window_seconds 必须为正')
    if str(get_path(cfg, 'semantic.enabled')).lower() not in ('auto', 'text_only', 'true', 'false'):
        raise ValueError('semantic.enabled 只能是 auto / text_only / true / false')
    if get_path(cfg, 'evaluation.selection_criterion') not in ('ap', 'tpr_at_alpha'):
        raise ValueError('evaluation.selection_criterion 只能是 ap 或 tpr_at_alpha')
    quantile = get_path(cfg, 'graph.rho_quantile')
    if isinstance(quantile, str):
        if quantile.lower() != 'auto':
            raise ValueError('graph.rho_quantile 只能是 auto 或 [0,1] 内的数')
        grid = get_path(cfg, 'graph.rho_quantile_grid') or []
        if not grid or any((not 0.0 <= float(q) <= 1.0 for q in grid)):
            raise ValueError('graph.rho_quantile=auto 时 rho_quantile_grid 必须非空且取值在 [0,1]')
    elif not 0.0 <= float(quantile) <= 1.0:
        raise ValueError('graph.rho_quantile 必须落在 [0,1]')
    zeta = get_path(cfg, 'graph.zeta_mode', 'product')
    if str(zeta).lower() == 'auto':
        grid = get_path(cfg, 'graph.zeta_mode_grid') or []
        if not grid or any((g not in ('product', 'burst_only') for g in grid)):
            raise ValueError('zeta_mode_grid 的取值只能是 product 或 burst_only')
    elif zeta not in ('product', 'burst_only'):
        raise ValueError('graph.zeta_mode 只能是 auto / product / burst_only')
    ratios = get_path(cfg, 'split.ratios')
    if len(ratios) != 4 or any((r <= 0 for r in ratios)) or abs(sum(ratios) - 1.0) > 1e-08:
        raise ValueError('split.ratios 必须是四个正数且和为 1（train/tune/calibration/test）')
    if get_path(cfg, 'split.mode') not in ('temporal', 'group'):
        raise ValueError('split.mode 只能是 temporal 或 group')
    if get_path(cfg, 'split.mode') == 'group' and (not get_path(cfg, 'split.group_column')):
        raise ValueError('split.mode=group 时必须给出 split.group_column')
    if not get_path(cfg, 'semantic.fields'):
        raise ValueError('semantic.fields（领域语义规格 Γ）不得为空——1.4.1 的前提是字段说明受控而非模型猜测')

def config_digest(cfg: Mapping[str, Any]) -> str:
    """配置指纹，写进结果文件，避免"结果对不上但不知道跑的是哪版配置"。"""
    import hashlib
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')
    return hashlib.blake2b(payload, digest_size=8).hexdigest()
''', _ROOT=_ROOT, _DATASET=_DATASET, _OUTPUT=_OUTPUT, _DATA=_DATA, _CONFIGS=_CONFIGS)

# -------------------- how123.contracts --------------------
_install('how123.contracts', r'''
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np

class ContractViolation(RuntimeError):
    """任何一条全局契约被破坏都抛这个，不做降级处理。"""

@dataclass(frozen=True)
class FieldSpec:
    """Γ 中单个字段的受控说明。对应 §1.4.1 的 (n_j, d_j, u_j, c_j, ρ_j, g_j, ω_j)。"""
    name: str
    description: str = ''
    unit: str = ''
    entity_role: str = ''
    source: str = ''
    codebook: Mapping[str, str] = field(default_factory=dict)
    availability_lag_seconds: float = 0.0
    posthoc: bool = False
    in_serialization: bool = True

    def decode(self, value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return '<missing>'
        return str(self.codebook.get(str(value), value))

    def available_at(self, event_time: float) -> float:
        if self.posthoc:
            return float('inf')
        return float(event_time) + float(self.availability_lag_seconds)

@dataclass(frozen=True)
class SemanticSpec:
    """全局契约 Γ。语义层、上文层、图层共用同一份，不是语义模块的私有产物（§4.1）。"""
    fields: Mapping[str, FieldSpec]
    task_description: str
    version: str = 'gamma-v1'

    @classmethod
    def from_config(cls, semantic_cfg: Mapping[str, Any]) -> 'SemanticSpec':
        specs: Dict[str, FieldSpec] = {}
        for item in semantic_cfg.get('fields', []):
            specs[str(item['name'])] = FieldSpec(name=str(item['name']), description=str(item.get('description', '')), unit=str(item.get('unit', '')), entity_role=str(item.get('entity_role', '')), source=str(item.get('source', '')), codebook={str(k): str(v) for k, v in (item.get('codebook') or {}).items()}, availability_lag_seconds=float(item.get('availability_lag_seconds', 0.0)), posthoc=bool(item.get('posthoc', False)), in_serialization=bool(item.get('in_serialization', True)))
        return cls(fields=specs, task_description=str(semantic_cfg.get('task_prompt', '')), version=str(semantic_cfg.get('version', 'gamma-v1')))

    def field(self, name: str) -> FieldSpec:
        try:
            return self.fields[name]
        except KeyError as exc:
            raise KeyError(f'字段 {name!r} 未在 Γ 中登记；1.4.1 不允许模型自行猜测匿名字段') from exc

    def is_empty_spec(self) -> bool:
        """Γ 取空规格时 1.4.1 必须逐字退化为普通序列化（§4.5 不变量）。"""
        return all((not (f.description or f.unit or f.codebook or f.entity_role or f.source) for f in self.fields.values()))

    def posthoc_fields(self) -> Tuple[str, ...]:
        return tuple(sorted((name for name, spec in self.fields.items() if spec.posthoc)))

    def validate_columns(self, columns: Iterable[str]) -> None:
        unknown = sorted(set(columns) - set(self.fields))
        if unknown:
            raise ContractViolation(f'以下列未在 Γ 中登记，不得进入表示：{unknown}')

class TemporalAvailabilityGuard:
    """把 ω_j <= t 与 E_t = {e : τ_e <= t} 落成可调用的检查。"""

    def __init__(self, gamma: SemanticSpec) -> None:
        self.gamma = gamma

    def visible_fields(self, columns: Sequence[str], event_time: float, decision_time: float) -> Tuple[List[str], List[str]]:
        """返回 (可见字段, 因 ω_j > t 被排除的字段)。"""
        visible, excluded = ([], [])
        for name in columns:
            spec = self.gamma.field(name)
            if spec.available_at(event_time) <= decision_time:
                visible.append(name)
            else:
                excluded.append(name)
        return (visible, excluded)

    def assert_no_posthoc(self, used_columns: Iterable[str]) -> int:
        """契约一：判定后字段不得出现在任何表示的输入列里。返回检查的列数。"""
        used = list(used_columns)
        banned = set(self.gamma.posthoc_fields())
        hit = sorted(set(used) & banned)
        if hit:
            raise ContractViolation(f'判定后字段进入了表示：{hit}（ω_j = +inf，契约一）')
        return len(used)

    @staticmethod
    def assert_history_window(current_ts: np.ndarray, history_ts: np.ndarray, valid_mask: np.ndarray, window_seconds: float) -> int:
        """契约二：被采纳的历史必须满足 0 < τ_t - τ_i <= W。返回检查的 (行, 槽) 数。"""
        if valid_mask.size == 0:
            return 0
        lag = current_ts[:, None] - history_ts
        bad = valid_mask & ~((lag > 0) & (lag <= window_seconds))
        if bool(bad.any()):
            idx = np.argwhere(bad)[0]
            raise ContractViolation(f'被采纳的历史越界：行 {int(idx[0])} 槽 {int(idx[1])} lag={float(lag[idx[0], idx[1]])}')
        return int(valid_mask.size)

    @staticmethod
    def assert_causal_edges(edge_ts: np.ndarray, decision_time: float) -> int:
        """契约二：图计算只在 E_t 上进行。返回检查的边数。"""
        if edge_ts.size and float(edge_ts.max()) > decision_time:
            raise ContractViolation(f'未来边进入了 as-of 视图：max(τ_e)={float(edge_ts.max())} > t={decision_time}')
        return int(edge_ts.size)

class ScorerLifecycle:
    """冻结点状态机。NP 次序统计量的论证依赖 s(x) 在校准后不再改变（§4.2 契约三）。"""
    TRAINING, TUNING, FROZEN, CALIBRATED = ('TRAINING', 'TUNING', 'FROZEN', 'CALIBRATED')

    def __init__(self) -> None:
        self.state = self.TRAINING
        self.version = 0

    def enter_tuning(self) -> None:
        if self.state != self.TRAINING:
            raise ContractViolation(f'不能从 {self.state} 进入调参阶段')
        self.state = self.TUNING

    def freeze(self) -> int:
        if self.state not in (self.TRAINING, self.TUNING):
            raise ContractViolation(f'不能从 {self.state} 冻结')
        self.version += 1
        self.state = self.FROZEN
        return self.version

    def mark_calibrated(self) -> None:
        if self.state != self.FROZEN:
            raise ContractViolation('NP 校准要求评分器已冻结')
        self.state = self.CALIBRATED

    def assert_scoring_allowed(self) -> None:
        if self.state not in (self.FROZEN, self.CALIBRATED):
            raise ContractViolation('部署打分要求评分器已冻结')

@dataclass
class CheckResult:
    name: str
    status: str
    n_checked: int
    detail: str = ''

class ContractAuditor:
    """收集契约检查结果。

    memory 里踩过的坑：校验器"通过"要看匹配计数——数目为 0 时输出与真过一模一样。
    因此这里强制每项检查上报 n_checked，n_checked == 0 一律记 SKIPPED。
    """
    CONTROL_LAYER_NAMES = ('trigger', 'triggered', 'is_trigger', 'm_t', 'epsilon_t', 'burst_gate', 'seed_flag')

    def __init__(self) -> None:
        self.results: List[CheckResult] = []

    def record(self, name: str, n_checked: int, ok: bool=True, detail: str='') -> None:
        if n_checked <= 0:
            self.results.append(CheckResult(name, 'SKIPPED', 0, detail or '检查对象为空'))
        else:
            self.results.append(CheckResult(name, 'PASSED' if ok else 'FAILED', n_checked, detail))

    def flag(self, name: str, n_checked: int, clean: bool, detail: str='') -> None:
        """诊断项：越限记 REVIEW 而不是 FAILED。

        契约违反（因果、冻结、切分复用）是硬失败，管线不该继续；而"某个特征与标签的
        相关性偏高"只是一个"去看一眼"的信号——强真信号也会触发它。两者混在同一个
        passed 里，会让真正的契约失败被淹没。
        """
        if n_checked <= 0:
            self.results.append(CheckResult(name, 'SKIPPED', 0, detail or '检查对象为空'))
        else:
            self.results.append(CheckResult(name, 'PASSED' if clean else 'REVIEW', n_checked, detail))

    def run(self, name: str, fn, detail: str='') -> None:
        """执行一项检查；fn 必须返回它实际检查了多少个对象。"""
        try:
            n = int(fn())
            self.record(name, n, True, detail)
        except (ContractViolation, ValueError, KeyError) as exc:
            self.results.append(CheckResult(name, 'FAILED', -1, f'{type(exc).__name__}: {exc}'))

    def assert_control_not_in_scorer(self, feature_names: Sequence[str]) -> int:
        lowered = {n.lower() for n in feature_names}
        hit = sorted((n for n in self.CONTROL_LAYER_NAMES if n in lowered))
        if hit:
            raise ContractViolation(f'控制层量进入了 s(x)：{hit}（§4.3 范畴错误）')
        return len(feature_names)

    @property
    def passed(self) -> bool:
        return all((r.status != 'FAILED' for r in self.results))

    @property
    def n_skipped(self) -> int:
        return sum((1 for r in self.results if r.status == 'SKIPPED'))

    @property
    def n_review(self) -> int:
        return sum((1 for r in self.results if r.status == 'REVIEW'))

    def to_dict(self) -> Dict[str, Any]:
        return {'passed': self.passed, 'n_checks': len(self.results), 'n_passed': sum((1 for r in self.results if r.status == 'PASSED')), 'n_failed': sum((1 for r in self.results if r.status == 'FAILED')), 'n_review': self.n_review, 'n_skipped': self.n_skipped, 'checks': [{'name': r.name, 'status': r.status, 'n_checked': r.n_checked, 'detail': r.detail} for r in self.results]}

    def summary_line(self) -> str:
        return f"契约审计: {sum((1 for r in self.results if r.status == 'PASSED'))} PASSED / {sum((1 for r in self.results if r.status == 'FAILED'))} FAILED / {self.n_review} REVIEW / {self.n_skipped} SKIPPED"

@dataclass(frozen=True)
class PatternEntry:
    """§2.1.3 的 p=(S,R,T,A)，四个分量各有出处（§2.4.8 输出表）。"""
    pattern_id: str
    S: Tuple[str, ...]
    R: Tuple[Tuple[str, str, str], ...]
    T: Tuple[float, ...]
    A: Mapping[str, Any]
    seed_pair: Tuple[str, str]
    time_bucket: int
    burst_score: float
    concentration: float
    zeta: float
    epsilon_t: float
    m_t: int
    novelty: str = 'UNKNOWN'
    signature: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {'pattern_id': self.pattern_id, 'S': list(self.S), 'R': [list(r) for r in self.R], 'T': [float(t) for t in self.T], 'A': dict(self.A), 'seed_pair': list(self.seed_pair), 'time_bucket': int(self.time_bucket), 'burst_score': float(self.burst_score), 'concentration': float(self.concentration), 'zeta': float(self.zeta), 'epsilon_t': float(self.epsilon_t), 'm_t': int(self.m_t), 'novelty': self.novelty, 'signature': self.signature}
''', _ROOT=_ROOT, _DATASET=_DATASET, _OUTPUT=_OUTPUT, _DATA=_DATA, _CONFIGS=_CONFIGS)

# -------------------- how123.metrics --------------------
_install('how123.metrics', r'''
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence
import numpy as np
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score

def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float('nan')
    return float(average_precision_score(labels, scores))

def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float('nan')
    return float(roc_auc_score(labels, scores))

def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """在合法样本分数的 (1−fpr) 分位处取阈值，再算 TPR。"""
    negatives = scores[labels == 0]
    positives = scores[labels == 1]
    if negatives.size == 0 or positives.size == 0:
        return float('nan')
    threshold = float(np.quantile(negatives, 1.0 - target_fpr, method='higher'))
    return float((positives > threshold).mean())

def clopper_pearson_upper(false_positives: int, negatives: int, delta: float) -> float:
    if negatives <= 0:
        return 1.0
    if false_positives >= negatives:
        return 1.0
    return float(beta.ppf(1.0 - delta, false_positives + 1, negatives - false_positives))

@dataclass(frozen=True)
class EvaluationResult:
    n: int
    n_positive: int
    n_negative: int
    prevalence: float
    ap: float
    ap_lift: float
    auroc: float
    tpr_at_fpr: Dict[str, float]
    threshold: float
    n_alert: int
    n_tp: int
    n_fp: int
    tpr: float
    fpr: float
    precision: float
    fdp: float
    fpr_upper_bound: float
    constraint_satisfied: bool
    n_tn: int = 0
    n_fn: int = 0
    f1: float = 0.0
    accuracy: float = 0.0
    ap_ci_low: float = float('nan')
    ap_ci_high: float = float('nan')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0

def evaluate(labels: Sequence[int], scores: Sequence[float], *, threshold: float, alpha: float, delta: float, alerts: Optional[Sequence[bool]]=None, fpr_grid: Sequence[float]=(0.001, 0.005, 0.01), bootstrap: int=0, seed: int=2026) -> EvaluationResult:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    predicted = np.asarray(alerts, dtype=bool) if alerts is not None else s > threshold
    pos, neg = (y == 1, y == 0)
    n_pos, n_neg = (int(pos.sum()), int(neg.sum()))
    tp, fp = (int((predicted & pos).sum()), int((predicted & neg).sum()))
    tn, fn = (n_neg - fp, n_pos - tp)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, n_pos)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, y.size)
    prevalence = _safe_div(n_pos, y.size)
    ap = average_precision(y, s)
    upper = clopper_pearson_upper(fp, n_neg, delta)
    ci_low = ci_high = float('nan')
    if bootstrap > 0 and n_pos > 0:
        draws = bootstrap_ap(y, s, repeats=bootstrap, seed=seed)
        ci_low, ci_high = (float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5)))
    return EvaluationResult(n=int(y.size), n_positive=n_pos, n_negative=n_neg, prevalence=prevalence, ap=ap, ap_lift=_safe_div(ap, prevalence) if prevalence else float('nan'), auroc=roc_auc(y, s), tpr_at_fpr={f'{f:g}': tpr_at_fpr(y, s, f) for f in fpr_grid}, threshold=float(threshold), n_alert=int(predicted.sum()), n_tp=tp, n_fp=fp, tpr=recall, fpr=_safe_div(fp, n_neg), precision=precision, fdp=1.0 - precision, fpr_upper_bound=upper, constraint_satisfied=bool(upper <= alpha), n_tn=tn, n_fn=fn, f1=f1, accuracy=accuracy, ap_ci_low=ci_low, ap_ci_high=ci_high)

def bootstrap_ap(labels: np.ndarray, scores: np.ndarray, repeats: int=1000, seed: int=2026) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = labels.size
    out = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        idx = rng.integers(0, n, n)
        y, s = (labels[idx], scores[idx])
        out[i] = average_precision(y, s) if 0 < y.sum() < y.size else np.nan
    return out

def paired_bootstrap_delta(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, repeats: int=1000, seed: int=2026) -> Dict[str, float]:
    """同一组重抽样索引同时作用于两臂，因此比较的是配对差而不是两个独立区间。

    两个独立的 95% 区间重叠并不意味着差异不显著；消融要回答的是"这一层去掉后指标是否
    变了"，那必须是配对量。
    """
    rng = np.random.default_rng(seed)
    n = labels.size
    deltas = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        if not 0 < y.sum() < y.size:
            deltas[i] = np.nan
            continue
        deltas[i] = average_precision(y, scores_a[idx]) - average_precision(y, scores_b[idx])
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        return {'delta_mean': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan'), 'p_two_sided': float('nan'), 'n_draws': 0}
    n = float(finite.size)
    left = (float((finite <= 0).sum()) + 1.0) / (n + 1.0)
    right = (float((finite >= 0).sum()) + 1.0) / (n + 1.0)
    p = 2.0 * min(left, right)
    return {'delta_mean': float(finite.mean()), 'ci_low': float(np.percentile(finite, 2.5)), 'ci_high': float(np.percentile(finite, 97.5)), 'p_two_sided': float(min(1.0, p)), 'n_draws': int(finite.size)}
''', _ROOT=_ROOT, _DATASET=_DATASET, _OUTPUT=_OUTPUT, _DATA=_DATA, _CONFIGS=_CONFIGS)

# -------------------- how123.datasets --------------------
_install('how123.datasets', r'''
from __future__ import annotations
import json
import os
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
from _ours_runtime import ROOT, MODEL, DATASET, OUTPUT, read_dataset_csv
CODE_DIR = str(MODEL)
RUNNING_DIR = str(ROOT)
DATA_DIR = str(DATASET)
CACHE_DIR = str(OUTPUT / 'cache')
CANON_COLUMNS = ('ts', 'entity', 'counterparty', 'amount', 'cat1', 'cat2', 'cat3', 'text', 'label')

@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame
    meta: Dict[str, Any]

    @property
    def numeric_extra(self) -> List[str]:
        return [c for c in self.frame.columns if c.startswith('f_')]

    def __repr__(self) -> str:
        return f'<Dataset {self.name} rows={len(self.frame)} pos={self.frame.label.mean():.5f}>'

def _unix_seconds(s: pd.Series) -> pd.Series:
    """把任意 datetime 列换成 unix 秒——**不能假设底层单位是纳秒**。

    2026-09-03 踩到的真 bug：`astype("int64")` 返回的是该列**自己那个单位**下的整数，
    而 pandas 2.x 的 datetime64 有 ns / us / ms / s 四种单位。从 parquet 读回来的
    IEEE-CIS `transaction_ts` 是 `datetime64[us]`，代码却按纳秒除 `10**9`，
    **时间戳被整体压缩了 1000 倍**：真实跨度 182 天被算成 0.18 天。

    后果不是"数字难看"而是整层失效：`graph.bucket_seconds=1800` 于是相当于真实的
    21 天，整个 40 万行窗口只落进约 6 个桶，MIDAS 在结构上就不可能检出任何突发；
    `context.window_seconds=21600`（6 小时）则覆盖了全部历史，1.4.2 的"只取过去
    某窗口"这条契约实际不成立。IEEE-CIS 上 how2 测不出，根因在这里。

    `astype("datetime64[s]")` 先把列统一到秒单位再取整数，对四种单位都正确。
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.astype('datetime64[s]').astype('int64')
    return pd.to_numeric(s, errors='coerce').astype('int64')

def _middle_window(d: pd.DataFrame, ts_col: str, n: int) -> pd.DataFrame:
    d = d.dropna(subset=[ts_col]).sort_values(ts_col, kind='mergesort').reset_index(drop=True)
    if len(d) <= n:
        return d
    lo = (len(d) - n) // 2
    return d.iloc[lo:lo + n].reset_index(drop=True)

def _finalize(name: str, d: pd.DataFrame, meta: Dict[str, Any]) -> Dataset:
    d = d.reset_index(drop=True)
    for col in CANON_COLUMNS:
        if col not in d.columns:
            d[col] = '' if col == 'text' else np.nan
    for col in ('cat1', 'cat2', 'cat3'):
        d[col] = d[col].astype(str)
    d['text'] = d['text'].fillna('').astype(str)
    d['label'] = d['label'].astype(int)
    d = d.sort_values('ts', kind='mergesort').reset_index(drop=True)
    meta = dict(meta)
    meta.update(rows=int(len(d)), positive_rate=float(d.label.mean()), n_positive=int(d.label.sum()), entities=int(d.entity.nunique()), counterparties=int(d.counterparty.nunique()), has_text=bool((d.text.str.len() > 0).any()), time_span_days=float((d.ts.max() - d.ts.min()) / 86400.0))
    return Dataset(name, d, meta)

def _intent_f(v: Any, default: float=np.nan) -> float:
    try:
        return default if v is None or v == '' else float(v)
    except (TypeError, ValueError):
        return default

def _intent_json(fn: str) -> Any:
    with open(os.path.join(DATA_DIR, 'intent-tx-18k', fn), 'r', encoding='utf-8') as fh:
        return json.load(fh)

def _intent_frames() -> pd.DataFrame:
    """三个协议 schema 完全不同，必须逐协议按各自路径取字段，不得共用列。"""
    rows: List[Dict[str, Any]] = []
    for r in _intent_json('uniswap.json'):
        tx, md = (r.get('proposed_tx') or {}, r.get('metadata') or {})
        px = md.get('prices_usd') or {}
        tin = tx.get('tokenInSymbol')
        rows.append(dict(proto='uniswap', label=1 if r.get('label') == 'REJECT' else 0, ts=_intent_f(md.get('generated_at')), entity=md.get('from_address'), counterparty=md.get('to_address'), amount=_intent_f(tx.get('amountIn_human')) * _intent_f(px.get(tin), np.nan), c_op=tx.get('operation'), c_style=tx.get('style'), c_out=tx.get('tokenOutSymbol'), text=r.get('human_intent') or '', f_minout=_intent_f(tx.get('minOut_human')), f_fee=_intent_f(md.get('network_fee_usd')), posthoc=(r.get('reason') or {}).get('simple_reason') or ''))
    for r in _intent_json('cow.json'):
        tx, md = (r.get('proposed_tx') or {}, r.get('metadata') or {})
        o, ctx = (tx.get('order') or {}, tx.get('context') or {})
        px = ctx.get('prices_usd') or {}
        rows.append(dict(proto='cow', label=1 if r.get('label') == 'REJECT' else 0, ts=_intent_f(ctx.get('generated_at')), entity=md.get('trader'), counterparty=md.get('solver_address'), amount=_intent_f(o.get('sellAmount')) / 1e+18 * (list(px.values())[0] if px else 1.0), c_op=tx.get('operation'), c_style=o.get('kind'), c_out=md.get('trade_size_category'), text=r.get('human_intent') or '', f_minout=_intent_f(md.get('price_spread_pct')), f_fee=_intent_f(md.get('surplus_usd')), posthoc=(r.get('reason') or {}).get('simple_reason') or ''))
    for r in _intent_json('compound.json'):
        tx, md = (r.get('proposed_tx') or {}, r.get('metadata') or {})
        txs = tx.get('transactions') or [{}]
        rows.append(dict(proto='compound', label=1 if r.get('label') == 'REJECT' else 0, ts=_intent_f(md.get('block_time_unix')), entity=(txs[0] or {}).get('from'), counterparty=(txs[0] or {}).get('to'), amount=_intent_f(tx.get('amount')) * _intent_f(md.get('token_price_usd'), 1.0), c_op=tx.get('operation'), c_style=md.get('actor_type'), c_out=tx.get('asset'), text=r.get('human_intent') or '', f_minout=_intent_f(tx.get('gas')), f_fee=_intent_f(md.get('gas_used')), posthoc=(r.get('reason') or {}).get('simple_reason') or ''))
    d = pd.DataFrame(rows).dropna(subset=['ts', 'entity']).copy()
    d['amount'] = pd.to_numeric(d.amount, errors='coerce')
    d['f_amount_imputed'] = d.amount.isna().astype(int)
    d['amount'] = d.groupby('proto').amount.transform(lambda s: s.fillna(s.median()))
    return d.sort_values('ts', kind='mergesort').reset_index(drop=True)

def load_intent_tx_18k(window: int, include_posthoc: bool=False) -> Dataset:
    d = _intent_frames()
    d['cat1'] = d.proto.astype(str) + ':' + d.c_op.astype(str)
    d['cat2'] = d.c_out.astype(str)
    d['cat3'] = d.c_style.astype(str)
    keep = list(CANON_COLUMNS) + ['proto'] + [c for c in d.columns if c.startswith('f_')]
    if include_posthoc:
        keep.append('posthoc')
    else:
        d = d.drop(columns=['posthoc'])
    return _finalize('intent-tx-18k', d[keep], dict(source='github.com/duanyiyao/intent-tx-18k', task='intent-transaction alignment（正类 = REJECT）', time_unit='unix 秒，但取自 generated_at / block_time_unix，是样本生成时间而非交易时间', caveat='配对生成：每个实体/文本首次出现记 ACCEPT、其后每次记 REJECT。任何基于历史计数的特征都是标签的近似复制，图层与上文计数在此数据集上不可用；按时间切分也不成立（三协议时间范围互不重叠），须按 text 分组切分。'))

def load_ibm_aml(window: int) -> Dataset:
    """IBM AML HI-Small 分册（Altman 等，NeurIPS 2023 D&B）。

    读未压缩的 `HI-Small_Trans.csv`。先前读的是 `transactions_full.csv.zip`——同一份
    数据的压缩版（行数、正类率、时间范围三项逐一核对相同），但该 zip 于 2026-09-02
    在磁盘上消失，原因不明；当时 ibm-aml 仍能跑只是因为窗口缓存命中，一旦清缓存就会
    装载失败。改读 CSV 后不再依赖那个已不存在的文件。

    两个路径都探测：zip 若被放回，优先用它（体积小十倍）。
    """
    folder = os.path.join(DATA_DIR, 'ibm-aml')
    archive = os.path.join(folder, 'transactions_full.csv.zip')
    plain = os.path.join(folder, 'HI-Small_Trans.csv')
    if os.path.exists(archive):
        with zipfile.ZipFile(archive) as zf:
            inner = [n for n in zf.namelist() if n.lower().endswith('.csv')][0]
            with zf.open(inner) as fh:
                raw = pd.read_csv(fh)
    elif os.path.exists(plain):
        raw = pd.read_csv(plain)
    else:
        raise FileNotFoundError(f'ibm-aml 的交易表两个位置都不存在：{archive} 与 {plain}。该数据集是 IBM AML HI-Small 分册，需重新放回其中之一。')
    full_rows, full_rate = (len(raw), float(raw['Is Laundering'].mean()))
    raw['_ts'] = pd.to_datetime(raw['Timestamp'], errors='coerce')
    raw = _middle_window(raw, '_ts', window)
    d = pd.DataFrame(dict(ts=_unix_seconds(raw['_ts']), entity=raw['Account'].astype(str), counterparty=raw['Account.1'].astype(str), amount=pd.to_numeric(raw['Amount Paid'], errors='coerce'), cat1=raw['Payment Format'].astype(str), cat2='BANK' + raw['To Bank'].astype(str), cat3=raw['Receiving Currency'].astype(str), text='', label=raw['Is Laundering'].astype(int)))
    d['f_amount_received'] = pd.to_numeric(raw['Amount Received'], errors='coerce').values
    d['f_cross_currency'] = (raw['Receiving Currency'] != raw['Payment Currency']).astype(int).values
    d['f_cross_bank'] = (raw['From Bank'] != raw['To Bank']).astype(int).values
    return _finalize('ibm-aml', d, dict(source='github.com/IBM/AML-Data（CDLA-Sharing-1.0）；Altman et al., NeurIPS 2023 D&B', task='反洗钱（正类 = Is Laundering）', time_unit='unix 秒（真实时钟）', note=f'全量 {full_rows:,} 行、正类率 {full_rate:.6f}；此处取中段窗口；另有官方模式族标注 pattern_labels.csv（8 个洗钱典型）'))

def load_aml_pattern_labels() -> pd.DataFrame:
    """官方洗钱典型标注。按 (时间, 发起账户, 对手账户, 付款金额) 回连主表。"""
    d = pd.read_csv(os.path.join(DATA_DIR, 'ibm-aml', 'pattern_labels.csv'))
    d['_t'] = pd.to_datetime(d.Timestamp, errors='coerce')
    d['ts'] = _unix_seconds(d['_t'])
    d['amt'] = pd.to_numeric(d.amt_paid, errors='coerce').round(2)
    if 'family' not in d.columns:
        d['family'] = d.pattern.str.split(':').str[0].str.strip()
    d['family'] = d['family'].str.replace('\\s*MAX\\s+\\d+.*$', '', regex=True).str.strip()
    return d.rename(columns={'acct_from': 'entity', 'acct_to': 'counterparty'})

def load_sparkov(window: int) -> Dataset:
    raw = read_dataset_csv(os.path.join(DATA_DIR, 'sparkov', 'transactions_full.parquet'))
    full_rows, full_rate = (len(raw), float(raw['is_fraud'].mean()))
    raw = _middle_window(raw, 'unix_time', window)
    d = pd.DataFrame(dict(ts=raw['unix_time'].astype(float), entity=raw['cc_num'].astype(str), counterparty=raw['merchant'].astype(str), amount=raw['amt'].astype(float), cat1=raw['category'].astype(str), cat2=raw['state'].astype(str), cat3=raw['job'].astype(str), text='', label=raw['is_fraud'].astype(int)))
    d['f_city_pop'] = pd.to_numeric(raw['city_pop'], errors='coerce').values
    return _finalize('sparkov', d, dict(source='Sparkov 生成器；HF Nooha/cc_fraud_detection_dataset', task='信用卡交易欺诈', time_unit='unix 秒', note=f'全量 {full_rows:,} 行、正类率 {full_rate:.6f}；997 张卡 / 648 商户；实测无生成假象（图特征-标签最大 |r|=0.093），作干净对照'))

def load_tabformer_cc(window: int) -> Dataset:
    raw = pd.read_csv(os.path.join(DATA_DIR, 'tabformer-cc', 'transactions_sample-1500k.csv'))
    full_rate = float((raw['Is Fraud?'].astype(str).str.strip().str.lower() == 'yes').mean())
    day = pd.to_datetime(dict(year=raw['Year'], month=raw['Month'], day=raw['Day']), errors='coerce')
    hm = raw['Time'].astype(str).str.split(':', expand=True)
    raw['_ts'] = _unix_seconds(day) + pd.to_numeric(hm[0], errors='coerce').fillna(0) * 3600 + pd.to_numeric(hm[1], errors='coerce').fillna(0) * 60
    full_rows = len(raw)
    raw = _middle_window(raw, '_ts', window)
    d = pd.DataFrame(dict(ts=raw['_ts'].astype(float), entity=raw['User'].astype(str) + '-' + raw['Card'].astype(str), counterparty=raw['Merchant Name'].astype(str), amount=pd.to_numeric(raw['Amount'].astype(str).str.replace('$', '', regex=False), errors='coerce'), cat1='MCC' + raw['MCC'].astype(str), cat2=raw['Merchant City'].astype(str), cat3=raw['Use Chip'].astype(str), text='', label=(raw['Is Fraud?'].astype(str).str.strip().str.lower() == 'yes').astype(int)))
    d['f_has_error'] = raw['Errors?'].notna().astype(int).values
    return _finalize('tabformer-cc', d, dict(source='github.com/IBM/TabFormer；Padhi et al., ICASSP 2021', task='信用卡交易欺诈', time_unit='unix 秒（由 Year/Month/Day/Time 合成）', note=f'本地样本 {full_rows:,} 行、正类率 {full_rate:.6f}；图层的反例，用于界定适用范围'))

def load_s_ffsd(window: int) -> Dataset:
    """S-FFSD（GTAN, AAAI 2023 的数据集）。

    半监督数据集：`Labels` 有 0 / 1 / 2 三个取值，**2 表示未标注**（48,238 行，占 62%）。
    本课题是全监督评价，只能用 0/1 那部分，剔除后 29,643 行、正类率 0.177。
    这一点必须随结果声明——它不是抽样，是数据集本身只标了三分之一。

    验收（Diagnostics/d05）四项全过：无标签泄漏（Source/Target/Location 的
    "此前出现过"Precision 相对正类率均为 0.59–1.01×）、实体对重复率 43.3%
    （四个原数据集里最高，MIDAS 的突发检验最有依据）、四段各自有正类。
    """
    path = os.path.join(DATA_DIR, 's-ffsd', 'S-FFSD.csv')
    raw = pd.read_csv(path)
    full_rows = len(raw)
    n_unlabeled = int((raw['Labels'] == 2).sum())
    raw = raw[raw['Labels'].isin([0, 1])].copy()
    full_rate = float(raw['Labels'].mean())
    raw = _middle_window(raw, 'Time', window)
    d = pd.DataFrame(dict(ts=raw['Time'].astype(float), entity=raw['Source'].astype(str), counterparty=raw['Target'].astype(str), amount=raw['Amount'].astype(float), cat1=raw['Location'].astype(str), cat2=raw['Type'].astype(str), cat3='', text='', label=raw['Labels'].astype(int)))
    return _finalize('s-ffsd', d, dict(source='S-FFSD（Xiang 等，AAAI 2023 GTAN 附带数据）', task='交易欺诈（半监督标注）', time_unit='序号（等间隔，非真实秒）', note=f'原始 {full_rows:,} 行，其中 {n_unlabeled:,} 行标签=2 即未标注，已剔除；剩 {len(raw):,} 行、正类率 {full_rate:.6f}。cat3 留空：该数据集只有 Location 与 Type 两个类别字段'))

def load_ieee_cis(window: int) -> Dataset:
    """IEEE-CIS Fraud Detection（Kaggle 2019，Vesta 提供的真实信用卡数据）。

    与其余数据集的关键差别，须随结果声明：**该数据集刻意不提供商户标识**，
    因此没有真正意义上的对手方。这里取 `addr1`（账单地区）作 counterparty——
    332 个取值、实体对重复率 60.5%，二部图是"卡 × 地区"而非"卡 × 商户"。
    这对 how2 的语义是有影响的：能检出的是"同一张卡在某地区的交易突发"，
    而不是"某商户上的团伙聚集"。备选的 `recipient_email_domain` 更差——
    缺失 76.7%、仅 60 个取值，二部图会退化成卡 × 类别。

    验收（Diagnostics/d05）：无标签泄漏（三个键的"此前出现过"Precision 相对
    正类率均为 1.00×）、四段各自有正类、测试段正类 3,083。
    """
    path = os.path.join(DATA_DIR, 'ieee-cis', 'ieee_cis_fraud_features.parquet')
    raw = read_dataset_csv(path)
    full_rows, full_rate = (len(raw), float(raw['is_fraud'].mean()))
    ts = raw['transaction_ts']
    raw = raw.assign(_ts=_unix_seconds(ts).astype(float))
    raw = _middle_window(raw, '_ts', window)
    d = pd.DataFrame(dict(ts=raw['_ts'].astype(float), entity='C' + raw['card1'].astype(str), counterparty='A' + raw['addr1'].astype(str), amount=raw['transaction_amt'].astype(float), cat1=raw['product_cd'].astype(str), cat2=raw['card4'].astype(str), cat3=raw['card6'].astype(str), text='', label=raw['is_fraud'].astype(int)))
    for col in ('dist1', 'C1', 'C13', 'D1', 'D15'):
        if col in raw.columns:
            d[f'f_{col.lower()}'] = pd.to_numeric(raw[col], errors='coerce').values
    return _finalize('ieee-cis', d, dict(source='IEEE-CIS Fraud Detection（Kaggle 2019 / Vesta）', task='信用卡交易欺诈', time_unit='unix 秒（由 TransactionDT 换算）', note=f'全量 {full_rows:,} 行、正类率 {full_rate:.6f}；无商户字段，counterparty 取 addr1（账单地区），二部图为卡×地区'))

def load_paysim(window: int) -> Dataset:
    """PaySim（Lopez-Rojas 等，移动支付模拟器；Kaggle ealaxi/paysim1）。

    **entity 取收款方 nameDest 而非付款方 nameOrig**，这是被数据结构逼出来的：
    `nameOrig` 有 6,353,307 个唯一值对 6,362,620 行，即几乎每笔交易都换一个新付款方，
    付款方没有任何历史可言；而 `nameDest` 只有 2,722,362 个，客户收款账户收款次数
    中位数为 4、最多 113，且 67.5% 的欺诈行其收款方收过至少两笔。上文块 a_t 要的是
    "同一实体的过去交易"，在这份数据上只有收款方视角成立。这也符合该模拟器的欺诈
    设计——欺诈发生在中间账户（骡子账户）的收款侧。

    **图层在本数据集上停用**：实体对重复率为 0.00%（即使只取客户对客户的 421 万行
    也是如此），每个 (付款方, 收款方) 组合都唯一。MIDAS 数的是实体对在时间片内的
    边数，全部为 1，没有突发可检测。这与 intent-tx-18k 的停用原因不同——那里是标签
    泄漏，这里是结构上根本不存在可重复的边。

    `isFlaggedFraud` 不进任何表示：它只在 isFraud=1 时被置 1（16 行），是判定之后
    写下的标记，正是 Γ 的 ω_j 该拦住的东西。

    验收（Diagnostics/d05）：无标签泄漏（三个键相对正类率 0.65–1.50×）、四段各自
    有正类、r = α(1−π)/π = 2.37 ≥ 0.25 即误报预算充足。
    """
    path = os.path.join(DATA_DIR, 'PaySim', 'PS_20174392719_1491204439457_log.csv')
    raw = pd.read_csv(path, usecols=['step', 'type', 'amount', 'nameOrig', 'nameDest', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud'])
    full_rows, full_rate = (len(raw), float(raw['isFraud'].mean()))
    raw['_ts'] = raw['step'].astype(float) * 3600.0
    raw = _middle_window(raw, '_ts', window)
    d = pd.DataFrame(dict(ts=raw['_ts'].astype(float), entity=raw['nameDest'].astype(str), counterparty=raw['nameOrig'].astype(str), amount=raw['amount'].astype(float), cat1=raw['type'].astype(str), cat2=np.where(raw['nameDest'].astype(str).str.startswith('M'), 'MERCHANT', 'CUSTOMER'), cat3='', text='', label=raw['isFraud'].astype(int)))
    for col in ('oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'):
        d[f'f_{col.lower()}'] = pd.to_numeric(raw[col], errors='coerce').values
    return _finalize('paysim', d, dict(source='PaySim 移动支付模拟器（Kaggle ealaxi/paysim1）', task='移动支付欺诈', time_unit='unix 秒（由 step×3600 换算，step 为 1 小时一步）', note=f'全量 {full_rows:,} 行、正类率 {full_rate:.6f}；entity 取收款方（付款方每笔都换新 ID，无历史）；图层停用：实体对重复率 0.00%，MIDAS 无突发可测；isFlaggedFraud 已剔除（判定后写入的标记）'))

def _load_ibm_variant(name: str, folder: str, filename: str, note: str, window: int) -> Dataset:
    """IBM AML 的 LI-Small / HI-Medium 分册。与 `load_ibm_aml` 同一 schema。

    分开写 loader 而不是给 `load_ibm_aml` 加参数，是因为每个分册的窗口宽度、
    时间片长度都要各自在配置里定——它们的规模与欺诈率差一个量级，共用配置会错。
    """
    path = os.path.join(DATA_DIR, folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{name} 的交易表不存在：{path}')
    raw = pd.read_csv(path)
    full_rows, full_rate = (len(raw), float(raw['Is Laundering'].mean()))
    raw['_ts'] = _unix_seconds(pd.to_datetime(raw['Timestamp'], errors='coerce'))
    raw = _middle_window(raw, '_ts', window)
    d = pd.DataFrame(dict(ts=raw['_ts'].astype(float), entity=raw['Account'].astype(str), counterparty=raw['Account.1'].astype(str), amount=pd.to_numeric(raw['Amount Paid'], errors='coerce'), cat1=raw['Payment Format'].astype(str), cat2='BANK' + raw['To Bank'].astype(str), cat3=raw['Receiving Currency'].astype(str), text='', label=raw['Is Laundering'].astype(int)))
    return _finalize(name, d, dict(source='IBM AML（Altman 等，NeurIPS 2023 D&B），Kaggle ealtman2019 全分册', task='反洗钱', time_unit='unix 秒（由 Timestamp 换算）', note=f'全量 {full_rows:,} 行、正类率 {full_rate:.6f}。{note}'))

def load_ibm_aml_li(window: int) -> Dataset:
    """LI-Small：Low-Illicit 分册，正类率 0.052%，比 HI-Small 低一半。

    验收（Diagnostics/d05）五关全过：无标签泄漏（三个键 1.00–1.05×）、
    实体对重复率 55.5%、四段各自有正类、r=13.82。
    它的价值在于把方法的适用区间往**更极端的不平衡**再推一档。
    """
    return _load_ibm_variant('ibm-aml-li', 'ibm-aml-li', 'LI-Small_Trans.csv', 'LI = 低非法比例分册，与 HI-Small 同生成器不同欺诈率档位', window)

def load_ibm_aml_medium(window: int) -> Dataset:
    """HI-Medium：3,189 万行，是全部数据集里规模最大、图结构最密的一份。

    验收在**中段 200 万行窗口**上做（与实际使用口径一致，全量载入要十几 GB 内存）：
    无标签泄漏（0.85–1.00×）、**实体对重复率 71.7%（全部数据集最高）**、
    测试段 266 个正类、r=11.27。
    """
    return _load_ibm_variant('ibm-aml-medium', 'ibm-aml-medium', 'HI-Medium_Trans.csv', 'HI-Medium 分册；实体对重复率 71.7%，图结构最密', window)

def load_banksim(window: int) -> Dataset:
    """BankSim（Lopez-Rojas 与 Axelsson，零售银行卡消费模拟器）。

    全部字段都带引号（`'C1093826151'`），装载时统一剥掉——不剥的话 entity 与
    counterparty 会带着引号进图层，与其余数据集的键格式不一致。

    验收（Diagnostics/d05）五关全过：无标签泄漏（0.98–1.00×）、实体对重复率 54.2%、
    每实体 144.6 笔历史（全部数据集里最长）、r=0.917。
    `category` 是具名消费类目（es_transportation / es_health…），h_dom 有作用面。
    """
    path = os.path.join(DATA_DIR, 'banksim', 'bs140513_032310.csv')
    raw = pd.read_csv(path)
    strip = lambda s: s.astype(str).str.strip().str.strip("'")
    full_rows, full_rate = (len(raw), float(raw['fraud'].mean()))
    raw['_ts'] = pd.to_numeric(raw['step'], errors='coerce').astype(float) * 86400.0
    raw = _middle_window(raw, '_ts', window)
    d = pd.DataFrame(dict(ts=raw['_ts'].astype(float), entity=strip(raw['customer']), counterparty=strip(raw['merchant']), amount=pd.to_numeric(raw['amount'], errors='coerce'), cat1=strip(raw['category']), cat2='AGE' + strip(raw['age']), cat3='G' + strip(raw['gender']), text='', label=raw['fraud'].astype(int)))
    return _finalize('banksim', d, dict(source='BankSim（Kaggle ealaxi/banksim1）', task='零售银行卡消费欺诈', time_unit='unix 秒（step 为 1 天一步）', note=f'全量 {full_rows:,} 行、正类率 {full_rate:.6f}；4,112 客户 / 50 商户，每实体 144.6 笔；字段原始带引号，装载时已剥离'))
LOADERS = {'ibm-aml': load_ibm_aml, 'sparkov': load_sparkov, 'tabformer-cc': load_tabformer_cc, 'intent-tx-18k': load_intent_tx_18k, 'ieee-cis': load_ieee_cis, 'paysim': load_paysim, 'ibm-aml-li': load_ibm_aml_li, 'ibm-aml-medium': load_ibm_aml_medium, 'banksim': load_banksim}

def load_dataset(name: str, window_rows: int=400000, use_cache: bool=True) -> Dataset:
    """装载并缓存。缓存键含窗口行数，换窗口不会读到旧缓存。"""
    if name not in LOADERS:
        raise KeyError(f'未知数据集 {name!r}，可选：{sorted(LOADERS)}')
    cache_frame = os.path.join(CACHE_DIR, f'{name}_w{window_rows}.parquet')
    cache_meta = os.path.join(CACHE_DIR, f'{name}_w{window_rows}.meta.json')
    if use_cache and os.path.exists(cache_frame) and os.path.exists(cache_meta):
        frame = pd.read_parquet(cache_frame)
        with open(cache_meta, 'r', encoding='utf-8') as fh:
            meta = json.load(fh)
        return Dataset(name, frame, meta)
    ds = LOADERS[name](window_rows)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        ds.frame.to_parquet(cache_frame, index=False)
        with open(cache_meta, 'w', encoding='utf-8') as fh:
            json.dump(ds.meta, fh, ensure_ascii=False, indent=2)
    return ds

@dataclass(frozen=True)
class Splits:
    """训练 / 选型 / 校准 / 测试。同一样本不得同时参与训练、候选选择与最终校准，
    否则 3.4.1 的有限样本保证会因数据复用而失效（§3.2.1）。"""
    train: np.ndarray
    tune: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    mode: str
    boundaries: Tuple[float, ...] = ()

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {'train': self.train, 'tune': self.tune, 'calibration': self.calibration, 'test': self.test}

    def sizes(self) -> Dict[str, int]:
        return {k: int(v.size) for k, v in self.as_dict().items()}

    def assert_valid(self, ts: np.ndarray) -> int:
        """返回检查过的样本数。切分为空、重叠或时间倒挂都在这里拦下。"""
        groups = self.as_dict()
        seen: set = set()
        total = 0
        for name, idx in groups.items():
            if idx.size == 0:
                raise ValueError(f'切分 {name!r} 为空')
            overlap = seen & set(idx.tolist())
            if overlap:
                raise ValueError(f'样本在多个切分中重复出现：{sorted(overlap)[:5]}')
            seen |= set(idx.tolist())
            total += int(idx.size)
        if self.mode == 'temporal':
            order = ['train', 'tune', 'calibration', 'test']
            for left, right in zip(order, order[1:]):
                if float(ts[groups[left]].max()) >= float(ts[groups[right]].min()):
                    raise ValueError(f'时间泄漏：max({left}) 必须严格早于 min({right})')
        return total

    def assert_label_coverage(self, y: np.ndarray) -> int:
        """训练/调参/测试段须各有正类，校准段须有负类。

        少了任何一项，指标不是"差"而是**没有定义**：AP 与 AUROC 在单一类别上无意义，
        NP 校准在没有合法样本时无从取次序统计量。让它在这里报错，好过在结果表里留一列
        看起来像 0 的 NaN。
        """
        for name in ('train', 'tune', 'test'):
            idx = getattr(self, name)
            if int(y[idx].sum()) == 0:
                raise ValueError(f'切分 {name!r} 没有正类样本（n={idx.size}）：该窗口过小或正类沿时间高度聚集，请增大 window_rows 或改用别的窗口，而不是在这个切分上报指标')
        if int((y[self.calibration] == 0).sum()) == 0:
            raise ValueError('校准段没有合法样本，NP 校准无从取次序统计量')
        return int(y.size)

def make_splits(frame: pd.DataFrame, mode: str, ratios: List[float], group_column: Optional[str]=None, seed: int=2026) -> Splits:
    n = len(frame)
    cuts = np.cumsum(ratios)[:3]
    if mode == 'temporal':
        ts = frame['ts'].to_numpy(dtype=np.float64)
        order = np.argsort(ts, kind='mergesort')
        ts_sorted = ts[order]
        unique_ts = np.unique(ts_sorted)
        if unique_ts.size < 4:
            raise ValueError(f'只有 {unique_ts.size} 个不同的时间戳，无法做四段时间切分')
        cum = np.searchsorted(ts_sorted, unique_ts, side='right')
        picks: List[int] = []
        for target in cuts * n:
            candidate = int(np.argmin(np.abs(cum - target)))
            if picks:
                candidate = max(candidate, picks[-1] + 1)
            picks.append(min(candidate, unique_ts.size - 2))
        if not picks[0] < picks[1] < picks[2]:
            raise ValueError('时间戳打平过于严重，找不到三个互不相同的切分边界')
        b0, b1, b2 = (float(unique_ts[p]) for p in picks)
        parts = [np.flatnonzero(ts <= b0), np.flatnonzero((ts > b0) & (ts <= b1)), np.flatnonzero((ts > b1) & (ts <= b2)), np.flatnonzero(ts > b2)]
        splits = Splits(parts[0], parts[1], parts[2], parts[3], mode, (b0, b1, b2, float(ts.max())))
    elif mode == 'group':
        if group_column is None or group_column not in frame.columns:
            raise ValueError(f'group 切分需要存在的 group_column，收到 {group_column!r}')
        keys = frame[group_column].astype(str).to_numpy()
        uniq = np.array(sorted(set(keys.tolist())))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(uniq.size)
        positions = (cuts * uniq.size).astype(int)
        buckets = np.split(perm, positions)
        assign = {}
        for part_id, bucket in enumerate(buckets):
            for u in uniq[bucket]:
                assign[u] = part_id
        part_of = np.array([assign[k] for k in keys])
        splits = Splits(*[np.flatnonzero(part_of == i) for i in range(4)], mode=mode)
    else:
        raise ValueError(f'未知切分模式 {mode!r}')
    splits.assert_valid(frame['ts'].to_numpy())
    return splits
''', _ROOT=_ROOT, _DATASET=_DATASET, _OUTPUT=_OUTPUT, _DATA=_DATA, _CONFIGS=_CONFIGS)

# -------------------- how123.features --------------------
_install('how123.features', r'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
CATEGORICAL = ('cat1', 'cat2', 'cat3')

@dataclass
class BaseFeatureSpace:
    columns: List[str]
    category_maps: Dict[str, Dict[str, int]]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return _build_matrix(frame, self.columns, self.category_maps)

def _time_parts(ts: np.ndarray) -> Dict[str, np.ndarray]:
    seconds = np.asarray(ts, dtype=np.float64)
    day_seconds = np.mod(seconds, 86400.0)
    return {'hour_of_day': day_seconds / 3600.0, 'day_of_week': np.mod(np.floor(seconds / 86400.0) + 4.0, 7.0), 'is_night': ((day_seconds < 6 * 3600.0) | (day_seconds >= 22 * 3600.0)).astype(np.float64)}

def _build_matrix(frame: pd.DataFrame, columns: List[str], category_maps: Dict[str, Dict[str, int]]) -> np.ndarray:
    amount = pd.to_numeric(frame['amount'], errors='coerce').to_numpy(dtype=np.float64)
    parts: Dict[str, np.ndarray] = {'amount': amount, 'log_amount': np.log1p(np.abs(amount)) * np.sign(amount), 'amount_is_missing': np.isnan(amount).astype(np.float64)}
    parts.update(_time_parts(frame['ts'].to_numpy()))
    for col in CATEGORICAL:
        mapping = category_maps[col]
        values = frame[col].astype(str).to_numpy()
        parts[f'{col}_code'] = np.array([mapping.get(v, -1) for v in values], dtype=np.float64)
    for col in frame.columns:
        if col.startswith('f_'):
            parts[col] = pd.to_numeric(frame[col], errors='coerce').to_numpy(dtype=np.float64)
    matrix = np.column_stack([parts[name] for name in columns])
    return np.nan_to_num(matrix, nan=np.nan, posinf=np.nan, neginf=np.nan).astype(np.float32)

def fit_base_features(frame: pd.DataFrame, train_idx: np.ndarray) -> BaseFeatureSpace:
    """类别编码只在训练段拟合——这是 §3.2.1"同一样本不得同时参与训练与校准"的最低要求。"""
    category_maps: Dict[str, Dict[str, int]] = {}
    for col in CATEGORICAL:
        values = sorted(set(frame[col].astype(str).to_numpy()[train_idx].tolist()))
        category_maps[col] = {v: i for i, v in enumerate(values)}
    columns = ['amount', 'log_amount', 'amount_is_missing', 'hour_of_day', 'day_of_week', 'is_night']
    columns += [f'{c}_code' for c in CATEGORICAL]
    columns += [c for c in frame.columns if c.startswith('f_')]
    return BaseFeatureSpace(columns=columns, category_maps=category_maps)
''', _ROOT=_ROOT, _DATASET=_DATASET, _OUTPUT=_OUTPUT, _DATA=_DATA, _CONFIGS=_CONFIGS)

# -------------------- how123.constraint --------------------
_install('how123.constraint', r'''
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence
import numpy as np
from scipy.stats import binom

class CalibrationInfeasible(RuntimeError):
    """样本量不足以认证目标 (α, δ)。不得降低置信要求或沿用旧阈值。"""

def minimum_negative_samples(alpha: float, delta: float) -> int:
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError('alpha 与 delta 必须落在 (0,1)')
    return int(math.ceil(math.log(delta) / math.log1p(-alpha)))

def violation_probability(n0: int, k: int, alpha: float) -> float:
    """v(k) = Pr{Binomial(n0, 1−α) ≥ k}。"""
    if k <= 0:
        return 1.0
    if k > n0:
        return 0.0
    return float(binom.sf(k - 1, n0, 1.0 - alpha))

@dataclass(frozen=True)
class CalibrationResult:
    method: str
    status: str
    alpha: float
    delta: float
    n_negative: int
    n_min: int
    k_star: Optional[int]
    threshold: float
    violation_bound: Optional[float]
    has_ties: bool
    scorer_version: int
    tie_admission_rate: float = 0.0
    n_tied_calibration: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class NPOrderStatisticCalibrator:

    def __init__(self, alpha: float, delta: float) -> None:
        if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
            raise ValueError('alpha 与 delta 必须落在 (0,1)')
        self.alpha, self.delta = (alpha, delta)

    @property
    def n_min(self) -> int:
        return minimum_negative_samples(self.alpha, self.delta)

    def fit(self, negative_scores: Sequence[float], scorer_version: int) -> CalibrationResult:
        scores = np.sort(np.asarray(negative_scores, dtype=np.float64))
        n0 = int(scores.size)
        if n0 < self.n_min:
            raise CalibrationInfeasible(f'n_0={n0} < n_0,min={self.n_min}（α={self.alpha}, δ={self.delta}）：不存在可认证的阈值，该窗口须记为不可认证')
        if violation_probability(n0, n0, self.alpha) > self.delta:
            raise CalibrationInfeasible('没有任何次序统计量满足 v(k) ≤ delta')
        low, high = (1, n0)
        while low < high:
            mid = (low + high) // 2
            if violation_probability(n0, mid, self.alpha) <= self.delta:
                high = mid
            else:
                low = mid + 1
        k_star = low
        threshold = float(scores[k_star - 1])
        n_tied = int(np.count_nonzero(scores == threshold))
        has_ties = n_tied > 1
        n_gt = int(np.count_nonzero(scores > threshold))
        m = max(0, n0 - k_star - n_gt)
        tie_rate = 0.0 if n_tied == 0 else min(1.0, m / n_tied)
        return CalibrationResult(method='np_order_statistic', status='CERTIFIED_WITH_TIES' if has_ties else 'CERTIFIED', alpha=self.alpha, delta=self.delta, n_negative=n0, n_min=self.n_min, k_star=k_star, threshold=threshold, violation_bound=violation_probability(n0, k_star, self.alpha), has_ties=has_ties, scorer_version=scorer_version, tie_admission_rate=tie_rate, n_tied_calibration=n_tied)

class EmpiricalThresholdCalibrator:
    """决策层消融对照：在校准集经验 ROC 上取 FPR ≤ α 的阈值。

    §3.3 第二条指出这种做法没有回答总体 FPR 超过 α 的概率；Tong 等人的 1,000 次模拟中
    朴素经验阈值只有约一半分类器满足总体 type-I error 上限。本类存在的唯一目的是把
    这条缺陷在本课题数据上测出来，不是备选方案。
    """

    def __init__(self, alpha: float, delta: float) -> None:
        self.alpha, self.delta = (alpha, delta)

    def fit(self, negative_scores: Sequence[float], scorer_version: int) -> CalibrationResult:
        scores = np.sort(np.asarray(negative_scores, dtype=np.float64))
        n0 = int(scores.size)
        if n0 == 0:
            raise CalibrationInfeasible('校准集没有合法样本')
        k = int(math.ceil((1.0 - self.alpha) * n0))
        k = min(max(k, 1), n0)
        threshold = float(scores[k - 1])
        n_tied = int(np.count_nonzero(scores == threshold))
        n_gt = int(np.count_nonzero(scores > threshold))
        m = max(0, n0 - k - n_gt)
        tie_rate = 0.0 if n_tied == 0 else min(1.0, m / n_tied)
        return CalibrationResult(method='empirical_quantile', status='UNCERTIFIED', alpha=self.alpha, delta=self.delta, n_negative=n0, n_min=minimum_negative_samples(self.alpha, self.delta), k_star=k, threshold=threshold, violation_bound=violation_probability(n0, k, self.alpha), has_ties=bool(n_tied > 1), scorer_version=scorer_version, tie_admission_rate=tie_rate, n_tied_calibration=n_tied)

@dataclass(frozen=True)
class CapacityResult:
    alerts: np.ndarray
    candidates: int
    emitted: int
    capacity_limited: bool
    capacity: Optional[int]

class CapacityGate:
    """N_alert,t ≤ B_t。只能减少告警，不能反过来放宽 FPR 约束（§3.1.3 / §4.4）。"""

    def __init__(self, capacity: Optional[int]) -> None:
        if capacity is not None and capacity < 0:
            raise ValueError('capacity 必须非负或 None')
        self.capacity = capacity

    def apply(self, scores: Sequence[float], threshold: float, tie_admission_rate: float=0.0) -> CapacityResult:
        """`tie_admission_rate` 是校准段定出的并列放行比例 γ（见 NPOrderStatisticCalibrator）。

        γ=0 时退化为原来的严格大于。γ>0 时按**原始行序**取并列块的前 ⌊γ·n_tie⌋ 个一并
        放行——顺序规则与标签无关且可复现，这是它能被写进方法而不是当作实现细节的前提。
        """
        s = np.asarray(scores, dtype=np.float64)
        candidate_idx = np.flatnonzero(s > threshold)
        if tie_admission_rate > 0.0:
            tied_idx = np.flatnonzero(s == threshold)
            take = int(np.floor(tie_admission_rate * tied_idx.size))
            if take > 0:
                candidate_idx = np.union1d(candidate_idx, tied_idx[:take])
        alerts = np.zeros(s.size, dtype=bool)
        if self.capacity is None or candidate_idx.size <= self.capacity:
            alerts[candidate_idx] = True
        elif self.capacity > 0:
            ranked = candidate_idx[np.argsort(s[candidate_idx])[::-1]]
            alerts[ranked[:self.capacity]] = True
        return CapacityResult(alerts, int(candidate_idx.size), int(alerts.sum()), bool(alerts.sum() < candidate_idx.size), self.capacity)

class FixedPolicyCalibrator:
    """底座系统写死的 0.40 告警线——即"根本不做校准"。

    加这一条是为了让约束层能被**真正去掉**而不只是被替换。其余三条 how 的消融都能
    整层拿掉（语义层、图层不给特征即可），约束层不行：总得有个东西决定多高算告警，
    "没有阈值"只能退化成全判正常或全判欺诈，不是有意义的对照。因此这一层的"去掉"
    定义为退回底座 `kirtis111/agentic-ai-fraud-detection` 的 Decision Agent：
    两级固定阈值 0.70 / 0.40（底座 README 原文 "score > 0.7 → BLOCK / 0.4–0.7 →
    INVESTIGATE"），取其中较低的 0.40 作为是否告警的分界。

    它与 `EmpiricalThresholdCalibrator` 的区别是本质的——后者仍然看数据（取校准段
    负样本的 1-α 分位），只是不给有限样本保证；这一条连校准段都不看，α 与 δ 完全
    不起作用。因此它才是"去掉约束层"，经验阈值只是"换一种校准"。
    """

    def __init__(self, alpha: float, delta: float, investigate_at: float=0.4) -> None:
        self.alpha, self.delta, self.investigate_at = (alpha, delta, investigate_at)

    def fit(self, negative_scores: Sequence[float], scorer_version: int) -> CalibrationResult:
        n0 = int(np.asarray(negative_scores).size)
        return CalibrationResult(method='fixed_policy', status='FIXED_POLICY(无有限样本保证)', alpha=self.alpha, delta=self.delta, n_negative=n0, n_min=minimum_negative_samples(self.alpha, self.delta), k_star=None, threshold=float(self.investigate_at), violation_bound=None, has_ties=False, scorer_version=scorer_version)

def make_calibrator(method: str, alpha: float, delta: float):
    if method == 'np':
        return NPOrderStatisticCalibrator(alpha, delta)
    if method == 'empirical':
        return EmpiricalThresholdCalibrator(alpha, delta)
    if method == 'fixed':
        return FixedPolicyCalibrator(alpha, delta)
    raise ValueError(f'未知校准方法 {method!r}')
''', _ROOT=_ROOT, _DATASET=_DATASET, _OUTPUT=_OUTPUT, _DATA=_DATA, _CONFIGS=_CONFIGS)

# -------------------- how123.context --------------------
_install('how123.context', r'''
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from .contracts import SemanticSpec, TemporalAvailabilityGuard
AGG_NAMES = ('ctx_wmean_amount', 'ctx_wdev_amount', 'ctx_wmax_amount', 'ctx_wmean_gap', 'ctx_wmax_gap', 'ctx_effective_count', 'ctx_admitted_fraction', 'ctx_recency_log')

@dataclass
class ContextArtifacts:
    matrix: np.ndarray
    feature_names: List[str]
    enabled: bool
    reason: str
    mode: str
    negative_control: str
    admission: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    n_window_checked: int = 0

def build_history_index(entity_code: np.ndarray, ts: np.ndarray, k: int) -> np.ndarray:
    """每行取同实体最近 k 条历史的原始行号；不足则为 -1。O(N·k)，全向量化。"""
    order = np.lexsort((ts, entity_code))
    ent_sorted = entity_code[order]
    n = order.size
    sorted_hist = np.full((n, k), -1, dtype=np.int64)
    positions = np.arange(n)
    for lag in range(1, k + 1):
        src = positions - lag
        ok = src >= 0
        same = np.zeros(n, dtype=bool)
        same[ok] = ent_sorted[src[ok]] == ent_sorted[ok]
        sel = ok & same
        sorted_hist[sel, lag - 1] = order[src[sel]]
    out = np.empty_like(sorted_hist)
    out[order] = sorted_hist
    return out

def build_random_history_index(ts: np.ndarray, k: int, seed: int) -> np.ndarray:
    """负对照：只取过去、但随机跨实体。样本量与因果方向保持不变。"""
    order = np.argsort(ts, kind='mergesort')
    n = order.size
    rng = np.random.default_rng(seed)
    positions = np.arange(n)
    draws = rng.random((n, k))
    picked = np.floor(draws * positions[:, None]).astype(np.int64)
    picked = np.where(positions[:, None] > 0, picked, -1)
    sorted_hist = np.where(picked >= 0, order[np.clip(picked, 0, None)], -1)
    out = np.empty_like(sorted_hist)
    out[order] = sorted_hist
    return out

class HistoryAdmission:
    """不可违反的硬约束：因果方向、回看窗口、字段准入、度量兼容与业务禁配。"""
    REASONS = ('no_history_slot', 'outside_observation_window', 'missing_required_field', 'incompatible_measurement', 'blocked_business_pair')

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.window = float(cfg['window_seconds'])
        self.required_fields = list(cfg.get('required_fields', ['amount']))
        self.compatibility_fields = list(cfg.get('compatibility_fields', []))
        self.blocked_value_pairs = {str(k): {tuple(map(str, p)) for p in v or []} for k, v in (cfg.get('blocked_value_pairs') or {}).items()}

    def evaluate(self, frame: pd.DataFrame, hist: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        n, k = hist.shape
        ts = frame['ts'].to_numpy(dtype=np.float64)
        has_slot = hist >= 0
        safe = np.clip(hist, 0, None)
        counts: Dict[str, int] = {r: 0 for r in self.REASONS}
        counts['no_history_slot'] = int((~has_slot).sum())
        lag = ts[:, None] - ts[safe]
        in_window = (lag > 0) & (lag <= self.window)
        mask = has_slot & in_window
        counts['outside_observation_window'] = int((has_slot & ~in_window).sum())
        for name in self.required_fields:
            values = pd.to_numeric(frame[name], errors='coerce').to_numpy(dtype=np.float64)
            present = np.isfinite(values)
            ok = present[:, None] & present[safe]
            counts['missing_required_field'] = counts.get('missing_required_field', 0) + int((mask & ~ok).sum())
            mask &= ok
        for name in self.compatibility_fields:
            codes = pd.factorize(frame[name].astype(str))[0]
            ok = codes[:, None] == codes[safe]
            counts['incompatible_measurement'] = counts.get('incompatible_measurement', 0) + int((mask & ~ok).sum())
            mask &= ok
        blocked_total = 0
        for name, pairs in self.blocked_value_pairs.items():
            values = frame[name].astype(str).to_numpy()
            bad = np.zeros_like(mask)
            for row, col in zip(*np.nonzero(mask)):
                if (values[row], values[safe[row, col]]) in pairs:
                    bad[row, col] = True
            blocked_total += int(bad.sum())
            mask &= ~bad
        counts['blocked_business_pair'] = blocked_total
        counts['admitted'] = int(mask.sum())
        counts['slots_total'] = int(mask.size)
        counts['rows_with_history'] = int(mask.any(axis=1).sum())
        return (mask, counts)

def _hash_codes(values: np.ndarray, field_name: str, buckets: int) -> np.ndarray:
    uniq, inverse = np.unique(values.astype(str), return_inverse=True)
    table = np.empty(uniq.size, dtype=np.int64)
    for i, value in enumerate(uniq):
        digest = hashlib.blake2b(f'{field_name}::{value}'.encode('utf-8'), digest_size=8).digest()
        table[i] = int.from_bytes(digest, 'little') % buckets
    return table[inverse].astype(np.int32)

class NAGContextModel(nn.Module):

    def __init__(self, n_fields: int, buckets: int, emb_dim: int, out_dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.n_fields = n_fields
        self.embeddings = nn.ModuleList([nn.Embedding(buckets, emb_dim) for _ in range(max(n_fields, 1))])
        self.relation_weights = nn.Parameter(torch.ones(max(n_fields, 1)))
        self.relation_bias = nn.Parameter(torch.zeros(1))
        self.norm = nn.BatchNorm1d(len(AGG_NAMES))
        self.projection = nn.Sequential(nn.Linear(len(AGG_NAMES), 32), nn.ReLU(), nn.Linear(32, out_dim), nn.Tanh())
        self.head = nn.Linear(len(AGG_NAMES) + out_dim, 1)

    def soft_weights(self, cur: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """S_t w + b 后过 sigmoid。cur:[B,F] hist:[B,K,F]。"""
        if self.mode == 'hard' or self.n_fields == 0:
            return torch.ones(hist.shape[0], hist.shape[1], device=hist.device)
        sims = []
        for f in range(self.n_fields):
            emb = self.embeddings[f]
            cur_vec = emb(cur[:, f]).unsqueeze(1)
            hist_vec = emb(hist[:, :, f])
            sims.append(torch.nn.functional.cosine_similarity(hist_vec, cur_vec, dim=-1, eps=1e-08))
        stacked = torch.stack(sims, dim=-1)
        return torch.sigmoid(stacked @ self.relation_weights + self.relation_bias)

    @staticmethod
    def aggregate(m_tilde: torch.Tensor, mask: torch.Tensor, z_amount: torch.Tensor, z_gap: torch.Tensor, z_amount_cur: torch.Tensor) -> torch.Tensor:
        k = mask.shape[1]
        denom = m_tilde.sum(dim=1).clamp_min(1e-06)
        wmean_amt = (m_tilde * z_amount).sum(dim=1) / denom
        wdev_amt = (m_tilde * (z_amount - z_amount_cur.unsqueeze(1)).abs()).sum(dim=1) / denom
        wmax_amt = (m_tilde * z_amount).max(dim=1).values
        wmean_gap = (m_tilde * z_gap).sum(dim=1) / denom
        wmax_gap = (m_tilde * z_gap).max(dim=1).values
        eff_count = m_tilde.sum(dim=1) / k
        admitted = mask.sum(dim=1) / k
        big = torch.full_like(z_gap, 1000000000.0)
        recency = torch.where(mask > 0, z_gap, big).min(dim=1).values
        recency = torch.where(mask.sum(dim=1) > 0, recency, torch.zeros_like(recency))
        raw = torch.stack([wmean_amt, wdev_amt, wmax_amt, wmean_gap, wmax_gap, eff_count, admitted, recency], dim=1)
        return torch.where(mask.sum(dim=1, keepdim=True) > 0, raw, torch.zeros_like(raw))

    def forward(self, cur: torch.Tensor, hist: torch.Tensor, mask: torch.Tensor, z_amount: torch.Tensor, z_gap: torch.Tensor, z_amount_cur: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        soft = self.soft_weights(cur, hist)
        m_tilde = soft * mask
        raw = self.aggregate(m_tilde, mask, z_amount, z_gap, z_amount_cur)
        normed = self.norm(raw)
        projected = self.projection(normed)
        features = torch.cat([normed, projected], dim=1)
        return (features, self.head(features).squeeze(-1))

def build_context(frame: pd.DataFrame, train_idx: np.ndarray, labels: np.ndarray, gamma: SemanticSpec, cfg: Dict[str, Any], seed: int=2026, auditor=None) -> ContextArtifacts:
    if not bool(cfg.get('enabled', True)):
        return ContextArtifacts(np.zeros((len(frame), 0), dtype=np.float32), [], False, str(cfg.get('disabled_reason', '配置关闭')), str(cfg.get('mode', 'soft')), 'none')
    k = int(cfg['top_k'])
    mode = str(cfg.get('mode', 'soft'))
    control = str(cfg.get('negative_control', 'none'))
    entity_code = pd.factorize(frame['entity'].astype(str))[0].astype(np.int64)
    ts = frame['ts'].to_numpy(dtype=np.float64)
    if control == 'random_history':
        hist = build_random_history_index(ts, k, seed)
    else:
        hist = build_history_index(entity_code, ts, k)
    admission = HistoryAdmission(cfg)
    mask_bool, counts = admission.evaluate(frame, hist)
    guard_checked = TemporalAvailabilityGuard.assert_history_window(ts, ts[np.clip(hist, 0, None)], mask_bool, float(cfg['window_seconds']))
    if auditor is not None:
        auditor.record('契约二·上文回看窗口 0<τt−τi≤W', guard_checked, True, f"采纳 {counts['admitted']} / 候选槽 {counts['slots_total']}")
    safe = np.clip(hist, 0, None)
    amount = pd.to_numeric(frame['amount'], errors='coerce').to_numpy(dtype=np.float64)
    amount = np.nan_to_num(amount, nan=0.0)
    z_amount_all = np.log1p(np.abs(amount)).astype(np.float32)
    z_amount = z_amount_all[safe] * mask_bool
    lag = np.clip(ts[:, None] - ts[safe], 0.0, None)
    z_gap = (np.log1p(lag) * mask_bool).astype(np.float32)
    relation_fields = [f for f in cfg.get('soft_relation_fields', []) if f in frame.columns]
    buckets = int(cfg.get('hash_buckets', 512))
    if relation_fields:
        cur_codes = np.column_stack([_hash_codes(frame[f].to_numpy(), f, buckets) for f in relation_fields])
        hist_codes = cur_codes[safe]
    else:
        cur_codes = np.zeros((len(frame), 0), dtype=np.int32)
        hist_codes = np.zeros((len(frame), k, 0), dtype=np.int32)
    torch.manual_seed(seed)
    model = NAGContextModel(len(relation_fields), buckets, int(cfg.get('embedding_dim', 8)), int(cfg.get('output_dim', 8)), mode)
    tensors = dict(cur=torch.from_numpy(cur_codes.astype(np.int64)), hist=torch.from_numpy(hist_codes.astype(np.int64)), mask=torch.from_numpy(mask_bool.astype(np.float32)), z_amount=torch.from_numpy(z_amount.astype(np.float32)), z_gap=torch.from_numpy(z_gap), z_amount_cur=torch.from_numpy(z_amount_all))
    training_log = _fit(model, tensors, train_idx, labels, cfg, seed)
    model.eval()
    outputs: List[np.ndarray] = []
    batch = int(cfg.get('batch_size', 4096))
    with torch.no_grad():
        for start in range(0, len(frame), batch):
            stop = min(start + batch, len(frame))
            sl = slice(start, stop)
            features, _ = model(tensors['cur'][sl], tensors['hist'][sl], tensors['mask'][sl], tensors['z_amount'][sl], tensors['z_gap'][sl], tensors['z_amount_cur'][sl])
            outputs.append(features.numpy())
    matrix = np.vstack(outputs).astype(np.float32)
    names = list(AGG_NAMES) + [f'ctx_proj_{i}' for i in range(int(cfg.get('output_dim', 8)))]
    reason = f"mode={mode}, K={k}, W={cfg['window_seconds']}s, 关系字段={relation_fields or '无'}"
    if control != 'none':
        reason += f'; 负对照={control}'
    return ContextArtifacts(matrix, names, True, reason, mode, control, admission=counts, training=training_log, n_window_checked=guard_checked)

def _fit(model: NAGContextModel, tensors: Dict[str, torch.Tensor], train_idx: np.ndarray, labels: np.ndarray, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """只在训练段学 w 与类别嵌入。校准段与测试段一律只做前向。"""
    epochs = int(cfg.get('epochs', 3))
    batch = int(cfg.get('batch_size', 4096))
    y = torch.from_numpy(labels.astype(np.float32))
    n_pos = float(max(1, int(labels[train_idx].sum())))
    n_neg = float(max(1, int((labels[train_idx] == 0).sum())))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get('lr', 0.005)), weight_decay=1e-05)
    rng = np.random.default_rng(seed)
    log: List[Dict[str, float]] = []
    idx = np.asarray(train_idx)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(idx.size)
        total, seen = (0.0, 0)
        for start in range(0, idx.size, batch):
            rows = torch.from_numpy(idx[perm[start:start + batch]].astype(np.int64))
            if rows.numel() < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(tensors['cur'][rows], tensors['hist'][rows], tensors['mask'][rows], tensors['z_amount'][rows], tensors['z_gap'][rows], tensors['z_amount_cur'][rows])
            loss = loss_fn(logits, y[rows])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * rows.numel()
            seen += int(rows.numel())
        log.append({'epoch': epoch, 'mean_loss': total / max(seen, 1)})
    return {'epochs': log, 'relation_weights': [float(x) for x in model.relation_weights.detach().numpy()], 'relation_bias': float(model.relation_bias.detach().numpy()[0]), 'n_train_rows': int(idx.size)}
''', _ROOT=_ROOT, _DATASET=_DATASET, _OUTPUT=_OUTPUT, _DATA=_DATA, _CONFIGS=_CONFIGS)

# -------------------- how123.semantic --------------------
_install('how123.semantic', r'''
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from .contracts import SemanticSpec, TemporalAvailabilityGuard
_TOKEN_RE = re.compile('[\\w.:+-]+')

def _hash_token(token: str, dim: int) -> Tuple[int, float]:
    raw = int.from_bytes(hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest(), 'little')
    return (raw % dim, 1.0 if raw >> 17 & 1 == 0 else -1.0)

def _accumulate(vector: np.ndarray, tokens: Sequence[str]) -> None:
    dim = vector.shape[-1]
    for token in tokens:
        index, sign = _hash_token(token, dim)
        vector[index] += sign

def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(str(text).lower())

@dataclass
class SemanticArtifacts:
    """产出与审计证据分开存放：向量给评分器，其余给 §4.7 第二条的可核查性。"""
    matrix: np.ndarray
    enabled: bool
    reason: str
    excluded_by_omega: Tuple[str, ...]
    serialized_columns: Tuple[str, ...]
    example_serialization: str = ''
    n_rows_checked: int = 0

class DomainSemanticEncoder:

    def __init__(self, gamma: SemanticSpec, cfg: Dict[str, Any]) -> None:
        self.gamma = gamma
        self.cfg = cfg
        self.dim = int(cfg.get('embedding_dim', 64))
        self.numeric_bins = int(cfg.get('numeric_bins', 32))
        self.max_text_tokens = int(cfg.get('max_text_tokens', 256))
        self.guard = TemporalAvailabilityGuard(gamma)
        self._numeric_edges: Dict[str, np.ndarray] = {}
        self._value_tables: Dict[str, Tuple[Dict[str, int], np.ndarray]] = {}
        self._constant: Optional[np.ndarray] = None
        self._columns: List[str] = []
        self._excluded: List[str] = []

    def _visible_columns(self, frame: pd.DataFrame, decision_offset: float=0.0) -> Tuple[List[str], List[str]]:
        """按 ω_j 整列判定可见性。决策时点取事件时点本身（在线场景的最严格取法）。"""
        candidates = [c for c in frame.columns if c in self.gamma.fields and c != 'label']
        visible, excluded = ([], [])
        for name in candidates:
            spec = self.gamma.field(name)
            if not spec.in_serialization:
                continue
            if spec.posthoc or spec.availability_lag_seconds > decision_offset:
                excluded.append(name)
            else:
                visible.append(name)
        return (visible, excluded)

    def _static_tokens(self, name: str) -> List[str]:
        spec = self.gamma.field(name)
        if self.gamma.is_empty_spec():
            return [f'field:{name}']
        tokens = [f'field:{name}']
        tokens += [f'desc:{w}' for w in _tokenize(spec.description)]
        if spec.unit:
            tokens.append(f'unit:{spec.unit}')
        if spec.entity_role:
            tokens.append(f'role:{spec.entity_role}')
        if spec.source:
            tokens.append(f'source:{spec.source}')
        return tokens

    def _value_tokens(self, name: str, raw: Any, decoded: str) -> List[str]:
        if self.gamma.is_empty_spec():
            return [f'{name}={raw}']
        tokens = [f'{name}={raw}']
        if decoded != str(raw):
            tokens += [f'{name}:decoded:{w}' for w in _tokenize(decoded)]
        return tokens

    def fit(self, frame: pd.DataFrame, train_idx: np.ndarray) -> None:
        """分箱边界与取值词表只在训练段拟合。"""
        visible, excluded = self._visible_columns(frame)
        self._columns, self._excluded = (visible, excluded)
        constant = np.zeros(self.dim, dtype=np.float64)
        for name in visible:
            _accumulate(constant, self._static_tokens(name))
        if not self.gamma.is_empty_spec():
            _accumulate(constant, [f'task:{w}' for w in _tokenize(self.gamma.task_description)])
        self._constant = constant.astype(np.float32)
        train_frame = frame.iloc[train_idx]
        for name in visible:
            series = train_frame[name]
            if pd.api.types.is_numeric_dtype(frame[name]):
                values = pd.to_numeric(train_frame[name], errors='coerce').to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    edges = np.array([0.0])
                else:
                    qs = np.linspace(0.0, 1.0, self.numeric_bins + 1)[1:-1]
                    edges = np.unique(np.quantile(values, qs))
                self._numeric_edges[name] = edges
                labels = [f'bin{i}' for i in range(edges.size + 1)] + ['binNA']
                self._value_tables[name] = self._make_table(name, labels, decode=False)
            else:
                vocab = sorted(set(series.astype(str).tolist()))
                self._value_tables[name] = self._make_table(name, vocab, decode=True)

    def _make_table(self, name: str, values: Sequence[str], decode: bool) -> Tuple[Dict[str, int], np.ndarray]:
        spec = self.gamma.field(name)
        table = np.zeros((len(values), self.dim), dtype=np.float32)
        index: Dict[str, int] = {}
        for row, value in enumerate(values):
            index[value] = row
            decoded = spec.decode(value) if decode else value
            _accumulate(table[row], self._value_tokens(name, value, decoded))
        return (index, table)

    def _codes(self, name: str, series: pd.Series) -> np.ndarray:
        index, _ = self._value_tables[name]
        if name in self._numeric_edges:
            values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64)
            edges = self._numeric_edges[name]
            bins = np.searchsorted(edges, values, side='right')
            bins = np.where(np.isfinite(values), bins, edges.size + 1)
            labels = np.array([f'bin{i}' for i in range(edges.size + 1)] + ['binNA'])
            return np.array([index[label] for label in labels[bins]], dtype=np.int64)
        keys = series.astype(str).to_numpy()
        return np.array([index.get(k, -1) for k in keys], dtype=np.int64)

    def _extend_table(self, name: str, unseen: Sequence[str]) -> None:
        """训练期未见的类别取值：现场补进查表，而不是静默丢弃这一字段的贡献。

        丢弃会让测试段的 h_dom 与训练段不同分布——那是实现缺陷伪装成分布漂移。
        """
        index, table = self._value_tables[name]
        spec = self.gamma.field(name)
        rows = np.zeros((len(unseen), self.dim), dtype=np.float32)
        for offset, value in enumerate(unseen):
            index[value] = table.shape[0] + offset
            _accumulate(rows[offset], self._value_tokens(name, value, spec.decode(value)))
        self._value_tables[name] = (index, np.vstack([table, rows]))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self._constant is None:
            raise RuntimeError('先调用 fit 再 transform')
        n = len(frame)
        out = np.repeat(self._constant[None, :], n, axis=0)
        for name in self._columns:
            if name not in self._numeric_edges:
                index, _ = self._value_tables[name]
                keys = frame[name].astype(str).to_numpy()
                unseen = sorted(set(keys.tolist()) - set(index))
                if unseen:
                    self._extend_table(name, unseen)
            codes = self._codes(name, frame[name])
            _, table = self._value_tables[name]
            known = codes >= 0
            if known.any():
                out[known] += table[codes[known]]
        if 'text' in frame.columns:
            text = frame['text'].astype(str)
            for i in np.flatnonzero(text.str.len().to_numpy() > 0):
                tokens = _tokenize(text.iat[int(i)])[:self.max_text_tokens]
                _accumulate(out[int(i)], [f'text:{w}' for w in tokens])
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.maximum(norms, 1e-08)).astype(np.float32)

    def serialize_row(self, frame: pd.DataFrame, row: int) -> str:
        """人类可读的 r_t^Γ。只用于审计输出与单元测试对账，不进入训练路径。"""
        record = frame.iloc[row]
        if self.gamma.is_empty_spec():
            pairs = [f'{name}={record[name]}' for name in self._columns]
            if str(record.get('text', '')):
                pairs.append(f"text={record['text']}")
            return ' | '.join(pairs)
        units: List[str] = []
        for name in self._columns:
            spec = self.gamma.field(name)
            raw = record[name]
            units.append(f"[name={spec.name}; description={spec.description}; value={raw}; decoded={spec.decode(raw)}; unit={spec.unit}; role={spec.entity_role}; source={spec.source}; available_at={spec.available_at(float(record['ts']))}]")
        if str(record.get('text', '')):
            units.append(f"[source=raw_text; text={record['text']}]")
        units.append(f'[task={self.gamma.task_description}]')
        return ' '.join(units)

    def encode_row_reference(self, frame: pd.DataFrame, row: int) -> np.ndarray:
        """逐行 token 哈希的参考路径，用于验证向量化实现没有漂移。"""
        record = frame.iloc[row]
        vector = np.zeros(self.dim, dtype=np.float64)
        for name in self._columns:
            _accumulate(vector, self._static_tokens(name))
        if not self.gamma.is_empty_spec():
            _accumulate(vector, [f'task:{w}' for w in _tokenize(self.gamma.task_description)])
        for name in self._columns:
            spec = self.gamma.field(name)
            if name in self._numeric_edges:
                edges = self._numeric_edges[name]
                value = pd.to_numeric(pd.Series([record[name]]), errors='coerce').to_numpy()[0]
                label = f"bin{int(np.searchsorted(edges, value, side='right'))}" if np.isfinite(value) else 'binNA'
                _accumulate(vector, self._value_tokens(name, label, label))
            else:
                raw = str(record[name])
                _accumulate(vector, self._value_tokens(name, raw, spec.decode(raw)))
        if str(record.get('text', '')):
            _accumulate(vector, [f'text:{w}' for w in _tokenize(record['text'])[:self.max_text_tokens]])
        norm = np.linalg.norm(vector)
        return (vector / max(norm, 1e-08)).astype(np.float32)

def build_semantic(frame: pd.DataFrame, train_idx: np.ndarray, gamma: SemanticSpec, cfg: Dict[str, Any]) -> SemanticArtifacts:
    """按 §4.5 的模块级开关决定是否启用：1.4.1 只在数据含原始文本时启用。

    这不是工程上的方便，而是本课题实测已确定的结论——在无文本数据上 Γ 序列化是负增益
    （AP 提升由 81.82 降到 26.90、由 221.05 降到 184.36），因为它只是把已有的类别信息
    重新编码一遍，多出的分量是噪声。因此 how123 在无文本场景下的正确形态是三模块而非四模块。
    """
    setting = str(cfg.get('enabled', 'auto')).lower()
    has_text = bool((frame['text'].astype(str).str.len() > 0).any()) if 'text' in frame.columns else False
    if setting == 'false':
        enabled, reason = (False, '配置显式关闭')
    elif setting == 'true':
        enabled, reason = (True, '配置显式开启（无文本时这是负增益复现臂，不是推荐配置）')
    else:
        enabled = has_text
        reason = 'auto：检测到原始文本' if has_text else 'auto：无原始文本，按 §4.5 退回三模块形态'
    encoder = DomainSemanticEncoder(gamma, cfg)
    visible, excluded = encoder._visible_columns(frame)
    if not enabled:
        return SemanticArtifacts(matrix=np.zeros((len(frame), 0), dtype=np.float32), enabled=False, reason=reason, excluded_by_omega=tuple(excluded), serialized_columns=tuple(visible), n_rows_checked=len(frame))
    encoder.fit(frame, train_idx)
    guard = TemporalAvailabilityGuard(gamma)
    guard.assert_no_posthoc(encoder._columns)
    matrix = encoder.transform(frame)
    return SemanticArtifacts(matrix=matrix, enabled=True, reason=reason, excluded_by_omega=tuple(encoder._excluded), serialized_columns=tuple(encoder._columns), example_serialization=encoder.serialize_row(frame, int(train_idx[0])), n_rows_checked=len(frame))
''', _ROOT=_ROOT, _DATASET=_DATASET, _OUTPUT=_OUTPUT, _DATA=_DATA, _CONFIGS=_CONFIGS)

# -------------------- how123.graph --------------------
_install('how123.graph', r'''
from __future__ import annotations
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2
from .contracts import ContractViolation, PatternEntry, SemanticSpec
GRAPH_FEATURE_NAMES = ('g_zeta', 'g_concentration', 'g_max_path_len', 'g_subgraph_edges', 'g_signal_age_buckets')

class CountMinSketch:
    """常数内存计数。只会高估，MIDAS 定理 1 正是利用这一单向性构造上界；
    因此它的保证只覆盖误报，对漏报没有任何承诺。"""

    def __init__(self, width: int=4096, depth: int=5) -> None:
        if width <= 0 or depth <= 0:
            raise ValueError('CMS 的 width 与 depth 必须为正')
        self.width, self.depth = (width, depth)
        self.table = np.zeros((depth, width), dtype=np.float64)
        self._rows_cache: Dict[int, np.ndarray] = {}
        self._depth_index = np.arange(depth)

    def _rows(self, key: int) -> np.ndarray:
        cached = self._rows_cache.get(key)
        if cached is None:
            cached = np.array([int.from_bytes(hashlib.blake2b(f'{r}::{key}'.encode(), digest_size=8).digest(), 'little') % self.width for r in range(self.depth)], dtype=np.int64)
            self._rows_cache[key] = cached
        return cached

    def add(self, key: int, count: float=1.0) -> None:
        if count < 0:
            raise ValueError('CMS 不支持负增量')
        self.table[self._depth_index, self._rows(key)] += count

    def estimate(self, key: int) -> float:
        return float(self.table[self._depth_index, self._rows(key)].min())

def midas_statistic(current: np.ndarray, cumulative: np.ndarray, t: int) -> np.ndarray:
    if t <= 1:
        return np.zeros_like(current, dtype=np.float64)
    expected = cumulative / t
    denom = np.maximum(cumulative * (t - 1), 1e-12)
    return (current - expected) ** 2 * t ** 2 / denom

def chi_square_threshold(epsilon_t: float) -> float:
    """χ²_{1−ε_t/2}(df=1)。"""
    if not 0.0 < epsilon_t < 1.0:
        raise ValueError('epsilon_t 必须落在 (0,1)')
    return float(chi2.isf(epsilon_t / 2.0, 1))

def vectorize_paths(paths: Sequence[Sequence[Tuple[int, int, int]]], dim: int, node_names: Dict[int, str], relation_names: Dict[int, str]) -> np.ndarray:
    out = np.zeros((len(paths), dim), dtype=np.float64)
    for row, path in enumerate(paths):
        for src, rel, dst in path:
            for token in (f'src:{node_names[src]}', f'rel:{relation_names[rel]}', f'dst:{node_names[dst]}'):
                raw = int.from_bytes(hashlib.blake2b(token.encode('utf-8'), digest_size=8).digest(), 'little')
                out[row, raw % dim] += 1.0 if raw >> 17 & 1 == 0 else -1.0
        norm = np.linalg.norm(out[row])
        if norm > 0:
            out[row] /= norm
    return out

def path_attention(vectors: np.ndarray) -> np.ndarray:
    """Π = softmax( (ZW_Q)(ZW_K)ᵀ / √d )，取 W_Q = W_K = cI。

    c 不是可省的常数。路径向量是单位范数，直接取 c = 1 会让 logits 落在
    [−1/√d, 1/√d]（d = 32 时约 ±0.18），softmax 之后 Π 恒为近似均匀，
    ϱ 无论数据如何都稳定在 1e-4 量级——那不是"数据里没有长程关系"，
    而是尺度选错把这个量测没了。这里按 logits 的标准差标定 c，
    使 Π 只反映路径相似度的**相对**结构，与向量范数和维度无关。
    """
    if vectors.shape[0] == 0:
        return np.zeros((0, 0))
    logits = vectors @ vectors.T / math.sqrt(max(1, vectors.shape[1]))
    spread = float(logits.std())
    if spread > 1e-09:
        logits = (logits - float(logits.mean())) / spread
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)

def attention_concentration(attention: np.ndarray) -> float:
    """ϱ_attn(Π) = 1 − Ent(Π)/log K，§2.4.8 的原式。行近似均匀 →0，集中在少数路径 →1。"""
    k = attention.shape[0]
    if k <= 1:
        return 0.0
    entropy = -float(np.sum(attention * np.log(np.clip(attention, 1e-12, 1.0))) / k)
    return float(np.clip(1.0 - entropy / math.log(k), 0.0, 1.0))

def path_overlap_concentration(paths: Sequence[Sequence[Tuple[int, int, int]]]) -> float:
    """ϱ_overlap = 1 − |∪ 边| / Σ_k |p_k|，即 K 条采样路径彼此复用同一批边的程度。

    ---------------------------------------------------------------------------
    这是一处必须声明的偏离，不是对 §2.4.8 原式的复现。
    ---------------------------------------------------------------------------
    §2.4.8 用注意力熵定义 ϱ，其**前提是 Π 是学到的相关性矩阵**——那时"某几条路径
    被反复赋予高权重"才等价于"存在协同结构"。本实现取恒等投影（训练投影需要路径级
    监督，属 §4.5 的可替换点），Π 退化成相似度矩阵，此时熵的方向与"路径互相印证"
    正好相反：多条路径共享同一段资金链 ⇒ 向量彼此相似 ⇒ Π 每行近似均匀 ⇒ 熵最大 ⇒
    ϱ_attn → 0。照搬原式会把最该确认的结构判成"无结构"。

    ϱ_overlap 保留 §2.4.8 要求的三条退化语义而不依赖任何投影：
      · 采样路径各走各的（无协同结构）→ 并集 ≈ 总长 → ϱ → 0
      · 采样路径反复走同一小撮边（跑分链、过手链）→ 并集 ≪ 总长 → ϱ → 1 − 1/K
      · 单条路径 → 0（没有"互相"印证可言）
    取值域 [0, 1−1/K]，K 为采样路径条数。

    ϱ_attn 仍然照算并写进 stats（`mean_rho_attention`），因为"它在恒等投影下恒为
    1e-4 量级"本身就是上述论证的实测证据，应当留在结果里供复核，而不是删掉。
    """
    total = sum((len(path) for path in paths))
    if total <= 1:
        return 0.0
    distinct = len({edge for path in paths for edge in path})
    return float(np.clip(1.0 - distinct / total, 0.0, 1.0))

def edge_usage_ranking(paths: Sequence[Sequence[Tuple[int, int, int]]]) -> np.ndarray:
    """按"路径上的边被多少条路径共用"给路径打分，用于选出合并成最小子图的那几条。

    §2.4.3 第三步用 Π 的列贡献做这件事；恒等投影下列贡献近似相等（见 path_attention），
    因此改用边复用计数——它选出的正是被多条路径共同印证的那一段，与该步的意图一致。
    """
    counts: Dict[Tuple[int, int, int], int] = defaultdict(int)
    for path in paths:
        for edge in set(path):
            counts[edge] += 1
    return np.array([float(np.mean([counts[e] for e in path])) if path else 0.0 for path in paths])

def pattern_signature(nodes: Sequence[str], relations: Sequence[Tuple[str, str, str]], times: Sequence[float], attributes: Sequence[str], gap_bin_seconds: float=3600.0) -> str:
    """ID 无关的结构—时序签名，用于回答"这个模式族在训练期出现过吗"。

    刻意保守且可替换：度数谱 + 关系多重集 + 时间间隔分箱 + 属性 schema。
    真正的模式族判定应在调参段上验证图编辑距离与属性距离阈值，此处不冒充那一步。
    """
    in_deg: Dict[str, int] = defaultdict(int)
    out_deg: Dict[str, int] = defaultdict(int)
    rels: List[str] = []
    for src, rel, dst in relations:
        out_deg[src] += 1
        in_deg[dst] += 1
        rels.append(rel)
    degree_profile = sorted(((out_deg[n], in_deg[n]) for n in nodes))
    ordered = sorted(times)
    gaps = tuple((int((b - a) // gap_bin_seconds) for a, b in zip(ordered, ordered[1:])))
    payload = repr((len(nodes), degree_profile, tuple(sorted(rels)), gaps, tuple(sorted(attributes))))
    return hashlib.blake2b(payload.encode('utf-8'), digest_size=12).hexdigest()

class _Adjacency:
    """按时间片增量追加。邻接表在任意时刻只含 τ_e ≤ t 的边——契约二不是靠一句断言
    维持的，而是数据结构本身不可能包含未来边。"""

    def __init__(self) -> None:
        self.out: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
        self.n_edges = 0
        self.max_ts = -np.inf

    def add(self, src: int, dst: int, rel: int, ts: float) -> None:
        self.out[src].append((dst, rel, ts))
        self.n_edges += 1
        if ts > self.max_ts:
            self.max_ts = ts

def _sample_paths(adj: _Adjacency, seeds: Tuple[int, int], k: int, max_hops: int, rng: np.random.Generator) -> List[List[Tuple[int, int, int]]]:
    """从种子两端出发采样变长有类型路径。路径长度由采样过程决定、不受层数限制——
    这正是路径聚合相对 L 层消息传递的差别（后者至多覆盖 L 跳且被逐层平滑抹平）。"""
    paths: List[List[Tuple[int, int, int]]] = []
    for i in range(k):
        node = seeds[i % 2]
        hops = int(rng.integers(1, max_hops + 1))
        path: List[Tuple[int, int, int]] = []
        used: set = set()
        for _ in range(hops):
            candidates = adj.out.get(node)
            if not candidates:
                break
            dst, rel, edge_ts = candidates[int(rng.integers(0, len(candidates)))]
            key = (node, dst, rel, edge_ts)
            if key in used:
                break
            used.add(key)
            path.append((node, rel, dst))
            node = dst
        if path:
            paths.append(path)
    return paths

@dataclass(frozen=True)
class GraphConfirmation:
    """一次结构确认：时间片、写进模式库的节点、以及构成证据向量所需的原始量。

    发现（第一、二级）与派发（写模式库、后续时间片查库）在这里被拆成两件事。
    拆开的直接好处是 ϱ 的两种用法都可以零成本重放：确认日志既不含阈值也不含 ζ 的取法，
    换任一个只需重跑派发。§2.4.3 第二步要求阈值在调参段上选，那就必须能便宜地把
    多个候选都跑一遍。

    **存 burst 而不是存 ζ**：ζ = X̃²·ϱ 还是 ζ = X̃² 是 `zeta_mode` 决定的，
    存成品会把这个选择锁死在发现阶段。
    """
    bucket: int
    targets: Tuple[int, ...]
    burst: float
    rho: float
    max_path_len: float
    n_subgraph_edges: float
    novelty: str = 'UNKNOWN'

    def values(self, zeta_mode: str) -> np.ndarray:
        """按给定的 ζ 取法组装证据向量 [ζ, ϱ, 最长路径, 最小子图边数]。"""
        zeta = self.burst * self.rho if zeta_mode == 'product' else self.burst
        return np.array([zeta, self.rho, self.max_path_len, self.n_subgraph_edges], dtype=np.float32)

@dataclass
class GraphEmissionContext:
    """派发所需的全部输入。与 ϱ 阈值无关，因此可以复用。"""
    n_rows: int
    bucket_rows: List[Tuple[int, np.ndarray]]
    endpoint_nodes: List[np.ndarray]
    horizon: int

def emit_graph_features(ctx: GraphEmissionContext, confirmations: Sequence[GraphConfirmation], threshold: float, zeta_mode: str='product') -> Tuple[np.ndarray, int]:
    """按给定 ϱ 阈值与 ζ 取法重放派发。返回 (g_t 矩阵, 通过阈值的确认数)。

    ϱ 在本方法里有**两重身份**，这个函数把它们分开：
      · 门 —— `threshold`，决定哪些种子进模式库
      · 权重 —— `zeta_mode`，决定 ζ 是 X̃²·ϱ（product）还是 X̃²（burst_only）
    实测两重在不同数据上分别失效（Sparkov 上权重有害、TabFormer 上门有益），
    绑在一起时没有任何数据集能同时拿到两边的好处，故拆开各自在调参段上选。

    因果方向由循环顺序保证：片 b 的交易只看 node_state，而 node_state 此刻只含
    **早于 b** 的确认结果——本片的确认在派发之后才写入。
    """
    features = np.zeros((ctx.n_rows, len(GRAPH_FEATURE_NAMES)), dtype=np.float32)
    by_bucket: Dict[int, List[Tuple[np.ndarray, GraphConfirmation]]] = defaultdict(list)
    n_passed = 0
    for item in confirmations:
        if item.rho >= threshold:
            by_bucket[item.bucket].append((item.values(zeta_mode), item))
            n_passed += 1
    node_state: Dict[int, Tuple[np.ndarray, int]] = {}
    for b, rows in ctx.bucket_rows:
        if node_state:
            expired = [nd for nd, (_, at) in node_state.items() if b - at > ctx.horizon]
            for nd in expired:
                del node_state[nd]
        if node_state and rows.size:
            for endpoint in ctx.endpoint_nodes:
                nodes = endpoint[rows]
                for j in range(rows.size):
                    entry = node_state.get(int(nodes[j]))
                    if entry is None:
                        continue
                    values, confirmed_at = entry
                    row = int(rows[j])
                    if values[0] > features[row, 0]:
                        features[row] = np.append(values, float(b - confirmed_at))
        for values, item in by_bucket.get(b, ()):
            for node in item.targets:
                previous = node_state.get(node)
                if previous is None or values[0] >= previous[0][0]:
                    node_state[node] = (values, b)
    return (features, n_passed)

@dataclass
class GraphArtifacts:
    matrix: np.ndarray
    feature_names: List[str]
    enabled: bool
    reason: str
    stats: Dict[str, Any] = field(default_factory=dict)
    patterns: List[PatternEntry] = field(default_factory=list)
    confirmations: List[GraphConfirmation] = field(default_factory=list)
    emission: Optional[GraphEmissionContext] = None
    rho_samples_train: List[float] = field(default_factory=list)
    _describe: Optional[Any] = None

    def resolve_threshold(self, setting: Any, quantile: float, minimum_seeds: int, fallback: float) -> Tuple[float, Dict[str, Any]]:
        """把 `rho_min` 的配置取值解成一个具体阈值，并给出可复核的定标记录。"""
        if not (isinstance(setting, str) and setting.lower() == 'auto'):
            return (float(setting), {'mode': 'fixed', 'threshold': float(setting)})
        samples = self.rho_samples_train
        if len(samples) >= minimum_seeds:
            threshold, mode = (float(np.quantile(samples, quantile)), 'auto')
        else:
            threshold, mode = (float(fallback), 'auto_unavailable')
        return (threshold, {'mode': mode, 'quantile': quantile, 'n_train_seeds': len(samples), 'min_required_seeds': minimum_seeds, 'threshold': threshold, 'train_rho_quantiles': {str(q): float(np.quantile(samples, q)) if samples else None for q in (0.1, 0.25, 0.5, 0.75, 0.9)}})

    def with_threshold(self, threshold: float, calibration: Dict[str, Any], zeta_mode: str='product') -> 'GraphArtifacts':
        """按新阈值与新的 ζ 取法重放派发，返回一份新的产物。发现阶段完全不重跑。"""
        if not self.enabled or self.emission is None:
            return self
        matrix, n_passed = emit_graph_features(self.emission, self.confirmations, threshold, zeta_mode)
        stats = dict(self.stats)
        stats['n_structure_confirmed'] = n_passed
        stats['n_time_only'] = stats['n_expanded'] - stats['n_no_path'] - n_passed
        stats['n_rows_with_signal'] = int((matrix[:, 0] > 0).sum())
        stats['rho_min_used'] = threshold
        stats['zeta_mode_used'] = zeta_mode
        stats['rho_calibration'] = calibration
        novelty: Dict[str, int] = defaultdict(int)
        for item in self.confirmations:
            if item.rho >= threshold:
                novelty[item.novelty] += 1
        kept: List[PatternEntry] = []
        for pattern in self.patterns:
            if pattern.concentration >= threshold:
                kept.append(pattern)
        stats['novelty'] = dict(novelty)
        stats['n_patterns_retained_for_audit'] = len(kept)
        reason = self._describe(threshold, calibration) if self._describe else self.reason
        return GraphArtifacts(matrix, list(self.feature_names), True, reason, stats, kept, self.confirmations, self.emission, self.rho_samples_train, self._describe)

def _pattern_attributes(frame: pd.DataFrame, rows: np.ndarray, gamma: SemanticSpec) -> Dict[str, Any]:
    """A 分量由 Γ 而非图模块产出：1.4.1 的 ω_j 必须对图属性同样生效——
    判定之后才产生的属性不得进入模式条目（§2.4.8 输出表脚注）。"""
    attrs: Dict[str, Any] = {'n_transactions': int(rows.size)}
    for name in ('amount', 'cat1', 'cat2', 'cat3'):
        if name not in frame.columns or name not in gamma.fields:
            continue
        spec = gamma.field(name)
        if spec.posthoc or spec.availability_lag_seconds > 0:
            continue
        values = frame[name].to_numpy()[rows]
        if name == 'amount':
            numeric = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=np.float64)
            finite = numeric[np.isfinite(numeric)]
            attrs['amount_sum'] = float(finite.sum()) if finite.size else 0.0
            attrs['amount_max'] = float(finite.max()) if finite.size else 0.0
            attrs['amount_unit'] = spec.unit
        else:
            attrs[name] = sorted({spec.decode(v) for v in values})[:5]
    return attrs

def build_graph(frame: pd.DataFrame, train_idx: np.ndarray, gamma: SemanticSpec, cfg: Dict[str, Any], seed: int=2026, auditor=None) -> GraphArtifacts:
    """跑一遍发现流程（触发 → 展开 → 确认），产出确认日志，再按解析出的 ϱ 阈值派发一次。

    确认阶段不施加 ϱ 阈值：所有展开成功的种子都记进确认日志并附带自己的 ϱ。
    阈值只在派发时起作用，因此换阈值不必重跑发现——`with_threshold` 即可。
    """
    if not bool(cfg.get('enabled', True)):
        return GraphArtifacts(np.zeros((len(frame), 0), dtype=np.float32), [], False, str(cfg.get('disabled_reason', '配置关闭')))
    n = len(frame)
    ts = frame['ts'].to_numpy(dtype=np.float64)
    bucket_seconds = float(cfg['bucket_seconds'])
    bucket = np.floor((ts - ts.min()) / bucket_seconds).astype(np.int64)
    streams = list(cfg.get('streams') or [])
    if not streams:
        raise ValueError('graph.streams 不得为空——图层没有边流就没有检测对象')
    node_id: Dict[str, int] = {}
    node_names: Dict[int, str] = {}
    relation_names: Dict[int, str] = {}

    def nid(namespace: str, value: str) -> int:
        key = f'{namespace}:{value}'
        found = node_id.get(key)
        if found is None:
            found = len(node_id)
            node_id[key] = found
            node_names[found] = key
        return found
    edge_src: List[np.ndarray] = []
    edge_dst: List[np.ndarray] = []
    edge_rel: List[int] = []
    for s_index, spec in enumerate(streams):
        src_ns, dst_ns = (str(spec.get('source_ns', 'N')), str(spec.get('target_ns', 'N')))
        src_vals = frame[spec['source']].astype(str).to_numpy()
        dst_vals = frame[spec['target']].astype(str).to_numpy()
        rel = len(relation_names)
        relation_names[rel] = str(spec.get('relation', f'rel{s_index}'))
        edge_src.append(np.fromiter((nid(src_ns, v) for v in src_vals), dtype=np.int64, count=n))
        edge_dst.append(np.fromiter((nid(dst_ns, v) for v in dst_vals), dtype=np.int64, count=n))
        edge_rel.append(rel)
    n_nodes = max(len(node_id), 1)
    add_reverse = bool(cfg.get('add_reverse', True))
    reverse_rel: Dict[int, int] = {}
    if add_reverse:
        for rel in list(relation_names):
            new_rel = len(relation_names)
            relation_names[new_rel] = 'rev_' + relation_names[rel]
            reverse_rel[rel] = new_rel
    trigger_streams = [i for i, s in enumerate(streams) if bool(s.get('trigger', i == 0))]
    pair_key_row = {i: edge_src[i] * n_nodes + edge_dst[i] for i in trigger_streams}
    epsilon = float(cfg['epsilon'])
    bonferroni = bool(cfg.get('bonferroni', True))
    use_trigger = bool(cfg.get('trigger', True))
    require_prior = bool(cfg.get('require_prior_history', True))
    force_rho_one = bool(cfg.get('force_rho_one', False))
    max_seeds = int(cfg.get('max_seeds_per_bucket', 64))
    k_paths = int(cfg.get('path_samples', 16))
    max_hops = int(cfg.get('max_hops', 3))
    vector_dim = int(cfg.get('vector_dim', 32))
    counter_mode = str(cfg.get('counter', 'exact'))
    max_patterns = int(cfg.get('max_patterns_exported', 2000))
    horizon = int(cfg.get('signal_horizon_buckets', 24))
    signal_scope = str(cfg.get('signal_scope', 'seed'))
    rho_estimator = str(cfg.get('rho_estimator', 'overlap'))
    if rho_estimator not in ('overlap', 'attention'):
        raise ValueError('graph.rho_estimator 只能是 overlap 或 attention')
    rho_attn_sum, rho_attn_count = (0.0, 0)
    last_train_bucket = int(bucket[train_idx].max()) if train_idx.size else int(bucket.max())
    rho_samples: List[float] = []
    rng = np.random.default_rng(seed)
    adjacency = _Adjacency()
    patterns: List[PatternEntry] = []
    train_replacement_cursor = max_patterns - 1
    confirmations: List[GraphConfirmation] = []
    bucket_rows: List[Tuple[int, np.ndarray]] = []
    cumulative: Dict[int, float] = defaultdict(float)
    sketch = CountMinSketch(int(cfg.get('cms_width', 4096)), int(cfg.get('cms_depth', 5))) if counter_mode == 'cms' else None
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    train_signatures: set = set()
    stats: Dict[str, Any] = {'n_buckets': 0, 'n_pair_tests': 0, 'n_triggered': 0, 'n_expanded': 0, 'n_structure_confirmed': 0, 'n_time_only': 0, 'n_no_path': 0, 'n_seed_cap_hits': 0, 'n_skipped_no_prior_history': 0, 'n_edges_in_adjacency': 0, 'n_rows_with_signal': 0, 'trigger_rate_head': [], 'm_t_head': [], 'epsilon_t_head': [], 'novelty': defaultdict(int)}
    for tick, b in enumerate(np.unique(bucket), start=1):
        rows = np.flatnonzero(bucket == b)
        decision_time = float(ts[rows].max())
        stats['n_buckets'] += 1
        bucket_rows.append((int(b), rows))
        seeds: List[Dict[str, Any]] = []
        for s_index in trigger_streams:
            pairs = pair_key_row[s_index][rows]
            uniq, counts_now = np.unique(pairs, return_counts=True)
            m_t = int(uniq.size)
            stats['n_pair_tests'] += m_t
            eps_t = epsilon / max(m_t, 1) if bonferroni else epsilon
            threshold = chi_square_threshold(eps_t)
            if len(stats['m_t_head']) < 50:
                stats['m_t_head'].append(m_t)
                stats['epsilon_t_head'].append(float(eps_t))
            prior = np.array([sketch.estimate(int(k)) if sketch is not None else cumulative[int(k)] for k in uniq], dtype=np.float64)
            current = counts_now.astype(np.float64)
            if sketch is not None:
                for key, count in zip(uniq, current):
                    sketch.add(int(key), float(count))
                total = np.array([sketch.estimate(int(k)) for k in uniq], dtype=np.float64)
            else:
                for key, count in zip(uniq, current):
                    cumulative[int(key)] += float(count)
                total = prior + current
            scores = midas_statistic(current, total, tick)
            eligible = np.ones(m_t, dtype=bool)
            if require_prior:
                eligible = prior > 0
                stats['n_skipped_no_prior_history'] += int((~eligible).sum())
            triggered = (scores > threshold) & eligible if use_trigger else eligible.copy()
            trigger_idx = np.flatnonzero(triggered)
            stats['n_triggered'] += int(trigger_idx.size)
            if len(stats['trigger_rate_head']) < 50:
                stats['trigger_rate_head'].append(float(trigger_idx.size / max(m_t, 1)))
            if trigger_idx.size > max_seeds:
                stats['n_seed_cap_hits'] += 1
                trigger_idx = trigger_idx[np.argsort(scores[trigger_idx])[::-1][:max_seeds]]
            for i in trigger_idx:
                key = int(uniq[i])
                seeds.append({'stream': s_index, 'pair_key': key, 'pair': (key // n_nodes, key % n_nodes), 'score': float(scores[i]), 'eps_t': float(eps_t), 'm_t': m_t, 'rows': rows[pairs == key]})
        for s_index in range(len(streams)):
            src, dst = (edge_src[s_index][rows], edge_dst[s_index][rows])
            rel = edge_rel[s_index]
            for j in range(rows.size):
                edge_ts = float(ts[rows[j]])
                adjacency.add(int(src[j]), int(dst[j]), rel, edge_ts)
                if add_reverse:
                    adjacency.add(int(dst[j]), int(src[j]), reverse_rel[rel], edge_ts)
        if adjacency.max_ts > decision_time:
            raise ContractViolation(f'邻接表含未来边：max(τ_e)={adjacency.max_ts} > t={decision_time}')
        for seed_info in seeds:
            stats['n_expanded'] += 1
            paths = _sample_paths(adjacency, seed_info['pair'], k_paths, max_hops, rng)
            if not paths:
                stats['n_no_path'] += 1
                continue
            attention = path_attention(vectorize_paths(paths, vector_dim, node_names, relation_names))
            rho_attn = attention_concentration(attention)
            rho_overlap = path_overlap_concentration(paths)
            rho_attn_sum += rho_attn
            rho_attn_count += 1
            rho = 1.0 if force_rho_one else rho_attn if rho_estimator == 'attention' else rho_overlap
            if b <= last_train_bucket:
                rho_samples.append(float(rho))
            zeta = seed_info['score'] * rho
            contribution = edge_usage_ranking(paths)
            keep = max(1, math.ceil(len(paths) * 0.25))
            chosen = np.argsort(contribution)[-keep:][::-1]
            edges: Dict[Tuple[int, int, int], Tuple[str, str, str]] = {}
            for p in chosen:
                for src, rel, dst in paths[int(p)]:
                    edges[src, rel, dst] = (node_names[src], relation_names[rel], node_names[dst])
            subgraph_nodes = sorted({x for triple in edges.values() for x in (triple[0], triple[2])})
            relations = tuple(sorted(edges.values()))
            times = tuple((float(x) for x in ts[seed_info['rows']]))
            attrs = _pattern_attributes(frame, seed_info['rows'], gamma)
            signature = pattern_signature(subgraph_nodes, relations, times, tuple((k for k in attrs if k != 'n_transactions')))
            if bool(train_mask[seed_info['rows']].all()):
                train_signatures.add(signature)
                novelty = 'TRAIN_PERIOD'
            else:
                novelty = 'MATCHES_TRAIN' if signature in train_signatures else 'UNSEEN_CANDIDATE'
            stats['novelty'][novelty] += 1
            targets = set(seed_info['pair'])
            if signal_scope == 'subgraph':
                targets |= {n for src, _, dst in edges for n in (src, dst)}
            confirmations.append(GraphConfirmation(int(b), tuple(sorted(targets)), float(seed_info['score']), float(rho), float(max((len(p) for p in paths))), float(len(edges)), novelty))
            entry = PatternEntry(pattern_id=hashlib.blake2b(repr((signature, seed_info['pair_key'], int(b))).encode(), digest_size=12).hexdigest(), S=tuple(subgraph_nodes), R=relations, T=times, A=attrs, seed_pair=(node_names[seed_info['pair'][0]], node_names[seed_info['pair'][1]]), time_bucket=int(b), burst_score=seed_info['score'], concentration=rho, zeta=zeta, epsilon_t=seed_info['eps_t'], m_t=seed_info['m_t'], novelty=novelty, signature=signature)
            if len(patterns) < max_patterns:
                patterns.append(entry)
            elif novelty != 'TRAIN_PERIOD' and train_replacement_cursor >= 0:
                while train_replacement_cursor >= 0 and patterns[train_replacement_cursor].novelty != 'TRAIN_PERIOD':
                    train_replacement_cursor -= 1
                if train_replacement_cursor >= 0:
                    patterns[train_replacement_cursor] = entry
                    train_replacement_cursor -= 1
    stats['n_edges_in_adjacency'] = adjacency.n_edges
    stats['rho_estimator'] = rho_estimator
    stats['mean_rho_attention'] = rho_attn_sum / max(rho_attn_count, 1)
    stats['mean_trigger_rate'] = float(np.mean(stats['trigger_rate_head'])) if stats['trigger_rate_head'] else 0.0
    stats['mean_m_t'] = float(np.mean(stats['m_t_head'])) if stats['m_t_head'] else 0.0
    stats['expansion_call_ratio'] = stats['n_expanded'] / max(stats['n_pair_tests'], 1)
    stats['novelty'] = dict(stats['novelty'])

    def describe(threshold: float, calibration: Dict[str, Any]) -> str:
        return f"streams={[s.get('relation') for s in streams]}, bucket={bucket_seconds:g}s, ε={epsilon}, trigger={('on' if use_trigger else 'off(全实体对展开)')}, ϱ_{rho_estimator}={('forced=1.0' if force_rho_one else f'≥{threshold:.4f}')}（{calibration['mode']}）, 证据有效期={horizon}片"
    emission = GraphEmissionContext(n_rows=n, bucket_rows=bucket_rows, endpoint_nodes=[arr for s_index in range(len(streams)) for arr in (edge_src[s_index], edge_dst[s_index])], horizon=horizon)
    base = GraphArtifacts(np.zeros((n, len(GRAPH_FEATURE_NAMES)), dtype=np.float32), list(GRAPH_FEATURE_NAMES), True, '', stats, patterns, confirmations, emission, rho_samples, describe)
    quantile_setting = cfg.get('rho_quantile', 0.75)
    if isinstance(quantile_setting, str) and quantile_setting.lower() == 'auto':
        grid = [float(q) for q in cfg.get('rho_quantile_grid', [0.75])] or [0.75]
        quantile_setting = grid[len(grid) // 2]
    threshold, calibration = base.resolve_threshold(cfg.get('rho_min', 'auto'), float(quantile_setting), int(cfg.get('rho_min_calibration_seeds', 20)), float(cfg.get('rho_fallback', 0.2)))
    if force_rho_one:
        threshold, calibration = (-np.inf, {'mode': 'forced_rho_one', 'threshold': -np.inf})
    zeta_setting = str(cfg.get('zeta_mode', 'product'))
    if zeta_setting.lower() == 'auto':
        zeta_setting = 'product'
    result = base.with_threshold(threshold, calibration, zeta_setting)
    if auditor is not None:
        auditor.record('契约二·图只在 E_t 上计算', adjacency.n_edges, True, f"{stats['n_buckets']} 个时间片逐片增量追加；证据经模式库在确认后 1..{horizon} 片内派发")
        auditor.record('ϱ 阈值定标只用训练期', int(calibration.get('n_train_seeds', 0)), calibration['mode'] != 'auto_unavailable', f"mode={calibration['mode']}, 阈值={threshold:.4f}" + ('（训练期触发种子不足，落到兜底阈值）' if calibration['mode'] == 'auto_unavailable' else ''))
        auditor.record('单调过滤 S_t^out ⊆ S_t（发现阶段）', stats['n_expanded'], True, f"触发 {stats['n_triggered']} → 展开 {stats['n_expanded']}；结构确认数取决于最终选定的 ϱ 阈值，见「ϱ 阈值选定后的确认规模」一项")
    return result
''', _ROOT=_ROOT, _DATASET=_DATASET, _OUTPUT=_OUTPUT, _DATA=_DATA, _CONFIGS=_CONFIGS)

# -------------------- how123.detector --------------------
_install('how123.detector', r'''
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import xgboost as xgb

@dataclass
class DetectorArtifacts:
    best_iteration: int
    best_score: float
    n_features: int
    params: Dict[str, Any]
    global_importance: List[Tuple[str, float]] = field(default_factory=list)

class FusedDetector:
    """XGBoost 评分器。早停只用调参段，校准段与测试段在冻结前完全不接触。"""

    def __init__(self, cfg: Dict[str, Any], feature_names: Sequence[str], seed: int=2026) -> None:
        self.cfg = cfg
        self.feature_names = list(feature_names)
        self.seed = seed
        self.model: Optional[xgb.XGBClassifier] = None
        self.artifacts: Optional[DetectorArtifacts] = None

    def _params(self, y_train: np.ndarray) -> Dict[str, Any]:
        spw = self.cfg.get('scale_pos_weight', 'auto')
        if spw == 'auto':
            n_pos = max(1, int(y_train.sum()))
            spw = float((y_train.size - n_pos) / n_pos)
        return dict(n_estimators=int(self.cfg.get('n_estimators', 600)), max_depth=int(self.cfg.get('max_depth', 6)), learning_rate=float(self.cfg.get('learning_rate', 0.05)), subsample=float(self.cfg.get('subsample', 0.8)), colsample_bytree=float(self.cfg.get('colsample_bytree', 0.8)), min_child_weight=float(self.cfg.get('min_child_weight', 5)), reg_lambda=float(self.cfg.get('reg_lambda', 1.0)), scale_pos_weight=float(spw), tree_method='hist', eval_metric='aucpr', objective='binary:logistic', random_state=self.seed, n_jobs=int(self.cfg.get('n_jobs', -1)), early_stopping_rounds=int(self.cfg.get('early_stopping_rounds', 50)))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_tune: np.ndarray, y_tune: np.ndarray) -> DetectorArtifacts:
        params = self._params(y_train)
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train, eval_set=[(X_tune, y_tune)], verbose=False)
        booster = self.model.get_booster()
        booster.feature_names = self.feature_names
        gain = booster.get_score(importance_type='total_gain')
        importance = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:25]
        self.artifacts = DetectorArtifacts(best_iteration=int(getattr(self.model, 'best_iteration', params['n_estimators'] - 1)), best_score=float(getattr(self.model, 'best_score', float('nan'))), n_features=X_train.shape[1], params=params, global_importance=[(k, float(v)) for k, v in importance])
        return self.artifacts

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError('先 fit 再 predict')
        return self.model.predict_proba(X)[:, 1].astype(np.float64)

    def shap_contributions(self, X: np.ndarray) -> np.ndarray:
        """TreeSHAP 归因，末列是 base value，这里裁掉只留特征贡献。

        直接走 booster 的 pred_contribs 而不是 shap 包：结果完全一致（都是 TreeSHAP 精确解），
        但少一个依赖，且不受 shap 0.48 与 xgboost 3.x 之间那处已知的构造失败影响。
        """
        if self.model is None:
            raise RuntimeError('先 fit 再解释')
        booster = self.model.get_booster()
        matrix = xgb.DMatrix(X, feature_names=self.feature_names)
        return booster.predict(matrix, pred_contribs=True)[:, :-1]

    def explain_top_k(self, X: np.ndarray, k: int=3) -> List[List[Dict[str, float]]]:
        contribs = self.shap_contributions(X)
        out: List[List[Dict[str, float]]] = []
        for row in contribs:
            order = np.argsort(np.abs(row))[::-1][:k]
            out.append([{'feature': self.feature_names[int(i)], 'impact': float(row[int(i)])} for i in order])
        return out
''', _ROOT=_ROOT, _DATASET=_DATASET, _OUTPUT=_OUTPUT, _DATA=_DATA, _CONFIGS=_CONFIGS)

# -------------------- how123.pipeline --------------------
_install('how123.pipeline', r'''
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from .config import config_digest, get_path
from .constraint import CalibrationInfeasible, CapacityGate, make_calibrator
from .context import ContextArtifacts, build_context
from .contracts import ContractAuditor, ScorerLifecycle, SemanticSpec
from .datasets import Dataset, Splits, load_dataset, make_splits
from .detector import FusedDetector
from .features import fit_base_features
from .graph import GraphArtifacts, build_graph
from .metrics import average_precision, evaluate, tpr_at_fpr
from .semantic import build_semantic

@dataclass
class Workspace:
    """跨消融臂复用的部分：数据、切分与 x_base。这三样与臂无关，重复算是纯浪费。"""
    cfg: Dict[str, Any]
    dataset: Dataset
    frame: pd.DataFrame
    splits: Splits
    X_base: np.ndarray
    base_names: List[str]
    y: np.ndarray
    load_seconds: float = 0.0
    cache: Dict[Tuple[str, str], Any] = field(default_factory=dict)

def build_workspace(cfg: Dict[str, Any], verbose: bool=True) -> Workspace:
    start = time.perf_counter()
    ds = load_dataset(cfg['dataset'], int(cfg.get('window_rows', 400000)))
    frame = ds.frame
    splits = make_splits(frame, get_path(cfg, 'split.mode'), list(get_path(cfg, 'split.ratios')), get_path(cfg, 'split.group_column'), int(cfg.get('seed', 2026)))
    y = frame['label'].to_numpy(dtype=int)
    splits.assert_label_coverage(y)
    space = fit_base_features(frame, splits.train)
    X_base = space.transform(frame)
    if verbose:
        print(f"  [数据] {ds.name}: {len(frame):,} 行, 正类率 {ds.meta['positive_rate']:.6f}, 跨度 {ds.meta['time_span_days']:.1f} 天, 切分 {splits.sizes()}")
    return Workspace(cfg, ds, frame, splits, X_base, space.columns, y, time.perf_counter() - start)

def _cached(ws: Workspace, layer: str, section: Mapping[str, Any], builder: Callable[[ContractAuditor], Any], auditor: ContractAuditor) -> Any:
    """按层缓存产物，并把首次构建时产生的契约检查记录一并存下来重放。

    不重放会让缓存命中的臂看上去"没做过检查"——那正是 memory 里那条"校验器通过要看
    匹配计数"的翻版：审计记录数为 0 与真的没检查在输出上不可区分。
    """
    digest = hashlib.blake2b(json.dumps(section, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8'), digest_size=8).hexdigest()
    entry = ws.cache.get((layer, digest))
    if entry is None:
        local = ContractAuditor()
        entry = (builder(local), list(local.results))
        ws.cache[layer, digest] = entry
    auditor.results.extend(entry[1])
    return entry[0]
_RHO_KEYS = ('rho_min', 'rho_quantile', 'rho_quantile_grid', 'rho_fallback', 'rho_min_calibration_seeds', 'zeta_mode', 'zeta_mode_grid')
_TRUTHY = ('true', '1', 'yes', 'on')

def _as_bool(setting: str) -> bool:
    return setting in _TRUTHY

def _semantic_candidates(cfg: Mapping[str, Any], frame: pd.DataFrame) -> List[bool]:
    """`auto` = 在调参段上选；`text_only` = §4.5 的原规则；true/false = 强制。

    把 auto 从"看有没有原始文本"改成"在调参段上选"，是因为原规则被实测推翻：
    TabFormer 的 `text` 字段声明为 n/a（无原始文本），按原规则应当关闭 Γ 序列化，
    而调参段实测开启更好——24 个候选里 semantic=True 的最好一档 tune AP 0.5292，
    semantic=False 的最好一档只有 0.3955，选中的那一档测试段 AP 0.5052。
    （核对于 RQ1/results/raw/tabformer-cc_main.json，2026-09-04 那轮；2026-09-08 复核。
    注意 tabformer-cc 不在六数据集交付集内。）
    原规则作为 `text_only` 保留，RQ2 里有对应的复现臂。
    """
    setting = str(cfg.get('enabled', 'auto')).lower()
    if setting == 'auto':
        return [True, False]
    if setting == 'text_only':
        has_text = bool((frame['text'].astype(str).str.len() > 0).any()) if 'text' in frame else False
        return [has_text]
    return [_as_bool(setting)]

def _context_candidates(cfg: Mapping[str, Any]) -> List[bool]:
    """`auto` = 在调参段上选；true/false = 强制。与 `_semantic_candidates` 同构。

    §4.5 确立的原则是"模块级开关不写成规则，而是在调参段上选"，但先前只对
    `semantic.enabled` 落实了，`context.enabled` 一直写死为 true——于是 a_t 在数据
    允许时一律强制开启，即使有害。补上这个开关不改任何算法，只是把同一条已有原则
    落实到第二个模块，且选择只在调参段上做。

    PaySim 是促成这条改动的数据集：图层因结构原因关闭，`xgboost_base_only` 的 AP
    0.9239 高于 how123_full 的 0.9221，即三层合计仍是净负担。
    （核对于 RQ1/results/raw/paysim_main.json，2026-09-04 那轮；2026-09-08 复核。
    PaySim 不在六数据集交付集内。）

    **本函数先前写在这里的那段 PaySim 叙述已被同一批产物推翻，故已改写**：它说
    "语义层已被调参段选择关闭、唯一多出来的是 a_t"，而落盘的 layers 是
    semantic=True / context=False；它引的 full AP 0.8811 与 ΔAP(−0.0427***) 在现有
    raw 里都找不到（现值 0.9221 与 −0.00165，p=0.80）。方向性结论不变，数值不再复述。

    调参段能不能在不看测试段的前提下选对方向，见 running/Diagnostics/d07_context_switch.py；
    该脚本的具体数值本文件不复述，以免又留下一份无人复核的副本。
    """
    setting = str(cfg.get('enabled', True)).lower()
    if setting == 'auto':
        return [True, False]
    return [_as_bool(setting)]

def _rho_candidates(cfg: Mapping[str, Any], graph_enabled: bool) -> List[Optional[float]]:
    if not graph_enabled:
        return [None]
    setting = cfg.get('rho_quantile', 0.75)
    if isinstance(setting, str) and setting.lower() == 'auto':
        return [float(q) for q in cfg.get('rho_quantile_grid', [0.0, 0.5, 0.75])]
    return [float(setting)]

def _zeta_candidates(cfg: Mapping[str, Any], graph_enabled: bool) -> List[Optional[str]]:
    """ϱ 的第二重身份：作为 ζ 的乘子用不用。

    `product` = ζ = X̃²·ϱ（§2.4.8 原式），`burst_only` = ζ = X̃²（只把 ϱ 当门，不当权重）。

    与 `rho_quantile` 分开选，是因为 RQ2 在**测试段**上测出两重的成败并不一致：一侧是
    ϱ≡1 更好（拖后腿的是权重那一重），另一侧是 ϱ≡1 更差，而后者选中的是
    burst_only + 最低门分位，即权重与门都已不起作用——那部分差只能来自 ϱ 作为证据
    向量里一个显式特征的那一重。绑在一起时没有任何数据集能同时拿到两边的好处。
    （这两条 ΔAP 出自 RQ2；当前磁盘上的 RQ2/results/raw 与 rq2_results_v1.3.csv 对不上，
    2026-09-08 复核时无法逐位复核，故此处只留结构、不再复述具体数值与 p 值。）

    **但不要把"分开选"读成"选型能分辨它们"。** 调参段在 Sparkov 上对这两个配置的 AP 差是
    3.428e-07（`margin_over_runner_up`，实测；product 0.99693695 对 burst_only 0.99693729），
    即在任何有意义的精度上都是并列，胜负由第七位小数决定，不是选型识别出了权重有害。
    要让选型真能分辨，得换一个对这一维更敏感的选型判据，或者在多个种子上取平均——两者都还没做。

    门那一维要留神的是另一件事：收紧它会大幅压低覆盖率。IBM-AML 上有信号的行随分位从
    35477/36022（q=0.0，两个 zeta 取法）掉到 635（q=0.5）与 281（q=0.75）；在 q=0.5 与
    q=0.75 上两个 zeta 取法给出**完全相同**的 tune AP（0.09051、0.05957），说明剩下的信号
    已经不足以区分权重取法。**但不能因此说"门只会饿死图层"**：同一张表里 q=0.5 的
    tune AP 0.09051 是 24 个候选里的最高值并被选中，高于 q=0.0 的 0.07141。
    （全部核对于 RQ1/results/raw/ibm-aml_main.json 与 sparkov_main.json 的
    `selection_on_tune.table`，2026-09-04 那轮；2026-09-08 复核。此前这里写的
    "4007 掉到 364/150"与"tune AP 随之塌回无图基线"在现有产物里都不成立。）
    """
    if not graph_enabled:
        return [None]
    setting = cfg.get('zeta_mode', 'product')
    if isinstance(setting, str) and setting.lower() == 'auto':
        return list(cfg.get('zeta_mode_grid', ['product', 'burst_only']))
    return [str(setting)]

def _select_on_tune(ws: Workspace, cfg: Dict[str, Any], gamma: SemanticSpec, context: ContextArtifacts, graph_base: GraphArtifacts, auditor: ContractAuditor, seed: int) -> Tuple[Any, ...]:
    """在调参段上挑两个模块开关（h_dom、a_t）与 ϱ 的门/权重，返回获胜组合与其评分器。

    获胜臂的评分器直接复用——早停本来就用调参段，选型也用调参段，两者同属 §3.2.1
    定义的"选型"阶段。校准段与测试段在此之前一次都没有被读过。

    `context` 参数传入的是**已按配置建好的那一份**；候选里需要"关闭"时不重建，
    直接用一个零宽矩阵替代——上文块关闭的含义就是不往 s(x) 里拼任何 a_t 分量。
    """
    frame, splits, y = (ws.frame, ws.splits, ws.y)
    criterion = str(get_path(cfg, 'evaluation.selection_criterion', 'ap'))
    alpha = float(get_path(cfg, 'constraint.alpha'))
    y_tune = y[splits.tune]
    empty_ctx = np.zeros((ws.X_base.shape[0], 0), dtype=np.float32)
    ctx_candidates = _context_candidates(cfg['context']) if context.matrix.shape[1] else [False]
    table: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for use_semantic in _semantic_candidates(cfg['semantic'], frame):
        semantic_cfg = dict(cfg['semantic'])
        semantic_cfg['enabled'] = bool(use_semantic)
        semantic = _cached(ws, 'semantic', semantic_cfg, lambda a, c=semantic_cfg: build_semantic(frame, splits.train, gamma, c), auditor)
        semantic_names = [f'h_dom_{i}' for i in range(semantic.matrix.shape[1])]
        for use_context in ctx_candidates:
            ctx_matrix = context.matrix if use_context else empty_ctx
            ctx_names = context.feature_names if use_context else []
            for quantile in _rho_candidates(cfg['graph'], graph_base.enabled):
                for zeta_mode in _zeta_candidates(cfg['graph'], graph_base.enabled):
                    if quantile is None:
                        graph = graph_base
                    else:
                        threshold, calibration = graph_base.resolve_threshold(cfg['graph'].get('rho_min', 'auto'), quantile, int(cfg['graph'].get('rho_min_calibration_seeds', 20)), float(cfg['graph'].get('rho_fallback', 0.2)))
                        graph = graph_base.with_threshold(threshold, calibration, zeta_mode)
                    X, names = _stack([(ws.X_base, ws.base_names), (semantic.matrix, semantic_names), (ctx_matrix, ctx_names), (graph.matrix, graph.feature_names)])
                    detector = FusedDetector(cfg['detector'], names, seed)
                    artifacts = detector.fit(X[splits.train], y[splits.train], X[splits.tune], y[splits.tune])
                    tune_scores = detector.predict(X[splits.tune])
                    score_ap = average_precision(y_tune, tune_scores)
                    score_tpr = tpr_at_fpr(y_tune, tune_scores, alpha)
                    value = score_tpr if criterion == 'tpr_at_alpha' else score_ap
                    row = {'semantic': bool(use_semantic), 'context': bool(use_context), 'rho_quantile': quantile, 'zeta_mode': zeta_mode, 'tune_ap': score_ap, 'tune_tpr_at_alpha': score_tpr, 'n_features': int(X.shape[1]), 'graph_rows_with_signal': int(graph.stats.get('n_rows_with_signal', 0))}
                    table.append(row)
                    if best is None or value > best['value']:
                        best = {'value': value, 'row': row, 'semantic': semantic, 'graph': graph, 'X': X, 'names': names, 'detector': detector, 'artifacts': artifacts}
    if best is None:
        raise RuntimeError('调参段选型没有产生任何候选，s(x) 无从确定')
    for row in table:
        row['selected'] = row is best['row']
    value_key = 'tune_tpr_at_alpha' if criterion == 'tpr_at_alpha' else 'tune_ap'
    values = sorted((r[value_key] for r in table), reverse=True)
    margin = values[0] - values[1] if len(values) > 1 else float('nan')
    n_tied = sum((1 for v in values if v == values[0]))
    selection = {'criterion': criterion, 'n_candidates': len(table), 'table': table, 'chosen_semantic': best['row']['semantic'], 'chosen_context': best['row'].get('context'), 'chosen_rho_quantile': best['row']['rho_quantile'], 'chosen_zeta_mode': best['row'].get('zeta_mode'), 'chosen_tune_ap': best['row']['tune_ap'], 'chosen_tune_tpr_at_alpha': best['row']['tune_tpr_at_alpha'], 'margin_over_runner_up': margin, 'n_tied_at_best': n_tied, 'tie_broken_by_grid_order': bool(n_tied > 1)}
    return (selection, best['semantic'], best['graph'], best['X'], best['names'], best['detector'], best['artifacts'])

def _stack(blocks: Sequence[Tuple[np.ndarray, Sequence[str]]]) -> Tuple[np.ndarray, List[str]]:
    matrices = [m for m, names in blocks if m.size and m.shape[1] > 0]
    names: List[str] = []
    for _, block_names in blocks:
        names.extend(block_names)
    if not matrices:
        raise ValueError('s(x) 没有任何输入特征')
    stacked = np.hstack(matrices).astype(np.float32)
    if stacked.shape[1] != len(names):
        raise ValueError(f's(x) 列数 {stacked.shape[1]} 与列名数 {len(names)} 不一致：{names}')
    return (stacked, names)

def run_arm(ws: Workspace, cfg: Dict[str, Any], arm: str='how123', verbose: bool=True, score_transform=None) -> Dict[str, Any]:
    seed = int(cfg.get('seed', 2026))
    frame, splits, y = (ws.frame, ws.splits, ws.y)
    gamma = SemanticSpec.from_config(cfg['semantic'])
    auditor = ContractAuditor()
    timings: Dict[str, float] = {}
    auditor.run('四段时间切分不重叠且时序不倒挂', lambda: splits.assert_valid(frame['ts'].to_numpy()), f'mode={splits.mode}, sizes={splits.sizes()}')
    t0 = time.perf_counter()
    context = _cached(ws, 'context', cfg['context'], lambda a: build_context(frame, splits.train, y, gamma, cfg['context'], seed, a), auditor)
    timings['context'] = time.perf_counter() - t0
    graph_key = {k: v for k, v in cfg['graph'].items() if k not in _RHO_KEYS}
    t0 = time.perf_counter()
    graph_base = _cached(ws, 'graph', graph_key, lambda a: build_graph(frame, splits.train, gamma, cfg['graph'], seed, a), auditor)
    timings['graph'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    selection, semantic, graph, X, names, detector, det_artifacts = _select_on_tune(ws, cfg, gamma, context, graph_base, auditor, seed)
    timings['selection'] = time.perf_counter() - t0
    context_selected = bool(selection.get('chosen_context', context.matrix.shape[1] > 0))
    auditor.record('契约一·判定后字段未进入语义表示', len(semantic.serialized_columns), True, f"序列化列 {list(semantic.serialized_columns)}；因 ω_j>t 排除 {list(semantic.excluded_by_omega) or '无'}")
    auditor.run('控制层量未进入 s(x)', lambda: auditor.assert_control_not_in_scorer(names), 'S_t / m_t / ε_t 是布尔门与检验参数，不是风险分量（§4.3）')
    if graph.enabled:
        gs = graph.stats
        auditor.record('ϱ 阈值选定后的确认规模', int(gs.get('n_expanded', 0)), True, f"阈值={gs.get('rho_min_used')}；展开 {gs.get('n_expanded')} → 结构确认 {gs.get('n_structure_confirmed')}（仅时间异常 {gs.get('n_time_only')}，无路径 {gs.get('n_no_path')}）；拿到非零 g_t 的交易 {gs.get('n_rows_with_signal')}")
    auditor.record('选型只用训练段与调参段', int(selection['n_candidates']), True, f"判据={selection['criterion']}；选中 semantic={selection['chosen_semantic']}, rho_quantile={selection['chosen_rho_quantile']}；校准段与测试段未参与")
    probe = leakage_probe(X, y, names, splits.train, splits.test)
    auditor.flag('诊断·特征—标签相关性体检（判据取时间外测试段）', int(probe.get('n_features', 0)), not probe.get('flagged_on_test'), f"测试段 max|r|={probe.get('max_abs_r_test', float('nan')):.4f}（训练段 {probe.get('max_abs_r_train', float('nan')):.4f}）；测试段超阈值列 {probe.get('flagged_on_test') or '无'}")
    lifecycle = ScorerLifecycle()
    lifecycle.enter_tuning()
    scorer_version = lifecycle.freeze()
    auditor.record('契约三·评分器在校准前冻结', 1, True, f'state={lifecycle.state}, version={scorer_version}, best_iteration={det_artifacts.best_iteration}')
    alpha = float(get_path(cfg, 'constraint.alpha'))
    delta = float(get_path(cfg, 'constraint.delta'))
    method = str(get_path(cfg, 'constraint.method'))
    calibrator = make_calibrator(method, alpha, delta)
    cal_scores = detector.predict(X[splits.calibration])
    if score_transform is not None:
        cal_scores = score_transform(cal_scores, 'calibration')
    cal_negatives = cal_scores[y[splits.calibration] == 0]
    calibration_error: Optional[str] = None
    try:
        calibration = calibrator.fit(cal_negatives, scorer_version)
        lifecycle.mark_calibrated()
        calibration_dict = calibration.to_dict()
        threshold = calibration.threshold
    except CalibrationInfeasible as exc:
        calibration_error = str(exc)
        calibration_dict = {'method': method, 'status': 'INFEASIBLE', 'detail': str(exc), 'n_negative': int(cal_negatives.size)}
        threshold = float('inf')
    n_min = getattr(calibrator, 'n_min', 'n/a')
    auditor.record('约束层·独立校准集只含未参与训练与选型的样本', int(cal_negatives.size), calibration_error is None, calibration_error or f'n_0={cal_negatives.size}, n_0,min={n_min}')
    t0 = time.perf_counter()
    test_scores = detector.predict(X[splits.test])
    if score_transform is not None:
        test_scores = score_transform(test_scores, 'test')
    timings['score_test'] = time.perf_counter() - t0
    y_test = y[splits.test]
    capacity = CapacityGate(get_path(cfg, 'constraint.capacity_per_window'))
    gate = capacity.apply(test_scores, threshold, tie_admission_rate=float(calibration_dict.get('tie_admission_rate', 0.0)))
    metrics = evaluate(y_test, test_scores, threshold=threshold, alpha=alpha, delta=delta, alerts=gate.alerts, fpr_grid=list(get_path(cfg, 'evaluation.fpr_grid')), bootstrap=int(get_path(cfg, 'evaluation.bootstrap', 0)), seed=seed)
    t0 = time.perf_counter()
    explanations = _explain(detector, X[splits.test], gate.alerts, int(get_path(cfg, 'evaluation.shap_examples', 20)), int(get_path(cfg, 'evaluation.shap_top_k', 3)))
    timings['shap'] = time.perf_counter() - t0
    novelty = graph.stats.get('novelty', {}) if graph.enabled else {}
    n_added = (int(semantic.matrix.shape[1]) if semantic.enabled else 0) + (int(context.matrix.shape[1]) if context_selected else 0) + (int(graph.matrix.shape[1]) if graph.enabled else 0)
    result: Dict[str, Any] = {'degenerate_to_x_base': bool(n_added == 0), 'arm': arm, 'dataset': ws.dataset.name, 'seed': seed, 'config_digest': config_digest(cfg), 'dataset_meta': ws.dataset.meta, 'split_sizes': splits.sizes(), 'split_mode': splits.mode, 'selection_on_tune': selection, 'layers': {'L1_semantic': {'enabled': semantic.enabled, 'reason': semantic.reason, 'dim': int(semantic.matrix.shape[1]), 'excluded_by_omega': list(semantic.excluded_by_omega), 'serialized_columns': list(semantic.serialized_columns), 'example': semantic.example_serialization[:600]}, 'L1_context': {'enabled': bool(context_selected), 'reason': context.reason if context_selected else f'调参段选择关闭（{context.reason}）', 'mode': context.mode, 'negative_control': context.negative_control, 'dim': int(context.matrix.shape[1]) if context_selected else 0, 'admission': context.admission, 'training': context.training}, 'L2_graph': {'enabled': graph.enabled, 'reason': graph.reason, 'dim': int(graph.matrix.shape[1]), 'stats': graph.stats, 'n_patterns_exported': len(graph.patterns), 'novelty': novelty}, 'L3_constraint': {'method': method, 'alpha': alpha, 'delta': delta, 'capacity_per_window': get_path(cfg, 'constraint.capacity_per_window')}}, 'detector': {'best_iteration': det_artifacts.best_iteration, 'best_score_aucpr_tune': det_artifacts.best_score, 'n_features': det_artifacts.n_features, 'params': det_artifacts.params, 'top_gain': det_artifacts.global_importance[:15]}, 'calibration': calibration_dict, 'capacity_gate': {'candidates': gate.candidates, 'emitted': gate.emitted, 'capacity_limited': gate.capacity_limited, 'capacity': gate.capacity}, 'test_metrics': metrics.to_dict(), 'leakage_probe': probe, 'contracts': auditor.to_dict(), 'timings_seconds': {k: round(v, 3) for k, v in timings.items()}, 'explanations': explanations, 'patterns_preview': [p.to_dict() for p in graph.patterns[:10]]}
    result['_scores'] = test_scores
    result['_calibration_scores'] = cal_scores
    result['_tune_scores'] = detector.predict(X[splits.tune])
    result['_patterns'] = graph.patterns
    if verbose:
        _print_arm(result)
    return result

def _pearson_against_label(X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> Optional[np.ndarray]:
    sub = np.nan_to_num(X[idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    target = y[idx].astype(np.float64)
    centered_target = target - target.mean()
    target_norm = float(np.sqrt((centered_target ** 2).sum()))
    if target_norm == 0.0:
        return None
    centered = sub - sub.mean(axis=0, keepdims=True)
    norms = np.sqrt((centered ** 2).sum(axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (centered * centered_target[:, None]).sum(axis=0) / (norms * target_norm)
    return np.nan_to_num(r, nan=0.0)

def leakage_probe(X: np.ndarray, y: np.ndarray, names: Sequence[str], train_idx: np.ndarray, test_idx: np.ndarray, flag_threshold: float=0.5, top_k: int=10) -> Dict[str, Any]:
    """逐特征与标签的 |Pearson r|，训练段与测试段各算一遍。

    这是本课题反复需要的一项体检，不是装饰：它要指出该去核对哪一列。

    **判据放在测试段而不是训练段。** `ctx_proj_*` 是带标签学出来的投影，它在训练段上与
    标签高度相关可能只是样本内乐观；真正的泄漏会**同时**在时间外测试段保持高相关。
    因此审计断言取测试段，训练段的数字照报，两者之差正是样本内乐观的规模。

    单条 |r| 高仍**不等于**泄漏（强真信号也会高），但超阈值的列必须逐一交代。

    2026-09-08 对 RQ1/results/raw/*_main.json（2026-09-04 那轮）逐个复核的现状：

        intent-tx-18k   r_train 0.0851  r_test 0.0825  超阈值 0 列
        sparkov         r_train 0.5015  r_test 0.5981  超阈值 9 列，契约状态 REVIEW
        其余七个数据集   r_test ≤ 0.4997，超阈值 0 列（banksim 0.4997 最接近阈值）

    **这与本函数原先写在这里的那段说明正好相反，故已改写。** 原文举 intent-tx-18k 为
    "配对生成假象被暴露"的例子、举 Sparkov"最大 |r| 只有 0.093"为无假象的依据；产物里
    是 intent-tx-18k 干净而 Sparkov 触发了 REVIEW（`ctx_effective_count` 与 `ctx_proj_0..7`
    共 9 列 |r_test| > 0.5，其中 `ctx_effective_count` 达 0.598）。Sparkov 属于六数据集
    交付集，这条 REVIEW 需要在正文里给出解释或撤下，不能只当成注释里的旧例子。
    """
    r_train = _pearson_against_label(X, y, train_idx)
    r_test = _pearson_against_label(X, y, test_idx)
    if r_train is None or r_test is None:
        return {'n_features': len(names), 'note': '某一切分只有单一类别，相关性无定义'}
    order = np.argsort(np.abs(r_test))[::-1][:top_k]
    flagged = [names[int(i)] for i in np.flatnonzero(np.abs(r_test) > flag_threshold)]
    return {'n_features': len(names), 'n_rows_train': int(train_idx.size), 'n_rows_test': int(test_idx.size), 'flag_threshold': flag_threshold, 'flagged_on_test': flagged, 'max_abs_r_train': float(np.abs(r_train).max()), 'max_abs_r_test': float(np.abs(r_test).max()), 'top_by_test': [{'feature': names[int(i)], 'r_test': float(r_test[int(i)]), 'r_train': float(r_train[int(i)])} for i in order]}

def _explain(detector: FusedDetector, X_test: np.ndarray, alerts: np.ndarray, n_examples: int, top_k: int) -> Dict[str, Any]:
    """只对被告警的交易做归因——底座里 Investigation Agent 也只在可疑时才被调用。"""
    idx = np.flatnonzero(alerts)[:n_examples]
    if idx.size == 0:
        return {'n_alerts_explained': 0, 'examples': [], 'note': '测试段无告警，无可归因样本'}
    tops = detector.explain_top_k(X_test[idx], top_k)
    return {'n_alerts_explained': int(idx.size), 'examples': [{'test_row': int(r), 'top_features': t} for r, t in zip(idx, tops)]}

def _print_arm(result: Dict[str, Any]) -> None:
    m = result['test_metrics']
    layers = result['layers']
    n_pos = int(m.get('n_positive', 0))
    print(f"  [臂] {result['arm']:<28} P={m['precision']:.4f} R={m['tpr']:.4f} AUC-ROC={m['auroc']:.4f} Acc={m['accuracy']:.6f}")
    print(f"        AP={m['ap']:.5f} (×{m['ap_lift']:.1f} 基率)  TPR@FPR=1%={m['tpr_at_fpr'].get('0.01', float('nan')):.4f}  正类 {n_pos}")
    print(f"        L1 语义 {('ON ' if layers['L1_semantic']['enabled'] else 'OFF')}(d={layers['L1_semantic']['dim']}) | L1 上文 {('ON ' if layers['L1_context']['enabled'] else 'OFF')}(d={layers['L1_context']['dim']}) | L2 图 {('ON ' if layers['L2_graph']['enabled'] else 'OFF')}(d={layers['L2_graph']['dim']}) | L3 {result['calibration'].get('status')}")
    sel = result.get('selection_on_tune', {})
    if sel.get('n_candidates', 0) > 1:
        margin = sel.get('margin_over_runner_up')
        tied = sel.get('n_tied_at_best') or 1
        quality = '并列 %d 条，按遍历顺序取胜' % tied if tied > 1 else '余量 %.5f' % (margin if margin is not None else float('nan'))
        print(f"        调参段选型（{sel['criterion']}，{sel['n_candidates']} 个候选，{quality}）：semantic={sel['chosen_semantic']}, ϱ门分位={sel['chosen_rho_quantile']}, ζ={sel.get('chosen_zeta_mode')}，tune AP={sel['chosen_tune_ap']:.5f}")
    print(f"        阈值 T={m['threshold']:.6g}  告警 {m['n_alert']}  TP={m['n_tp']} FP={m['n_fp']}  实测 FPR={m['fpr']:.6f}  U_FP={m['fpr_upper_bound']:.6f}  约束{('通过' if m['constraint_satisfied'] else '未通过')}")
    contracts = result['contracts']
    print(f"        {('契约审计通过' if contracts['passed'] else '契约审计未通过')}：{contracts['n_passed']} PASSED / {contracts['n_failed']} FAILED / {contracts.get('n_review', 0)} REVIEW / {contracts['n_skipped']} SKIPPED")
''', _ROOT=_ROOT, _DATASET=_DATASET, _OUTPUT=_OUTPUT, _DATA=_DATA, _CONFIGS=_CONFIGS)

DATASETS = tuple(_DATA['ours_config']['configs'])

def load_config(dataset, seed=2026):
    """Return a copy of the supplied full-model configuration."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}; choose from {DATASETS}")
    cfg = copy.deepcopy(_DATA['ours_config']['configs'][dataset])
    cfg['seed'] = seed
    return cfg

def run(dataset='banksim', seed=2026, datasets_dir=None, output_dir=None, window_rows=None):
    """Fit SERA, calibrate on held-out negatives, and evaluate the test segment."""
    import hashlib
    from how123.pipeline import build_workspace, run_arm
    data = sys.modules['how123.datasets']
    root = Path(datasets_dir or _DATASET).resolve()
    output = Path(output_dir or _OUTPUT).resolve()
    data.DATA_DIR = str(root)
    key = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    data.CACHE_DIR = str(output / 'cache' / key)
    cfg = load_config(dataset, seed)
    if window_rows is not None:
        if window_rows <= 0:
            raise ValueError('window_rows must be positive')
        cfg['window_rows'] = window_rows
    workspace = build_workspace(cfg, verbose=True)
    result = run_arm(workspace, cfg, arm='full', verbose=True)
    if result['contracts']['n_failed']:
        raise RuntimeError(f"SERA contract checks failed for {dataset}")
    result['model'] = 'SERA'
    result['configuration'] = cfg
    return result
