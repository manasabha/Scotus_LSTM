
# coding: utf-8

# In[4]:


"""
Supreme Court prediction model library.

NOTE : This file is taken from https://github.com/mjbommar/scotus-predict-v2/. It has been modified to remove all feature engineering. Rest of the code is simply reused in order to not duplicate effort.

Also, we wanted to reuse the creation of the outcome map since that required legal know how.

@date 20150926
@author mjbommar
"""

# Imports
import datetime
import dateutil.parser
import numpy
import os
import pandas
import scipy.stats
import statsmodels

# sklearn imports
import sklearn
import sklearn.dummy
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.preprocessing

# Project imports
from legacy_model_data import party_map_data, court_circuit_map


# Path constants
DATA_PATH="../data/"
SCDB_RELEASE="2015_01"

# Model constants
SCDB_OUTCOME_MAP=None


def get_raw_scdb_data(scdb_path=None):
    """
    Get raw SCDB data in pandas.DataFrame.
    """
    
    # Get path
    if not scdb_path:
        scdb_path = os.path.join(DATA_PATH,
                                 "SCDB_{0}_justiceCentered_Citation.csv".format(SCDB_RELEASE))
    
    # Load and return
    raw_scdb_df = pandas.read_csv(scdb_path, encoding = "ISO-8859-1", low_memory=False)
    
    # Get outcome data
    outcome_map = get_outcome_map()
    raw_scdb_df.loc[:, "case_outcome_disposition"] = outcome_map.loc[1, raw_scdb_df.loc[:, "caseDisposition"]].values
    raw_scdb_df.loc[:, "lc_case_outcome_disposition"] = outcome_map.loc[1, raw_scdb_df.loc[:, "lcDisposition"]].values

    # Map the justice-level disposition outcome
    raw_scdb_df.loc[:, "justice_outcome_disposition"] = raw_scdb_df.loc[:, ("vote", "caseDisposition")]         .apply(lambda row: get_outcome(row["vote"], row["caseDisposition"]), axis=1)

    return raw_scdb_df


def get_outcome_map():
    """
    Get the outcome map to convert an SCDB outcome into
    an affirm/reverse/other mapping.
    
    Rows correspond to vote types.  Columns correspond to disposition types.
    Element values correspond to:
    * -1: no precedential issued opinion or uncodable, i.e., DIGs
    * 0: affirm, i.e., no change in precedent
    * 1: reverse, i.e., change in precent
    """

    # Create map; see appendix of paper for further documentation
    outcome_map = pandas.DataFrame([[-1, 0, 1, 1, 1, 0, 1, -1, -1, -1, -1],
               [-1, 1, 0, 0, 0, 1, 0, -1, -1, -1, -1],
               [-1, 0, 1, 1, 1, 0, 1, -1, -1, -1, -1],
               [-1, 0, 1, 1, 1, 0, 1, -1, -1, -1, -1],
               [-1, 0, 1, 1, 1, 0, 1, -1, -1, -1, -1],
               [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
               [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
               [-1, 0, 0, 0, -1, 0, -1, -1, -1, -1, -1]])
    outcome_map.columns = range(1, 12)
    outcome_map.index = range(1, 9)

    return outcome_map


def get_outcome(vote, disposition, outcome_map=SCDB_OUTCOME_MAP):
    """
    Return the outcome code based on outcome map.
    """
    if not outcome_map:
        SCDB_OUTCOME_MAP=get_outcome_map()
        outcome_map = SCDB_OUTCOME_MAP

    if pandas.isnull(vote) or pandas.isnull(disposition):
        return -1
    
    return outcome_map.loc[int(vote), int(disposition)]


def get_unique_values(values):
    """
    Get unique, sorted list of values with NA coding.
    """
    # Initialize list and extend
    value_list = [-1]
    value_list.extend(values.fillna(-1).astype(int).unique())

    # Return sorted set
    return sorted(set(value_list))


def binarize_values(values):
    """
    Binarize values.
    """
    # Get value codes 
    value_codes = get_unique_values(values)
    
    # Return value codes and binarized matrix
    return value_codes, sklearn.preprocessing.label_binarize(values.fillna(-1).astype(int),
                                    value_codes)


def get_date(value):
    """
    Get date value from SCDB string format.
    """
    try:
        return datetime.datetime.strptime(value, "%m/%d/%Y").date()
    except:
        return None


def get_date_month(value):
    """
    Get month from date.
    """
    try:
        return value.month
    except:
        return -1


def map_circuit(value):
    try:
        return court_circuit_map[value]
    except:
        return 0

    
def map_party(value):
    try:
        return party_map_data[value]
    except:
        return 0

def as_column_vector(values):
    # Return values as column vector
    return numpy.array(values, ndmin=2).T


def preprocess_raw_data(raw_data, include_direction=False):
    """
    Preprocess the raw SCDB data.
    """
    # Get chronological variables
    term_raw = as_column_vector(raw_data.loc[:, "term"])
    term_codes, term_encoded = binarize_values(raw_data.loc[:, "term"])
    natural_court_raw = as_column_vector(raw_data.loc[:, "naturalCourt"])
    natural_court_codes, natural_court_encoded = binarize_values(raw_data.loc[:, "naturalCourt"])

    # Get argument/decision features
    if raw_data.loc[:, "dateArgument"].dtype != numpy.dtype('<M8[ns]'):
        argument_date_raw = raw_data.loc[:, "dateArgument"].apply(get_date)
    else:
        argument_date_raw = raw_data.loc[:, "dateArgument"]
    if raw_data.loc[:, "dateDecision"].dtype != numpy.dtype('<M8[ns]'):
        decision_date_raw = raw_data.loc[:, "dateDecision"].apply(get_date)
    else:
        decision_date_raw = raw_data.loc[:, "dateDecision"]

    argument_month_codes, argument_month_encoded = binarize_values(argument_date_raw.apply(get_date_month))
    decision_month_codes, decision_month_encoded = binarize_values(decision_date_raw.apply(get_date_month))

    #decision_delay = ((decision_date_raw - argument_date_raw) / numpy.timedelta64(1, 'W'))\
    #    .fillna(-1)\
    #    .astype(int)

    # Get justice identification variables
    justice_codes, justice_encoded = binarize_values(raw_data.loc[:, "justice"])

    # Binarize and bin parties
    petitioner_codes, petitioner_encoded = binarize_values(raw_data.loc[:, 'petitioner'])
    respondent_codes, respondent_encoded = binarize_values(raw_data.loc[:, 'respondent'])
    petitioner_group_codes, petitioner_group_encoded = binarize_values(raw_data.loc[:, 'petitioner'].apply(map_party))
    respondent_group_codes, respondent_group_encoded = binarize_values(raw_data.loc[:, 'respondent'].apply(map_party))

    # Encode jurisdiction and other procedural metadata
    jurisdiction_codes, jurisdiction_encoded = binarize_values(raw_data.loc[:, 'jurisdiction'])
    admin_action_codes, admin_action_encoded = binarize_values(raw_data.loc[:, 'adminAction'])
    case_source_codes, case_source_encoded = binarize_values(raw_data.loc[:, 'caseSource'].apply(map_circuit).astype(int))
    case_origin_codes, case_origin_encoded = binarize_values(raw_data.loc[:, 'caseOrigin'].apply(map_circuit).astype(int))
    lc_disagreement_codes, lc_disagreement_encoded = binarize_values(raw_data.loc[:, 'lcDisagreement'])
    cert_reason_codes, cert_reason_encoded = binarize_values(raw_data.loc[:, 'certReason'])
    lc_outcome_codes, lc_outcome_encoded = binarize_values(raw_data.loc[:, 'lc_case_outcome_disposition'])

    # Encode issue/topical data
    issue_codes, issue_encoded = binarize_values(raw_data.loc[:, "issue"])
    issue_area_codes, issue_area_encoded = binarize_values(raw_data.loc[:, "issueArea"])
    law_type_codes, law_type_encoded = binarize_values(raw_data.loc[:, "lawType"])
    law_supp_codes, law_supp_encoded = binarize_values(raw_data.loc[:, "lawSupp"])

    # Caching for inner loop
    #previous_court_direction_cache = {}
    #cumulative_court_direction_cache = {}
    #previous_court_action_cache = {}
    #cumulative_court_action_cache = {}
    #previous_court_agreement_cache = {}
    #cumulative_court_agreement_cache = {}
    
    # Create final data frame
    feature_data = numpy.hstack((term_raw,
              term_encoded,
              natural_court_raw,
              natural_court_encoded,
              argument_month_encoded,
              decision_month_encoded,
              #as_column_vector(decision_delay),
              justice_encoded,
              petitioner_encoded,
              respondent_encoded,
              petitioner_group_encoded,
              respondent_group_encoded,
              jurisdiction_encoded,
              admin_action_encoded,
              case_source_encoded,
              case_origin_encoded,
              lc_disagreement_encoded,
              cert_reason_encoded,
              lc_outcome_encoded,
              issue_encoded,
              issue_area_encoded,
              law_type_encoded,
              law_supp_encoded
             ))

    feature_labels = ["term_raw"]
    feature_labels.extend(["term_{0}".format(x) for x in term_codes])
    feature_labels.append("natural_court_raw")
    feature_labels.extend(["natural_court_{0}".format(x) for x in natural_court_codes])
    feature_labels.extend(["argument_month_{0}".format(x) for x in argument_month_codes])
    feature_labels.extend(["decision_month_{0}".format(x) for x in decision_month_codes])
    #feature_labels.append("decision_delay")
    feature_labels.extend(["justice_{0}".format(x) for x in justice_codes])
    feature_labels.extend(["petitioner_{0}".format(x) for x in petitioner_codes])
    feature_labels.extend(["respondent_{0}".format(x) for x in respondent_codes])
    feature_labels.extend(["petitioner_group_{0}".format(x) for x in petitioner_group_codes])
    feature_labels.extend(["respondent_group_{0}".format(x) for x in respondent_group_codes])
    feature_labels.extend(["jurisdiction_{0}".format(x) for x in jurisdiction_codes])
    feature_labels.extend(["admin_action_{0}".format(x) for x in admin_action_codes])
    feature_labels.extend(["case_source_{0}".format(x) for x in case_source_codes])
    feature_labels.extend(["case_origin_{0}".format(x) for x in case_origin_codes])
    feature_labels.extend(["lc_disagreement_{0}".format(x) for x in lc_disagreement_codes])
    feature_labels.extend(["cert_reason_{0}".format(x) for x in cert_reason_codes])
    feature_labels.extend(["lc_outcome_{0}".format(x) for x in lc_outcome_codes])
    feature_labels.extend(["issue_{0}".format(x) for x in issue_codes])
    feature_labels.extend(["issue_area_{0}".format(x) for x in issue_area_codes])
    feature_labels.extend(["law_type_{0}".format(x) for x in law_type_codes])
    feature_labels.extend(["law_supp_{0}".format(x) for x in law_supp_codes])

    feature_df = pandas.DataFrame(feature_data,
                                 columns=feature_labels)
    '''
    # Check direction
    if not include_direction:
        feature_df = feature_df.loc[:,
                                    [c for c in feature_df.columns if "direction" not in c]]
    
    # At last, return
    '''
    return feature_df





