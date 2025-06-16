import pandas as pd

baseline_categories = {
    'age_group': '30-35',
    'bmi_group': 'normal',
    'been_pregnant_before_binary': False,
    'average_cycle_length_group': '21-35',
    'regular_cycle': True,
    'university_education': False,
    'regular_sleep': True,
    'intercourse_frequency_group': 'medium',
    'dedication_group': 'medium',
    'cycle_cat': '1-3'
}

education_map = {
    "Elementary school": 0,
    "High school": 1,
    "Trade/technical/vocational training": 2,
    "University": 3,
    "PhD": 4
}

sleeping_map = {
    "Several times during the night": 0,
    "Shift worker": 1,
    "Late and snoozer": 2,
    "Wake same every workday": 3,
    "Wake same every day": 4
}

pregnancy_map = {
    "No, never": 0,
    "Yes, once": 1,
    "Yes, twice": 2,
    "Yes 3 times or more": 3
}

outcome_map = {
    "pregnant": 1, 
    "not_pregnant": 0
}


def age_bins(age):
    if pd.isna(age):
        return pd.NA
    elif age <= 29:
        return '19-29'
    elif age <= 35:
        return '30-35'
    else:
        return '35-44'

def intercourse_freq_bins(freq):
    if pd.isna(freq):
        return pd.NA
    elif freq <= 0.02:
        return 'low'
    elif freq <= 0.162:
        return 'medium'
    else:
        return 'high'

def dedication_bins(ded):
    if pd.isna(ded):
        return pd.NA
    elif ded <= 0.133:
        return 'low'
    elif ded <= 0.781:
        return 'medium'
    else:
        return 'high'

def bmi_bins(bmi):
    if pd.isna(bmi):
        return pd.NA
    elif bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    else:
        return 'overweight'

def cycle_length_range(length):
    if pd.isna(length):
        return pd.NA
    elif 21 <= length <= 35:
        return '21-35'
    else:
        return '<21 OR >35'
