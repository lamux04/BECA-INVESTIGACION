import pandas as pd
import numpy as np
import glob
import os

def get_file_list(path, ext='*.csv'):
    return glob.glob(os.path.join(path, ext))

def open_and_merge_dataset_filelist(all_files, encoding='utf-8'):
    return pd.concat((pd.read_csv(f, encoding=encoding) for f in all_files), ignore_index=True)

def open_and_merge_dataset(path, encoding='utf-8'):
    all_files = get_file_list(path)
    return open_and_merge_dataset_filelist(all_files, encoding)

# Sanitizing column names
def sanitize_column_names(input_df):
    df = input_df.copy()
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.upper()
    return df

# Datetime to timestamp
def datetime_to_timestamp(input_df, column, format=None):
    df = input_df.copy()
    df[column] = pd.to_datetime(df[column], format=format).astype(int) / 10**9
    return df

# Removing values with std deviation 0 or almost 0, which means that they have unique values or almost all the values are the same
def get_columns_std_dev_equals(df, input_std):
    std = df[df.select_dtypes(include=[int, float]).columns].std()
    return std[std==input_std].index

# Getting values with std deviation 0( or almost 0), which means that they have unique values (or almost all the values are the same)
def get_columns_std_dev_less_equals(df, input_std):
    std = df[df.select_dtypes(include=[int, float]).columns].std()
    return std[std<=input_std].index

# Removing values with std deviation 0( or almost 0), which means that they have unique values (or almost all the values are the same)
def remove_columns_std_dev_equals_zero(df):
    cols = get_columns_std_dev_equals(df, 0)
    return df.drop(cols, axis=1)

# Getting columns with high correlation
def get_columns_high_correlation(df, input_corr = 0.95):
    df_num = df.select_dtypes(exclude=['object'])
    cor_matrix = df_num.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
    return [column for column in upper_tri.columns if any(upper_tri[column] > input_corr)]

# Dropping columns with high correlation
def remove_columns_high_correlation(df, input_corr = 0.95):
    to_drop = get_columns_high_correlation(df, input_corr)
    return df.drop(to_drop, axis=1)

# Replacing infinite values by NaN
def replace_inf_by_nan(input_df):
    df = input_df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# Removing missing values
def remove_missing_values(df):
    return df.dropna()

# Removing duplicates
def remove_duplicate_rows(df):
    return df.drop_duplicates(keep='first', ignore_index=True)

# Saving dataframe
def save_dataframe(df, name):
    df.to_csv(name, index=False)
    return df

def save_dataframe_split(df, labels, label_column, filename_pattern):
    for attack_name, label in labels.items():
        print(attack_name + ':')
        df_split = df.loc[df[label_column] == attack_name]
        df_split.to_csv(filename_pattern.format(label), index=False)
        print(df_split[label_column].value_counts())
    return df

def remove_columns(df, columns):
    for column in columns:
        if column in df.columns:
            del df[column]
    return df

def keep_columns(df, columns):
    df = df[columns]
    return df

def print_existing_columns(df, columns):
    for column in columns:
        if column in df.columns:
            print('Removed column ' + column)
    return df

def print_column_types(df, numeric_types=['float64', 'int64']):
    df_num = df.select_dtypes(exclude=['object'])
    df_string = df.select_dtypes(exclude=numeric_types)
    df_num.shape, df_string.shape
    print('Columns: {} {}; {} object'.format(df_num.shape[1], str(numeric_types).strip('[]').replace('\'', ''), df_string.shape[1]))
    return df

def print_object_columns(df, numeric_types=[np.int64, np.float64]):
    print(df.dtypes[~df.dtypes.map(lambda x: x in numeric_types)])
    return df

def print_shape(df):
    print('Dimensions: {} columns x {} rows'.format(df.shape[1], df.shape[0]))
    return df

def get_shape(df):
    return df.shape

def get_columns(df):
    return df.columns

def check_type(df, column_name, expected_type):
    return df[column_name][~df[column_name].map(lambda x: isinstance(x, expected_type))]



# -------------------------------------------------------------------------------
#                                  Preparation
# -------------------------------------------------------------------------------
def remove_fraction_of_rows(df, fraction):
    return df.drop(df.sample(frac=fraction).index)

def remove_fraction_of_class_rows(df, class_label, label, fraction):
    return remove_fraction_of_rows(df[df[class_label] == label], fraction)

def concatenate_columns(input_df, label_column, new_column):
    df = input_df.copy()
    if new_column in df:
        df.drop(columns=[new_column], inplace=True)
    df[new_column] = df[df.columns[df.columns != label_column]].agg(lambda x: '|'.join(('%.8f' % v).rstrip('0').rstrip('.') for v in x.values), axis=1).T
    return df

def select_columns_remove_unselected(df, selected_columns):
    return df[selected_columns].copy()

def rename_columns(df, columns):
    return df.rename(columns=columns)

def normalize_columns(df):
    return (df - df.min()) / (df.max() - df.min())

def normalize_all_columns_except(input_df, excluded_columns):
    df = input_df.copy()
    df.loc[:, ~df.columns.isin(excluded_columns)] = normalize_columns(df.loc[:, ~df.columns.isin(excluded_columns)])
    return df

def prepare_labels(input_df, label_column, labels):
    df = input_df.copy()
    df[label_column].replace(labels, inplace=True)
    return df

def print_value_counts(df, column):
    print(df[column].value_counts())
    return df

def get_build_custom_dataset(filename_pattern, dataset_config):
    dataset_parts = []
    for attack_config in dataset_config:
        label = attack_config['label']
        label_data_dup = 'dup' if attack_config['duplicates'] else 'no-dup'
        fraction = attack_config['fraction']
        filename = filename_pattern.format(label_data_dup, label_data_dup, label)
        df_part = pd.read_csv(filename)
        if fraction < 1:
            df_part = remove_fraction_of_rows(df_part, 1 - fraction)
        dataset_parts.append(df_part)
    return pd.concat(dataset_parts)



# -------------------------------------------------------------------------------
#                                  Splitting
# -------------------------------------------------------------------------------
def stratified_kfold_cross_validation(df, k, test_fraction, labels, prefix, ext):
    selected_columns=['source_text', 'target_text']
    df['group'] = 0
    output_files = []

    for label in labels:
        df.loc[df[(df['group'] == 0) & (df['target_text'] == label)].sample(frac=test_fraction, random_state=1).index, 'group'] = k + 1
    
    for group in range(1, k + 1): # 1..5
        for label in labels:
            df.loc[df[(df['group'] == 0) & (df['target_text'] == label)].sample(frac=1/(k-group+1), random_state=1).index, 'group'] = group

    df_split = df.loc[df['group'] == k + 1]
    test_file = '{}{}{}'.format(prefix, 'test', ext)
    df_split.loc[:, selected_columns].to_csv(test_file, index=False)
    
    for i in range(1, k + 1): # 1..5
        df_split = df.loc[(df['group'] != i) & (df['group'] != k + 1)]
        train_file = '{}{}{}{}'.format(prefix, i, '-train', ext)
        df_split.loc[:, selected_columns].to_csv(train_file, index=False)
        df_split = df.loc[df['group'] == i]
        val_file = '{}{}{}{}'.format(prefix, i, '-val', ext)
        df_split.loc[:, selected_columns].to_csv(val_file, index=False)
        output_files.append(train_file)
        output_files.append(val_file)

    output_files.append(test_file)
    return output_files

def stratified_kfold_cross_validation_equal_parts(df, n_classes=15, prefix='03-split-', ext='.csv', parts=5, has_test=False, selected_columns=['source_text', 'target_text']):
    cond = True
    df['group'] = 0
    
    if has_test:
        parts += 1
    
    for group in range(1, parts + 1):
        for i in range(n_classes + 1):
            df.loc[df[(df['group'] == 0) & (df['target_text'] == i)].sample(frac=1/(parts-group+1)).index, 'group'] = group

    if has_test:
        parts -= 1
        cond = df['group'] != parts + 1
        df_split = df.loc[df['group'] == parts + 1]
        df_split.loc[:, selected_columns].to_csv('{}{}{}'.format(prefix, 'test', ext), index=False)
    
    for i in range(1, parts + 1):
        df_split = df.loc[(df['group'] != i) & cond]
        df_split.loc[:, selected_columns].to_csv('{}{}{}{}'.format(prefix, i, '-train', ext), index=False)
        df_split = df.loc[df['group'] == i]
        df_split.loc[:, selected_columns].to_csv('{}{}{}{}'.format(prefix, i, '-val', ext), index=False)
    
    #return df

def stratified_kfold_cross_validation_bm_not_a_class(df, k, test_label, labels, prefix, ext):
    selected_columns=['source_text', 'target_text']
    df['group'] = 0
    output_files = []

    df.loc[df[(df['group'] == 0) & (df['target_text'] == test_label)].index, 'group'] = k + 1
    
    for group in range(1, k + 1): # 1..5
        for label in labels:
            if label != test_label:
                df.loc[df[(df['group'] == 0) & (df['target_text'] == label)].sample(frac=1/(k-group+1), random_state=1).index, 'group'] = group

    df_split = df.loc[df['group'] == k + 1]
    test_file = '{}{}{}'.format(prefix, 'test', ext)
    df_split.loc[:, selected_columns].to_csv(test_file, index=False)
    
    for i in range(1, k + 1): # 1..5
        df_split = df.loc[(df['group'] != i) & (df['group'] != k + 1)]
        train_file = '{}{}{}{}'.format(prefix, i, '-train', ext)
        df_split.loc[:, selected_columns].to_csv(train_file, index=False)
        df_split = df.loc[df['group'] == i]
        val_file = '{}{}{}{}'.format(prefix, i, '-val', ext)
        df_split.loc[:, selected_columns].to_csv(val_file, index=False)
        output_files.append(train_file)
        output_files.append(val_file)

    output_files.append(test_file)
    return output_files