import os
import numpy as np
import pandas as pd
from painnet import config
from painnet import data as eeg_data
from painnet import windows as eeg_windows

WATCH_FEATURE_COLS = ['bvp', 'eda', 'x', 'y', 'z', 'temperature']

WATCH_DROP_COLS = ['pain_scale', 'pain_type', 'person_id']

WATCH_HZ = 4
SUBJECT_COL = 'person_id'
LABEL_COL = 'pain_type'

def load_watch_all(tier_hz = WATCH_HZ):
    root = str(config.WATCH_PROCESSED_DIR)

    if not os.path.exists(root):
        raise FileNotFoundError(f"Watch processed directory not found: {root}")

    all_frames = []

    for pain_type in config.PAIN_CLASSES:
        folder = os.path.join(root, pain_type, 'signal_' + str(tier_hz))

        if not os.path.exists(folder):
            found = None
            for name in os.listdir(root):
                if name.replace('u', '') == pain_type.replace('u',''):
                    found = name

            if found is None:
                print("Warning: No folder found for pain type:", pain_type)
                continue

            folder = os.path.join(root, found, 'signal_' + str(tier_hz))
            if not os.path.exists(folder):
                print("Warning: No signal folder found for pain type:", pain_type)
                continue

        filenames = sorted(os.listdir(folder))

        for name in filenames:
            if not name.endswith('.csv'):
                continue

            path = os.path.join(folder, name)
            df = pd.read_csv(path)

            new_cols = []
            for col in df.columns:
                new_cols.append(col.strip())

            df.columns = new_cols

            df = df.iloc[1:]
            df = df.reset_index(drop=True)

            if SUBJECT_COL not in df.columns:
                df[SUBJECT_COL] = name.split('_')[0]

            if LABEL_COL not in df.columns:
                df[LABEL_COL] = pain_type

            all_frames.append(df)
    if len(all_frames) == 0:
        raise ValueError("No data frames were loaded. Please check the directory structure and file names.")

    data = pd.concat(all_frames, ignore_index=True)

    for col in WATCH_FEATURE_COLS:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    for col in WATCH_FEATURE_COLS:
        data[col] = data.groupby(SUBJECT_COL)[col].transform(lambda x: x.ffill().bfill())

    print(f"Loaded watch data with shape: {data.shape}")
    print('Participants:', data[SUBJECT_COL].nunique())

    return data


def watch_summary(data = None):
    if data is None:
        data = load_watch_all()

    rows = []

    for subject_id, group in data.groupby(SUBJECT_COL):
        row = {
            'subject_id': subject_id,
            'num_samples': len(group),
            'num_pain_types': group[LABEL_COL].nunique(),
            'pain_types': ', '.join(group[LABEL_COL].unique())
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.set_index('subject_id')
    return summary_df

def zscore_per_subject(data, feature_cols = WATCH_FEATURE_COLS):
    out = data.copy()

    for col in feature_cols:
        mean = out.groupby(SUBJECT_COL)[col].transform('mean')
        std = out.groupby(SUBJECT_COL)[col].transform('std')

        std = std.replace(0, np.nan)

        out[col] = (out[col] - mean) / std
        out[col] = out[col].fillna(0)

    return out


def build_watch_dataset(window_seconds = config.WINDOW_SECONDS, step_seconds = config.STEP_SECONDS, tier_hz = WATCH_HZ):
    data = load_watch_all(tier_hz)
    data = zscore_per_subject(data)

    window_rows = window_seconds * tier_hz
    step_rows = step_seconds * tier_hz

    X = []
    y = []
    groups = []

    for subject_id, group in data.groupby(SUBJECT_COL, sort = True):
        signals = group[WATCH_FEATURE_COLS].to_numpy(dtype=np.float32)
        label = group[LABEL_COL].iloc[0]
 
        if len(signals) < window_rows:
            print('Skipping', subject_id, '- only', len(signals), 'rows')
            continue
 
        start = 0
        while start + window_rows <= len(signals):
            X.append(signals[start:start + window_rows])
            y.append(label)
            groups.append(subject_id)
            start = start + step_rows

    if len(X) == 0:
        raise ValueError("No windows were created. Please check the window and step sizes.")

    X = np.stack(X)
    y = np.array(y)

    groups = np.array(groups)

    print(f"Built watch dataset with {len(X)} windows, shape: {X.shape}, labels: {np.unique(y)}")
    print("participants:", np.unique(groups))

    return X, y, groups

def build_fusion_dataset(window_seconds = config.WINDOW_SECONDS, step_seconds = config.STEP_SECONDS, tier_hz = WATCH_HZ):
    eeg = eeg_data.load_raw_eeg()
    eeg = eeg_windows.zscore_per_subject(eeg, config.EEG_FEATURE_COLS)

    watch = load_watch_all(tier_hz)
    watch = zscore_per_subject(watch)

    eeg_groups = {}
    for subject_id, group in eeg.groupby(config.SUBJECT_COL):
        eeg_groups[subject_id] = group

    watch_groups = {}
    for subject_id, group in watch.groupby(SUBJECT_COL):
        watch_groups[subject_id] = group

    common_subjects = []
    for subject_id in sorted(eeg_groups.keys()):
        if subject_id in watch_groups:
            common_subjects.append(subject_id)

    print('EEG subjects:', len(eeg_groups), 'Watch subjects:', len(watch_groups), 'Common subjects:', len(common_subjects))

    X_eeg = []
    X_watch = []

    y = []
    groups = []
    skipped = []

    for subject_id in common_subjects:
        eeg_group = eeg_groups[subject_id]
        watch_group = watch_groups[subject_id]

        eeg_signals = eeg_group[config.EEG_FEATURE_COLS].to_numpy(dtype=np.float32)
        watch_signals = watch_group[WATCH_FEATURE_COLS].to_numpy(dtype=np.float32)

        eeg_seconds = len(eeg_signals)
        watch_seconds = len(watch_signals) / tier_hz

        total_seconds = min(eeg_seconds, watch_seconds)

        if total_seconds < window_seconds:
            skipped.append(subject_id)
            continue

        label = eeg_group[config.LABEL_COL].iloc[0]

        t = 0
        while t + window_seconds <= total_seconds:
            eeg_window = eeg_signals[t:t + window_seconds]

            watch_start = t * tier_hz
            watch_end = (t + window_seconds) * tier_hz
            watch_window = watch_signals[watch_start:watch_end]

            if len(eeg_window) == window_seconds:
                if len(watch_window) == window_seconds * tier_hz:
                    X_eeg.append(eeg_window)
                    X_watch.append(watch_window)
                    y.append(label)
                    groups.append(subject_id)
            t += step_seconds

    if len(X_eeg) == 0:
        raise ValueError("No windows were created for the fusion dataset. Please check the window and step sizes.")

    X_eeg = np.stack(X_eeg)
    X_watch = np.stack(X_watch)
    y = np.array(y)
    groups = np.array(groups)

    print('Pain windows: ', len(X_eeg))
    print('X_eeg shape:', X_eeg.shape)
    print('X_watch shape:', X_watch.shape)

    if len(skipped) > 0:
        print('Skipped subjects (not enough data):', skipped)

    print('Class Participants Windows')
    for pain_type in config.PAIN_CLASSES:
        mask = (y == pain_type)
        n_windows = int(mask.sum())
        n_people = len(set(groups[mask]))
        print(pain_type.ljust(16), str(n_people).rjust(12), str(n_windows).rjust(8))
 
    return X_eeg, X_watch, y, groups


def duration_check(tier_hz = WATCH_HZ):
    eeg = eeg_data.load_raw_eeg()
    watch = load_watch_all(tier_hz)

    eeg_counts = eeg.groupby(config.SUBJECT_COL).size()
    watch_counts = watch.groupby(SUBJECT_COL).size()

    rows = []

    for subject_id in sorted(eeg_counts.index):
        if subject_id not in watch_counts.index:
            continue

        eeg_seconds = eeg_counts[subject_id] 
        watch_seconds = watch_counts[subject_id] / tier_hz
        diff = eeg_seconds - watch_seconds

        smaller = min(eeg_seconds, watch_seconds)
        percent = 100 * abs(diff) / smaller

        rows.append({
            'subject_id': subject_id,
            'eeg_seconds': eeg_seconds,
            'watch_seconds': watch_seconds,
            'diff': diff,
            'percent': round(percent, 2)
        })

    result_df = pd.DataFrame(rows)
    result_df = result_df.set_index('subject_id')
    result_df = result_df.sort_values('percent', ascending=False)

    print('Subjects in both tiers:', len(result_df))
    print('Median difference in duration (seconds):', round(result_df['diff'].abs().median(), 1) )

    return result_df
    
