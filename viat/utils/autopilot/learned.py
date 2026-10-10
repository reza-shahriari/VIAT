"""Learned judge: a small classifier trained on decisions about real tracks.

Every review decision (Accept / Delete / Relabel / Restore / add a miss) is appended
to ``decisions.jsonl`` with the track's features and the other judges' answers.
``autopilot_benchmark.py`` adds rows automatically from your hand-labeled videos.
Training fits a logistic regression (scikit-learn) and stores it as plain JSON, so
the judge learns *your* labeling rules (e.g. "the turret commander is not labeled").
"""
import json
import os
import time

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
except ImportError:
    LogisticRegression = None

TRACK_FEATURES = ['mean_score', 'max_score', 'p25_score', 'support', 'span_sec', 'log_size', 'links',
                  'ext_frac', 'class_margin', 'agree', 'inside_frac', 'jumps', 'overlap', 'promoted',
                  'longest_gap_sec']
JUDGE_FEATURES = ['clip', 'sam_zoom', 'vlm']
MODEL_FILE = 'learned_judge.json'
LOG_FILE = 'decisions.jsonl'


def _workdir(workdir):
    if workdir:
        return workdir
    base = (os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'VideoAnnotationTool')
            if os.name == 'nt' else os.path.join(os.path.expanduser('~'), '.config', 'VideoAnnotationTool'))
    return os.path.join(base, 'autopilot')


def vectorize(features, judges):
    x = [float(features.get(k, 0.0)) for k in TRACK_FEATURES]
    for k in JUDGE_FEATURES:
        v = judges.get(k) if judges else None
        x += [0.5 if v is None else float(v), 0.0 if v is None else 1.0]
    return x


def feature_names():
    names = list(TRACK_FEATURES)
    for k in JUDGE_FEATURES:
        names += [k, f'has_{k}']
    return names


def log_decision(track, label, source='user', video='', workdir=None):
    """label: 1 = real object of this class, 0 = not."""
    if 'features' not in track:
        return
    path = os.path.join(_workdir(workdir), LOG_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'source': source, 'video': video,
           'class': track.get('class'), 'label': int(label), 'features': track['features'],
           'judges': {k: v for k, v in (track.get('judges') or {}).items() if k in JUDGE_FEATURES}}
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(row) + '\n')


def load_decisions(workdir=None):
    path = os.path.join(_workdir(workdir), LOG_FILE)
    rows = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def train(workdir=None, min_rows=40):
    """Fit the judge from the decision log. Returns a summary dict (or raises ValueError)."""
    if LogisticRegression is None:
        raise ValueError('scikit-learn is not installed')
    rows = load_decisions(workdir)
    X = np.array([vectorize(r['features'], r.get('judges')) for r in rows], np.float64)
    y = np.array([r['label'] for r in rows], int)
    if len(rows) < min_rows or y.sum() < 10 or (len(y) - y.sum()) < 10:
        raise ValueError(f'need at least {min_rows} decisions with 10+ of each kind '
                         f'(have {len(rows)}: {int(y.sum())} real, {int(len(y) - y.sum())} not real)')
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mean) / std
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
    auc = float(np.mean(cross_val_score(clf, Xs, y, cv=min(5, int(min(y.sum(), len(y) - y.sum()))),
                                        scoring='roc_auc')))
    clf.fit(Xs, y)
    model = {'features': feature_names(), 'mean': mean.tolist(), 'std': std.tolist(),
             'coef': clf.coef_[0].tolist(), 'intercept': float(clf.intercept_[0]),
             'n': int(len(y)), 'n_real': int(y.sum()), 'cv_auc': round(auc, 4),
             'trained': time.strftime('%Y-%m-%d %H:%M:%S')}
    path = os.path.join(_workdir(workdir), MODEL_FILE)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(model, fh, indent=1)
    weights = sorted(zip(model['features'], model['coef']), key=lambda kv: -abs(kv[1]))[:6]
    return {'rows': model['n'], 'real': model['n_real'], 'cv_auc': model['cv_auc'], 'top_features': weights,
            'path': path}


class LearnedJudge:
    def __init__(self, model):
        self.m = model
        self.mean = np.array(model['mean'])
        self.std = np.array(model['std'])
        self.coef = np.array(model['coef'])

    @classmethod
    def load(cls, workdir=None):
        path = os.path.join(_workdir(workdir), MODEL_FILE)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                model = json.load(fh)
            if model.get('features') != feature_names():
                return None  # trained with another feature set: retrain
            return cls(model)
        except (OSError, ValueError, KeyError):
            return None

    def predict(self, features, judges):
        x = (np.array(vectorize(features, judges)) - self.mean) / self.std
        z = float(x @ self.coef + self.m['intercept'])
        return 1.0 / (1.0 + np.exp(-z))
