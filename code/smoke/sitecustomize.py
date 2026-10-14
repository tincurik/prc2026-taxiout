# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke-test shim (loaded only with PYTHONPATH=smoke and EC_SMOKE_CAP=<n>): caps the number of boosting rounds of every XGBoost, CatBoost and
LightGBM fit at n, and the neural members' --epochs / --dae_epochs / --head_epochs at 3 and --hidden at 128, so the real stage scripts finish in minutes
on a tiny synthetic slice.  Nothing else changes; without EC_SMOKE_CAP it does nothing."""
import argparse
import functools
import importlib.abc
import importlib.machinery
import os
import sys

CAP = int(os.environ.get("EC_SMOKE_CAP", "0") or 0)


def _cap_kw(kw, names):
    for n in names:
        if n in kw and kw[n] is not None:
            kw[n] = min(int(kw[n]), CAP)


def _patch(name, mod):
    if name == "xgboost":
        f = mod.train

        @functools.wraps(f)
        def train(params, dtrain, *a, **kw):
            a = list(a)
            if a:
                a[0] = min(int(a[0]), CAP)
            else:
                kw["num_boost_round"] = min(int(kw.get("num_boost_round", 10)), CAP)
            return f(params, dtrain, *a, **kw)
        mod.train = train
        bp = mod.Booster.predict

        @functools.wraps(bp)
        def booster_predict(self, data, *a, iteration_range=(0, 0), **kw):
            if iteration_range and iteration_range[1]:
                iteration_range = (iteration_range[0], min(int(iteration_range[1]), self.num_boosted_rounds()))
            return bp(self, data, *a, iteration_range=iteration_range, **kw)
        mod.Booster.predict = booster_predict
    elif name == "lightgbm":
        f = mod.train

        @functools.wraps(f)
        def train(params, train_set, *a, **kw):
            a = list(a)
            if a:
                a[0] = min(int(a[0]), CAP)
            else:
                kw["num_boost_round"] = min(int(kw.get("num_boost_round", 100)), CAP)
            return f(params, train_set, *a, **kw)
        mod.train = train
    elif name == "catboost":
        base = mod.CatBoost
        init = base.__init__

        @functools.wraps(init)
        def __init__(self, params=None, *a, **kw):
            if isinstance(params, dict):
                params = dict(params)
                _cap_kw(params, ("iterations", "n_estimators", "num_boost_round", "num_trees"))
            _cap_kw(kw, ("iterations", "n_estimators", "num_boost_round", "num_trees"))
            init(self, params, *a, **kw)
        base.__init__ = __init__
        # the real code sizes its prediction range from the rounds the cross-validation fits chose (about 1.1 x the mean); with the rounds
        # capped the model can have fewer trees than that, so the range is cut to the trees there are
        pred = base.predict

        @functools.wraps(pred)
        def predict(self, data, *a, ntree_end=0, **kw):
            if ntree_end:
                ntree_end = min(int(ntree_end), self.tree_count_)
            return pred(self, data, *a, ntree_end=ntree_end, **kw)
        base.predict = predict
        for cls in (mod.CatBoostRegressor, mod.CatBoostClassifier):
            if "predict" in vars(cls):   # the sklearn-style classes define their own predict
                cp = cls.predict

                def make_p(cp):
                    @functools.wraps(cp)
                    def predict(self, data, *a, ntree_end=0, **kw):
                        if ntree_end:
                            ntree_end = min(int(ntree_end), self.tree_count_)
                        return cp(self, data, *a, ntree_end=ntree_end, **kw)
                    return predict
                cls.predict = make_p(cp)
            ci = cls.__init__

            def make(ci):
                @functools.wraps(ci)
                def __init__(self, *a, **kw):
                    _cap_kw(kw, ("iterations", "n_estimators", "num_boost_round", "num_trees"))
                    ci(self, *a, **kw)
                return __init__
            cls.__init__ = make(ci)


class _Finder(importlib.abc.MetaPathFinder):
    names = {"xgboost", "lightgbm", "catboost"}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.names or getattr(self, "busy", None) == fullname:
            return None
        self.busy = fullname
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            self.busy = None
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        orig = loader.exec_module

        def exec_module(module, _o=orig, _n=fullname):
            _o(module)
            _patch(_n, module)
        loader.exec_module = exec_module
        return spec


def _cap_args():
    parse = argparse.ArgumentParser.parse_args

    @functools.wraps(parse)
    def parse_args(self, *a, **kw):
        ns = parse(self, *a, **kw)
        for name, cap in (("epochs", 3), ("dae_epochs", 3), ("head_epochs", 3), ("hidden", 128)):
            if isinstance(getattr(ns, name, None), int):
                setattr(ns, name, min(getattr(ns, name), cap))
        return ns
    argparse.ArgumentParser.parse_args = parse_args


if CAP > 0:
    sys.meta_path.insert(0, _Finder())
    _cap_args()


# exit marker for smoke/bin/python (see there): written when the interpreter starts shutting down, holding the exit status the script had
_MARK = os.environ.get("PRC_EXIT_MARK")
if _MARK and str(os.getppid()) == os.environ.get("PRC_EXIT_PARENT"):
    import threading
    _status = ["0"]
    _hook = sys.excepthook

    def _excepthook(*a):
        _status[0] = "1"
        _hook(*a)
    sys.excepthook = _excepthook
    threading._register_atexit(lambda: open(_MARK, "w").write(_status[0]))
