from django.shortcuts import render
import pandas as pd
from .models import AutoClassification

from multiprocessing import Process, Manager

def helper(train, test, settings, return_objects):
    auto_class = AutoClassification(train, test, settings)
    auto_class.run()
    results = {
        'best_model': auto_class.models[auto_class.best_model],
        'feature_importances_plot': auto_class.feature_imp_html,
        'decision_line_plot': auto_class.dec_line_html,
        'acc': auto_class.acc.round(4),
        'auc': auto_class.auc.round(4),
        'used_features': auto_class.use_columns
    }
    return_objects['result'] = results
    return auto_class

def classification(request):
    if request.method == 'POST':
        train = pd.read_csv(request.FILES.get('train_file'))
        train = train.reset_index()
        train.drop('index', axis=1, inplace=True)

        test = pd.read_csv(request.FILES.get('test_file'))
        test = test.reset_index()
        test.drop('index', axis=1, inplace=True)

        m = Manager()
        return_objects = m.dict()
        p = Process(target=helper, args=(train, test, request.POST, return_objects))
        p.start()
        p.join()
        return render(request, 'classification/results.html', return_objects['result'])
    else:
        d = {'page': 'classification'}
        return render(request, 'classification/classification.html', d)
