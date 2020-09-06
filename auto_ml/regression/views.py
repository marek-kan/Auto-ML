from django.shortcuts import render
import pandas as pd
from .models import AutoRegression

def regression(request):
    if request.method == 'POST':
        train = pd.read_csv(request.FILES.get('train_file'))
        train = train.reset_index()
        train.drop('index', axis=1, inplace=True)

        test = pd.read_csv(request.FILES.get('test_file'))
        test = test.reset_index()
        test.drop('index', axis=1, inplace=True)

        auto_reg = AutoRegression(train, test)
        auto_reg.run()

        results = {
            'feature_importances_plot': auto_reg.feature_imp_html,
            'regressin_line_plot': auto_reg.reg_line_html,
            'mae': auto_reg.mae_test.round(3),
            'r2': auto_reg.r2_test.round(4),
            'used_features': auto_reg.use_columns
        }
        return render(request, 'regression/results.html', results)
    else:
        return render(request, 'regression/regression.html')
