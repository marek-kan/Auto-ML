from django.shortcuts import render
import pandas as pd

def regression(request):
    # TODO reset indexes after reading a files
    if request.method == 'POST':
        return render(request, 'regression/results.html')
    else:
        return render(request, 'regression/regression.html')
