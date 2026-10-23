# Notebook: California house price prediction
Self-contained: `california_house_prices.ipynb` reads `data/california_housing.csv` from this folder.
```bash
pip install -r requirements.txt
jupyter notebook california_house_prices.ipynb
```
It writes `cv_results.json`. Data: Pace & Barry (1997), StatLib, via scikit-learn. Run time is about 5–10 minutes on 8 cores.
