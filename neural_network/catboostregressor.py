'''
Представлена модель прогнозирования электропотребления
с помощью CatBoostRegressor с параметрами по умолчанию.
В коде подготовлены признаки и проверка качества данных 
Результат предсказания выведен графически, рассчитана MAPE 
и показан топ наиболее важных признаков
'''
#pip install catboost  # при необходимости
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error

# загрузка и очистка данных
def load_and_clean(path):
    for enc in ['utf-8', 'cp1251']:
        try:
            df = pd.read_csv(path, sep=';', decimal=',', encoding=enc)
            if df.shape[1] > 1: break
        except: continue
    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
    return df

df_input = load_and_clean('01 df_input.csv')
df_equip = load_and_clean('02 План по оборудованию.csv')

# обработка дат
df_input['Дата'] = pd.to_datetime(df_input['Дата'], dayfirst=True, errors='coerce')
df_input['Часы'] = pd.to_numeric(df_input['Часы'], errors='coerce')
df_input = df_input.dropna(subset=['Дата', 'Часы'])
df_input['datetime'] = df_input['Дата'] + pd.to_timedelta(df_input['Часы'], unit='h')

df_equip['Дата'] = pd.to_datetime(df_equip['Дата'], dayfirst=True, errors='coerce')
df_equip = df_equip.dropna(subset=['Дата'])
df_equip['Дата'] = df_equip['Дата'].dt.normalize()

# данные по оборудованию
eq_cols = [c for c in df_equip.columns if c.startswith('eq')]
for col in eq_cols:
    df_equip[col] = df_equip[col].astype(str).str.contains('в работе', case=False).astype(int)

# считаем сумму станков
df_equip['total_equip'] = df_equip[eq_cols].sum(axis=1)

# объединение
df = pd.merge(df_input, df_equip[['Дата', 'total_equip']], on='Дата', how='left')
df['total_equip'] = df['total_equip'].fillna(0)

# обработка факта и температуры
df['Факт, Мвт*ч'] = pd.to_numeric(df['Факт, Мвт*ч'], errors='coerce')
df['Плановая температура'] = pd.to_numeric(df['Плановая температура'], errors='coerce').interpolate().bfill()
df = df.dropna(subset=['Факт, Мвт*ч']).sort_values('datetime')

# набор признаков
def build_features(df):
    df = df.copy()

    df['hour'] = df['datetime'].dt.hour.astype(int)
    df['day_of_week'] = df['datetime'].dt.dayofweek.astype(int)
    df['month'] = df['datetime'].dt.month.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df['lag_24h'] = df['Факт, Мвт*ч'].shift(24)
    df['lag_168h'] = df['Факт, Мвт*ч'].shift(168)

    df['temp_hour'] = df['Плановая температура'] * df['hour']
    df['temp_rolling_12h'] = df['Плановая температура'].rolling(12).mean()

    return df.dropna(subset=['lag_24h']).reset_index(drop=True)

df_feat = build_features(df)

split = int(len(df_feat) * 0.8)
train, test = df_feat.iloc[:split], df_feat.iloc[split:]

# исключаем total_equip из обучения, так как он "отравляет" модель и повышает MAPE - данный признак был признан шумовым
cat_features = ['hour', 'day_of_week', 'month', 'is_weekend']
features = cat_features + ['Плановая температура', 'temp_hour', 'temp_rolling_12h', 'lag_24h', 'lag_168h']

X_train, y_train = train[features], train['Факт, Мвт*ч']
X_test, y_test = test[features], test['Факт, Мвт*ч']

print(f"Обучение на {len(X_train)} строках...")
model = CatBoostRegressor(iterations=1000,
                          random_state=42,
                          verbose=0, cat_features=cat_features)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("-" * 30)
print(f"Итоговая MAPE: {mape:.2%}")
print("-" * 30)

imp = pd.DataFrame({'признак': features, 'важность': model.feature_importances_}).sort_values('важность', ascending=False)
print("Топ признаков:")
print(imp.to_string(index=False))

plt.figure(figsize=(15, 5))
plt.plot(test['datetime'], y_test, label='Факт', color='steelblue')
plt.plot(test['datetime'], y_pred, label='Прогноз', ls='--', color='darkorange')
plt.title(f'Электропотребление (MAPE: {mape:.2%})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
