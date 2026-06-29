"""
Анализ данных с изокинетического динамометра.
Задача: выделить фазы сгибания и разгибания по отрицательным значениям скорости,
найти максимальный момент (torque) в каждом сегменте и построить графики.

Алгоритм:
1. Загружаем Excel-файл с колонками: time, grad, torque, speed.
2. Находим строки, где speed < 0 (отрицательная скорость).
3. Разбиваем данные на непрерывные блоки между отрицательными зонами.
4. Для каждого блока вычисляем максимум torque и его индекс.
5. Строим график torque от угла (grad) с выделением найденных максимумов.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

excel_data = pd.read_excel('R_60.xlsx')
excel_data1 = pd.read_excel('L_60.xlsx')

data = pd.DataFrame(excel_data, columns=['time', 'grad', 'torque', 'speed'])
data1 = pd.DataFrame(excel_data1, columns=['time', 'grad', 'torque', 'speed'])

print(f"Загружено {len(data)} строк для правой, {len(data1)} для левой.")

row_indices = np.where(data['speed'] < 0)[0]
row_indices1 = np.where(data1['speed'] < 0)[0]

max_list = []
max_list1 = []
bound = []         
bound1 = []        
maxval_list = []   
maxval1_list = []

t = 0
s = 0
i = 0

for row_idx in row_indices:
    if row_idx - s != 1:   # если отрицательная строка не идёт подряд
        m = row_idx - s
        # Интервал для правой: от t+2 до s+3 (смещения для захвата переходной зоны)
        g1 = t + 2
        g2 = s + 3
        # Интервал для левой (аналогичный по времени, но свои индексы)
        g3 = s + 2
        g4 = s + m + 3

        bound.append(g1)
        bound.append(g2)
        bound1.append(g3)
        bound1.append(g4)

        # Вырезаем сегменты
        interv = data.iloc[g1:g2]
        maximum = max(interv['torque'])
        interv1 = data1.iloc[g3:g4]
        maximum1 = max(interv1['torque'])

        max_list.append(maximum)
        max_list1.append(maximum1)

        # Индекс (позиция) максимума в DataFrame
        maxval = interv['torque'].idxmax()
        maxval1 = interv1['torque'].idxmax()
        maxval_list.append(maxval)
        maxval1_list.append(maxval1)

        t = s + m
    s = row_idx

# Обработка последнего интервала после последней отрицательной строки
m = row_idx - s
g1 = t + 2
g2 = s + 3
g3 = s + 2
g4 = len(data)

bound.append(g1)
bound.append(g2)
bound1.append(g3)
bound1.append(g4)

interv = data.iloc[t+2:s+3]
maximum = max(interv['torque'])
max_list.append(maximum)
interv1 = data1.iloc[s+2:]
maximum1 = max(interv1['torque'])
max_list1.append(maximum1)

maxval = interv['torque'].idxmax()
maxval1 = interv1['torque'].idxmax()
maxval_list.append(maxval)
maxval1_list.append(maxval1)

t = s + m

idx_max = max_list.index(max(max_list))

a = min(bound, key=lambda x: abs(x - maxval_list[idx_max]))
b = bound[bound.index(a) + 1]
c = min(bound1, key=lambda x: abs(x - maxval1_list[idx_max]))
d = bound1[bound1.index(c) + 1]

# Вырезаем выбранные сегменты для построения
df1 = data.iloc[a:b]
df2 = data1.iloc[c:d]

plt.figure(figsize=(10, 6))

# График разгибателей
plt.plot(df1['grad'], df1['torque'], label='Правая (разгибатели)')
plt.plot(df2['grad'], df2['torque'], label='Левая (разгибатели)')
plt.gca().invert_xaxis()  # так как угол обычно убывает
plt.xlabel("Угол (град)")
plt.ylabel("Момент (Н·м)")
plt.title("Разгибатели (60°/с)")
plt.legend()
plt.grid(True)
plt.text(0.5, 0.95, f"Правая max: {max(df1['torque']):.2f}, Левая max: {max(df2['torque']):.2f}",
         transform=plt.gca().transAxes, ha='center', va='top', bbox=dict(boxstyle="round", facecolor="wheat"))

plt.show()

print(f"Найдено интервалов: {len(max_list)}")
print("Максимумы torque (правая):", [round(x, 2) for x in max_list])
print("Максимумы torque (левая):", [round(x, 2) for x in max_list1])
print(f"\nВыбран интервал {idx_max+1} (с наибольшим максимумом правой).")
print(f"Правая: максимум = {max(df1['torque']):.2f} Н·м при угле {df1[df1['torque'] == max(df1['torque'])]['grad'].values[0]:.1f}°")
print(f"Левая:  максимум = {max(df2['torque']):.2f} Н·м при угле {df2[df2['torque'] == max(df2['torque'])]['grad'].values[0]:.1f}°")
print(f"Латеральная асимметрия (прав/лев): {(1 - max(df2['torque'])/max(df1['torque']))*100:.2f}%")
