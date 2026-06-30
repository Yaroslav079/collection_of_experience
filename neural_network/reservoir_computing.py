"""
Данный скрипт на основе резервуарных вычислений подбирает параметры биологической 
модели (ОДУ) по наблюдаемым экспериментальным данным о росте клеток.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge

class ReservoirComputingModel:
    def __init__(self, input_dim, reservoir_size, output_dim, spectral_radius=0.9, leakage=0.2):
        self.reservoir_size = reservoir_size
        self.leakage = leakage
        np.random.seed(42)
        self.Win = np.random.uniform(-1, 1, (reservoir_size, input_dim))
        W = np.random.uniform(-1, 1, (reservoir_size, reservoir_size))
        eigenvalues = np.linalg.eigvals(W)
        self.Wres = W * (spectral_radius / np.max(np.abs(eigenvalues)))
        self.readout = Ridge(alpha=0.1)

    def _get_states(self, X_series):
        n_samples = X_series.shape[0]
        time_steps = X_series.shape[1]
        states = np.zeros((n_samples, self.reservoir_size))
        for i in range(n_samples):
            x_t = np.zeros((self.reservoir_size, 1))
            series = X_series[i]
            for t in range(time_steps):
                u = series[t].reshape(-1, 1)
                pre_activation = np.dot(self.Win, u) + np.dot(self.Wres, x_t)
                x_t = (1 - self.leakage) * x_t + self.leakage * np.tanh(pre_activation)
            states[i] = x_t.flatten()
        return states

    def train(self, X_train, Y_train):
        reservoir_states = self._get_states(X_train)
        self.readout.fit(reservoir_states, Y_train)

    def predict(self, X_test):
        reservoir_states = self._get_states(X_test)
        return self.readout.predict(reservoir_states)

def heterogeneous_model(t, y, alpha, beta, delta, J):
    N = y[:J + 1]
    D = y[-1]
    dNdt = np.zeros(J + 1)
    N = np.maximum(N, 0)
    
    dNdt[0] = -(alpha[0] + beta[0]) * N[0]
    for j in range(1, J + 1):
        dNdt[j] = 2 * alpha[j - 1] * N[j - 1] - (alpha[j] + beta[j]) * N[j]

    dDdt = np.sum([beta[j] * N[j] for j in range(J + 1)]) - delta * D
    return np.hstack([dNdt, dDdt])

# --- Генератор данных ---
def generate_synthetic_data(n_samples, initial_conditions, time_points, J, bounds):
    X_data = []
    Y_data = []
    print(f"Генерация {n_samples} примеров...")
    
    cnt = 0
    while cnt < n_samples:
        params = []
        for b in bounds:
            exponent = np.random.uniform(np.log10(b[0]), np.log10(b[1]))
            val = 10**exponent
            params.append(val)

                
        params = np.array(params)
        alpha = params[:J + 1]
        beta = params[J+1:2*(J+1)]
        delta = params[-1]
        
        try:
            solution = solve_ivp(
                lambda t, y: heterogeneous_model(t, y, alpha, beta, delta, J),
                [72, 168], initial_conditions, t_eval=time_points, method='Radau'
            )
            if solution.status == 0 and solution.y.shape[1] == len(time_points):
                data = np.log1p(solution.y.T)
                # Проверка на NaN
                if not np.isnan(data).any():
                    X_data.append(data)
                    Y_data.append(params)
                    cnt += 1
        except:
            pass

    return np.array(X_data), np.array(Y_data)

def step2_rc():
    res_ref_str = ['1.31357284e-02', '3.09893156e-02', '5.21147849e-02', '4.95095089e-02',
                   '2.94412900e-02', '7.28078976e-03', '2.26816027e-02', '1.37996127e+00',
                   '1.00000000e-15', '1.00000000e-15', '1.00000000e-15', '1.00000000e-15',
                   '7.12593794e-03', '2.68532688e-02', '1.00000000e-15', '1.00000000e-15',
                   '4.51715405e-02']
    res_ref = np.array(res_ref_str, dtype=float)

    time_points = [72, 96, 120, 144, 168]
    total_dead = [1.6e4, 2.4e4, 6.0e4, 1.2e5, 1.3e5]
    N_j_data = [
        [29358, 22876, 43372, 39970, 5208, 98, 14, 0],
        [16050, 12600, 22650, 57025, 96350, 46950, 2500, 25],
        [14476, 14784, 25344, 58652, 141460, 156290, 32076, 440],
        [13500, 12150, 24150, 55000, 137850, 188950, 69450, 2150],
        [13509, 12198, 21603, 51927, 140560, 232160, 96102, 3420]
    ]
    J = 7
    initial_conditions = N_j_data[0] + [total_dead[0]]
    identification_time_points = time_points[1:]

    param_bounds = []
    
    # Alpha (0-7)
    for i in range(J+1):
        if i == J:
            param_bounds.append((1, 2))
        else:
            param_bounds.append((1e-3, 0.1))
    for i in range(J+1):
        param_bounds.append((1e-3, 0.1))
    param_bounds.append((1e-3, 0.1))

    NUM_SAMPLES = 10000
    X_train, Y_train = generate_synthetic_data(
        NUM_SAMPLES, initial_conditions, identification_time_points, J, param_bounds
    )
    
    rc_model = ReservoirComputingModel(input_dim=J+2, reservoir_size=1000, output_dim=17)
    rc_model.train(X_train, Y_train)

    # 4. Предсказание на реальных данных
    real_data_input = []
    for i in range(1, len(time_points)):
        real_data_input.append(N_j_data[i] + [total_dead[i]])
    real_data_input = np.log1p(np.array([real_data_input])) # Batch dim added

    predicted_params = rc_model.predict(real_data_input)[0]
    predicted_params = np.maximum(predicted_params, 1e-15) # Убираем отрицательные

    # 5. Вывод сравнения в терминал
    print("\n" + "="*65)
    print(f"{'Параметр':<10} | {'Референс':<15} | {'Предсказание':<15} | {'Разность':<10}")
    print("-" * 65)
    
    param_names = [f"Alpha_{i}" for i in range(J+1)] + [f"Beta_{i}" for i in range(J+1)] + ["Delta"]
    
    for i in range(len(param_names)):
        ref_val = res_ref[i]
        pred_val = predicted_params[i]
        diff = pred_val - ref_val
        print(f"{param_names[i]:<10} | {ref_val:15.8f} | {pred_val:15.8f} | {diff:10.2f}")
    print("="*65)

    t_span = [time_points[0], time_points[-1]]
    t_plot = np.linspace(time_points[0], time_points[-1], 100)

    sol_pred = solve_ivp(
        lambda t, y: heterogeneous_model(t, y, predicted_params[:J+1], predicted_params[J+1:2*J+2], predicted_params[-1], J),
        t_span, initial_conditions, dense_output=True
    )
    y_pred = sol_pred.sol(t_plot)

    sol_ref = solve_ivp(
        lambda t, y: heterogeneous_model(t, y, res_ref[:J+1], res_ref[J+1:2*J+2], res_ref[-1], J),
        t_span, initial_conditions, dense_output=True
    )
    y_ref = sol_ref.sol(t_plot)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for j in range(J + 1):
        ax = axes[j]
        
        ax.plot(t_plot, y_ref[j, :], 'g--', linewidth=2, label='Референс')
        ax.plot(t_plot, y_pred[j, :], 'r-', linewidth=2, alpha=0.8, label='Нейросеть')
        exp_values = [data[j] for data in N_j_data]
        ax.scatter(time_points, exp_values, s=40, facecolor='w', edgecolor='k', zorder=5, label='Эксперимент')

        ax.set_title(f'Поколение $N_{j}$')
        ax.set_xlabel('Часы')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = axes[8]
    ax.plot(t_plot, y_ref[J+1, :], 'g--', linewidth=2, label='Референс')
    ax.plot(t_plot, y_pred[J+1, :], 'r-', linewidth=2, alpha=0.8, label='Нейросеть')
    ax.scatter(time_points, total_dead, s=40, facecolor='w', edgecolor='k', zorder=5, label='Эксперимент')
    ax.set_title('Мертвые клетки (Dead)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Сравнение: Нейросеть vs Референсные параметры", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    step2_rc()
