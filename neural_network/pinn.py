"""
Данный скрипт содержит обучение PINN, которая подбирает параметры уравнения Дюффинга
по зашумленным данным измерения колебаний металлического консольного стержня с электромагнитным 
возбудителем, который возбуждает колебания свободного конца стержня
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

df = pd.read_csv("data_2.csv")
t_full = df["t"].values.astype(np.float64)
x_full = df["x"].values.astype(np.float64)

DECIMATION_FACTOR = 2
t_real = t_full[::DECIMATION_FACTOR]
x_real = x_full[::DECIMATION_FACTOR]
T_max = t_real[-1]

b, a = butter(4, 0.8, 'low')
x_smooth = filtfilt(b, a, x_real)
noise_est = np.std(x_real - x_smooth)
print(f"Оценка σ шума: {noise_est:.3f}")

# Тензоры для наблюдений

t_obs = torch.tensor(t_real, dtype=torch.float32).view(-1, 1).to(device)
x_obs = torch.tensor(x_real, dtype=torch.float32).view(-1, 1).to(device)

# Нейросеть с периодической активацией

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class PINN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), Sin(),
            nn.Linear(128, 128), Sin(),
            nn.Linear(128, 128), Sin(),
            nn.Linear(128, 128), Sin(),
            nn.Linear(128, 1)
        )
        # Инициализация весов
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)


log_c  = nn.Parameter(torch.tensor(np.log(np.sqrt(1e-4*30))))
log_k1 = nn.Parameter(torch.tensor(np.log(np.sqrt(1e-4*30))))
log_k3 = nn.Parameter(torch.tensor(np.log(np.sqrt(1e-4*30))))
log_A  = nn.Parameter(torch.tensor(np.log(np.sqrt(1e-4*30))))

m = 4.0
omega = 1.2
x0 = 0.01
v0 = 0.0

# Невязка

def physics_loss(model, t):
    t.requires_grad = True
    x = model(t)
    dx = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x),
                             create_graph=True)[0]
    d2x = torch.autograd.grad(dx, t, grad_outputs=torch.ones_like(dx),
                              create_graph=True)[0]

    c  = torch.exp(log_c)
    k1 = torch.exp(log_k1)
    k3 = torch.exp(log_k3)
    A  = torch.exp(log_A)

    residual = m*d2x + c*dx + k1*x + k3*x**3 - A*torch.cos(omega*t)
    return (residual**2).mean()


model = PINN_Net().to(device)

optimizer = torch.optim.AdamW(
    list(model.parameters()) + [log_c, log_k1, log_k3, log_A],
    lr=1e-2, weight_decay=1e-5
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

t_physics = torch.linspace(0, T_max, 800).view(-1, 1).to(device)
t_extra = torch.linspace(0, 0.5, 100).view(-1, 1).to(device)  # гуще в начале
t_physics = torch.cat([t_physics, t_extra])

data_weight = 1.0 / (noise_est**2) 
phys_weight = 0.5 
ic_weight = 10.0

epochs = 30000
best_loss = float('inf')
best_params = None

for epoch in range(epochs):
    optimizer.zero_grad()
    x_pred = model(t_obs)
    data_loss = ((x_pred - x_obs)**2).mean()
    phys_loss = physics_loss(model, t_physics)
    t0 = torch.tensor([[0.0]], device=device)
    t0.requires_grad = True
    x0_pred = model(t0)
    dx0 = torch.autograd.grad(x0_pred, t0, torch.ones_like(x0_pred), create_graph=True)[0]
    ic_loss = ((x0_pred - x0)**2 + (dx0 - v0)**2)

    loss = data_weight * data_loss + phys_weight * phys_loss + ic_weight * ic_loss

    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 500 == 0:
        c_val = torch.exp(log_c).item()
        k1_val = torch.exp(log_k1).item()
        k3_val = torch.exp(log_k3).item()
        A_val = torch.exp(log_A).item()
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4e} | "
              f"c={c_val:.4f}, k1={k1_val:.4f}, k3={k3_val:.4f}, A={A_val:.4f}")
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = (c_val, k1_val, k3_val, A_val)

print("\nНаилучшие параметры (c, k1, k3, A):")
print(best_params)

model.eval()
t_plot = torch.linspace(0, T_max, 500).view(-1,1).to(device)
x_fit = model(t_plot).detach().cpu().numpy().flatten()
t_np = t_plot.cpu().numpy().flatten()

plt.figure(figsize=(10,6))
plt.scatter(t_real, x_real, s=5, label="Данные (с шумом)")
plt.plot(t_np, x_fit, 'r', linewidth=2, label="PINN восстановление")
plt.xlabel("t"); plt.ylabel("x")
plt.legend(); plt.grid(True)
plt.title("Physics-Informed Neural Network (PyTorch)")
plt.show()

# Наилучшие параметры (c, k1, k3, A):
#(0.09637305528745775, 1.648909131658802, 25.229838245657355, 5.913263269211196)