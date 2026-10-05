import numpy as np, pandas as pd
rng = np.random.default_rng(42)
# Gaussiano 3 atributos, 4 classes (n < 100 e n > 100)
means = {'F': [0.20, 0.05, 0.30], 'VS': [0.22, 0.07, 0.28], 'PA': [0.10, 0.15, 0.20], 'SE': [0.05, 0.25, 0.10]}
ns = {'F': 80, 'VS': 60, 'PA': 150, 'SE': 45}
rows = []
for c, m in means.items():
    A = rng.normal(0, 0.02, (3, 3)); cov = A @ A.T + np.eye(3) * 1e-4
    x = rng.multivariate_normal(m, cov, ns[c])
    rows.append(pd.DataFrame(x, columns=['B1', 'B2', 'B3']).assign(Class=c))
g = pd.concat(rows, ignore_index=True)[['Class', 'B1', 'B2', 'B3']]
g.to_csv('gauss.csv', index=False)

# Gama univariada (intensidade com L = 3 looks)
L = 3.0
mus = {'F': 0.08, 'VS': 0.06, 'PA': 0.03, 'SE': 0.015}
nsg = {'F': 50, 'VS': 120, 'PA': 200, 'SE': 70}
rows = [pd.DataFrame({'Class': c, 'HH': rng.gamma(L, mu / L, nsg[c])}) for c, mu in mus.items()]
pd.concat(rows, ignore_index=True).to_csv('gamma.csv', index=False)

# Par de intensidades correlacionadas (speckle multilook, L = 4)
def pair(n, mu1, mu2, rho, Lk):
    z1 = (rng.normal(size=(n, Lk)) + 1j * rng.normal(size=(n, Lk))) / np.sqrt(2)
    w = (rng.normal(size=(n, Lk)) + 1j * rng.normal(size=(n, Lk))) / np.sqrt(2)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * w
    return mu1 * np.mean(abs(z1) ** 2, 1), mu2 * np.mean(abs(z2) ** 2, 1)
rows = []
for c, (m1, m2, r) in {'F': (0.08, 0.02, 0.5), 'PA': (0.03, 0.006, 0.3), 'SE': (0.015, 0.004, 0.7)}.items():
    i1, i2 = pair(90, m1, m2, r, 4)
    rows.append(pd.DataFrame({'Class': c, 'HH': i1, 'HV': i2}))
pd.concat(rows, ignore_index=True).to_csv('pair.csv', index=False)

# Divisão fixa treino/teste (para comparar uma iteração da validação sem depender do RNG)
for name in ['gauss', 'gamma', 'pair']:
    d = pd.read_csv(f'{name}.csv')
    is_train = np.zeros(len(d), bool)
    for c, grp in d.groupby('Class'):
        idx = rng.choice(grp.index, int(np.ceil(0.7 * len(grp))), replace=False)
        is_train[idx] = True
    d[is_train].to_csv(f'{name}_train.csv', index=False)
    d[~is_train].to_csv(f'{name}_test.csv', index=False)

# Matriz de confusão (linhas = Prediction, colunas = Reference), com uma linha só prevista
cm = pd.DataFrame([[40, 5, 2, 0], [8, 30, 4, 1], [1, 3, 50, 6], [0, 1, 2, 20]],
                  index=['F', 'PA', 'SE', 'VS'], columns=['F', 'PA', 'SE', 'VS'])
cm.to_csv('cm.csv')
print('ok')
