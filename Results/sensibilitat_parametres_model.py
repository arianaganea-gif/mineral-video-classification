##############################################################################################################

# © 2026, Ariana Ganea
# Aquesta obra està subjecta a una llicència de Creative Commons Reconeixement-NoComercial-SenseObraDerivada
# 4.0 Internacional (CC BY-NC-ND 4.0). Veure : http://creativecommons.org/licenses/by-nc-nd/4.0/

##############################################################################################################

# CODI PER MIRAR LA SENSIBILITAT DE CADA PARAMETRE
# DURANT LA FASE D'EXPLORACIO DEL MODEL

##############################################################################################################

import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Puntuació F1': [69.0, 88.0, 95.8, 91.0, 30.0, 91.0, 95.0, 95.0, 95.6, 81.4, 89.3], # puntuacio F1
    'Resolució 280px': [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 1 es un si, 0 es un no
    'ResNet50': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], # 1 es un si, 0 es no
    'Augmentació de dades': [0, 1, 2, 4, 4, 4, 4, 4, 4, 4, 4], # quantitat de funcions per augmentar dades
    'Dropout': [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.15], # nivell de dropout

    # Algoritme de la segona capa : 0=LSTM/LSTM+Att, 1=MeanPooling, 2=Att(256), 3=Att(512)
    'Algorisme temporal': [0, 0, 1, 1, 3, 3, 3, 3, 3, 2, 3],

    # Features desglossades (1 = s'utilitza, 0 = no s'utilitza)
    'Característica PPL': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'Característica XPL': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'Característica |PPL-XPL|': [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],  # La fila 5 no té el valor absolut
    'Característica intensitat PPL ': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # La fila 8 té 'avg_intensity', ho compto com 0 aquí
    'Característica intensitat XPL': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    'Característica intensitat ΔPPL': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    'Característica intensitat ΔXPL': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    'Característica intensitat Δ²XPL': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
correlations = df.corr(method='spearman')[['Puntuació F1']].drop('Puntuació F1')
correlations = correlations.dropna()
correlations = correlations.sort_values(by='Puntuació F1', ascending=True)
fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
ax.set_facecolor('white')

# Colors: Blau per positiu, Vermell per negatiu
colors = ['#E06666' if x < 0 else '#4292C6' for x in correlations['Puntuació F1']]
bars = ax.barh(correlations.index, correlations['Puntuació F1'], color=colors,
               height=0.6, edgecolor='white', linewidth=1.5)
ax.set_title("Impacte de les Variables sobre l'F1-Score",
             fontsize=16, fontweight='bold', color='#333333', pad=25)
ax.set_xlabel("Correlació Spearman", fontsize=12, fontweight='bold', color='#555555', labelpad=15)
labels = [col if col != 'Dropout' else r'$\mathit{Dropout}$' for col in df.columns]

for bar in bars:
    width = bar.get_width()
    x_pos = width + 0.02 if width > 0 else width - 0.02
    ha = 'left' if width > 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.2f}',
            va='center', ha=ha, fontsize=11, fontweight='bold', color='#333333')

ax.set_xlim(-0.85, 0.85)
ax.axvline(0, color='#333333', linewidth=1.5)

for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(False)

ax.tick_params(axis='y', length=0, labelsize=11, colors='#333333')
ax.tick_params(axis='x', colors='#777777', labelsize=11)
ax.xaxis.grid(True, linestyle='--', which='major', color='#EEEEEE', alpha=1.0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()
