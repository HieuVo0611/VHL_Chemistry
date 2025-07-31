import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

plt.style.use('seaborn-v0_8-whitegrid')

folder = r"e:\VHL Project\Chemistry\VHL_Chemistrygit\VHL_Chemistrygit\data\raw\1053 so lieu Cam bien dien hoa(HP4)\1053 so lieu phan tich Cd, Pb"

pb_concentrations = []
cd_concentrations = []

# Hỗ trợ cả số thập phân
pattern = re.compile(r'Pb([\d\.]+)_Cd([\d\.]+)')

for filename in os.listdir(folder):
    match = pattern.search(filename)
    if match:
        pb = float(match.group(1))
        cd = float(match.group(2))
        pb_concentrations.append(pb)
        cd_concentrations.append(cd)

def plot_concentration(concs, label, color, filename):
    counter = Counter(concs)
    all_conc = sorted(counter.keys())
    counts = [counter[c] for c in all_conc]
    x = np.arange(len(all_conc))
    width = 0.7

    fig, ax = plt.subplots(figsize=(max(10, len(all_conc)*0.5), 6))
    bars = ax.bar(x, counts, width=width, color=color, alpha=0.85, edgecolor='black')
    for i, rect in enumerate(bars):
        if counts[i] > 0:
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height()+0.5, f'{counts[i]}', 
                    ha='center', va='bottom', color=color, fontsize=11, fontweight='bold')
    ax.set_xlabel(f'Nồng độ {label} (ppm)', fontsize=13)
    ax.set_ylabel('Số lượng mẫu', fontsize=13)
    ax.set_title(f'Phân phối nồng độ {label}', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    # Hiển thị số thập phân nếu cần
    xtick_labels = [f'{c:.2f}' if c < 1 else (f'{c:.1f}' if c % 1 != 0 else f'{int(c)}') for c in all_conc]
    ax.set_xticklabels(xtick_labels, rotation=45, fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    # Giới hạn trục Y vừa đủ
    ax.set_ylim(0, max(counts)*1.15 if counts else 1)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

plot_concentration(pb_concentrations, 'Pb', '#1976D2', 'dist_pb.png')
plot_concentration(cd_concentrations, 'Cd', '#D32F2F', 'dist_cd.png')