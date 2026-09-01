"""1주차 오리엔테이션 — 15주 로드맵 그림 (fig_roadmap.png)"""
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.size': 14, 'axes.labelsize': 14, 'legend.fontsize': 13,
    'font.family': 'Pretendard',
    'axes.unicode_minus': False,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# (시작 주, 폭, 회색, 블록 라벨, 라벨 회전)
blocks = [
    (1,  2, '0.92', '데이터·환경',  0),
    (3,  3, '0.78', 'ML',          0),
    (6,  2, '0.60', 'DL',          0),
    (8,  1, '1.00', '휴강',        90),
    (9,  1, '0.25', '중간고사',     90),
    (10, 1, '0.60', 'PyTorch',     90),
    (11, 2, '0.52', 'Transformer',  0),
    (13, 2, '0.38', 'ETA 프로젝트',  0),
    (15, 1, '0.12', '발표',        90),
]
# 블록 아래 교재·주제 주석: (중심 주, 텍스트)
notes = [
    (4.0,  '핸즈온 머신러닝 1권'),
    (6.5,  '밑바닥 딥러닝 1'),
    (11.5, '강의노트 · 시계열'),
    (13.5, '서울시 BIS'),
]

fig, ax = plt.subplots(figsize=(11.8, 4.7))
y0, h = 0.42, 0.34
for start, width, gray, label, rot in blocks:
    hatch = '///' if label == '휴강' else None
    ax.broken_barh([(start - 0.5, width)], (y0, h), facecolors=gray,
                   edgecolor='black', linewidth=1.2, hatch=hatch)
    tcolor = 'white' if float(gray) < 0.5 else 'black'
    fs = 15 if rot == 0 else 13
    bbox = (dict(facecolor='white', edgecolor='none', pad=1.5)
            if hatch else None)
    ax.text(start - 0.5 + width / 2, y0 + h / 2, label, ha='center',
            va='center', color=tcolor, rotation=rot, fontsize=fs,
            fontweight='bold' if rot == 0 else 'normal', bbox=bbox)
for cx, txt in notes:
    ax.text(cx, y0 - 0.10, txt, ha='center', va='top', fontsize=13, color='0.35')

for w in range(1, 16):
    ax.text(w, y0 + h + 0.06, str(w), ha='center', va='bottom',
            fontsize=13, color='0.45')
ax.text(0.35, y0 + h + 0.06, '주차', ha='right', va='bottom',
        fontsize=13, color='0.45')

ax.set_xlim(0.0, 15.7)
ax.set_ylim(0, 1)
ax.axis('off')
fig.tight_layout()
fig.savefig('fig_roadmap.png', dpi=200, bbox_inches='tight')
print('saved fig_roadmap.png')
