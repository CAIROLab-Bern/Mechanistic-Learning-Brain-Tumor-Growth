import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATIENT_MAP = {
    'PIDz035': 1,
    'PIDz057': 2,
    'PIDz069': 3, 
    'PIDz074': 4, 
    'PIDz077': 5, 
    'PIDz140': 6, 
    'PIDz161': 7, 
    'PIDz224': 8
}


folder = 'results_dmg_fit_goodinit'

pats =   [
    'PIDz035',
    'PIDz057',
    'PIDz069',
    # 'PIDz074',
    # 'PIDz077',
    # 'PIDz140',  
    # # 'PIDz160',
    # 'PIDz161',
    # 'PIDz224',
    # # 'PIDz254',
    # # 'PIDz279',
]

slices = ['initial','largest']

df_perf = pd.DataFrame(columns = ['pats','slice','R2all','R2train','RMSE','nRMSE'])
for pat in pats:
    for s in slices:
        for i in range(0,5):
            try:
                name = pat+'_'+s+'_'+str(i)+'_dual_mode_summary.csv'
                df_pat = pd.read_csv(os.path.join(folder,pat,name))
                df_perf.loc[len(df_perf)] = [
                    f"PID{PATIENT_MAP[pat]}",
                    s+str(i),
                    df_pat['all_r2'].iloc[0],
                    df_pat['train_train_r2'].iloc[0],
                    df_pat['train_test_rmse'].iloc[0],
                    df_pat['train_test_rmse'].iloc[0]/df_pat['train_ground_truth'].iloc[0]
                ]
            except:
                continue

# Define patient order based on PID numbers (PID1, PID2, ..., PID8)
patient_order = [f"PID{i}" for i in range(1, 9) if f"PID{i}" in df_perf['pats'].unique()]

df_melted = df_perf.melt(id_vars='pats', value_vars=['R2all', 'R2train'],
                         var_name='Metric', value_name='Value')

# Map old names to pretty labels
metric_rename = {
    'R2all': r'R$^2$ all Data',
    'R2train': r'R$^2$ train Data'
}

# Apply renaming
df_melted_plot = df_melted.copy()
df_melted_plot['Metric'] = df_melted_plot['Metric'].map(metric_rename)

# Define custom colors
box_colors = {
    r'R$^2$ all Data': 'grey',
    r'R$^2$ train Data': 'lightcoral'
}

plt.figure(figsize=(6, 4))

# Boxplot with ordered x-axis
sns.boxplot(
    data=df_melted_plot,
    x='pats',
    y='Value',
    hue='Metric',
    width=0.6,
    fliersize=0,
    palette=box_colors,
    order=patient_order
)

# Dots with black edge
sns.stripplot(
    data=df_melted_plot,
    x='pats',
    y='Value',
    hue='Metric',
    dodge=True,
    jitter=True,
    alpha=0.7,
    marker='o',
    size=6,
    edgecolor='black',
    linewidth=0.5,
    palette=box_colors,
    order=patient_order
)

# Labels + limits
plt.ylabel('Value', fontsize=14)
plt.xlabel('Patient', fontsize=14)
plt.ylim([0, 1])

# Ticks
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Clean up legend (remove duplicate)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], title='Metric', loc='lower left', fontsize=12, title_fontsize=12)

plt.tight_layout()
plt.savefig('performance.png', dpi=300)
plt.savefig('performance.pdf')
plt.show()

# Second figure - larger size and smaller legend
# Map old names to pretty labels
metric_rename = {
    'R2all': r'R$^2$ all Data',
    'R2train': r'R$^2$ train Data'
}

# Prepare R² data
df_r2 = df_melted.copy()
df_r2['Metric'] = df_r2['Metric'].map(metric_rename)

# Prepare nRMSE data
df_nrmse = df_perf[['pats', 'nRMSE']].copy()
df_nrmse = df_nrmse.melt(id_vars='pats', var_name='Metric', value_name='Value')
df_nrmse['Metric'] = 'nRMSE'

# Use the same patient order as first plot
# Plot with larger figure size
# fig, ax1 = plt.subplots(figsize=(9, 5))  # Made larger
fig, ax1 = plt.subplots(figsize=(7, 3))

# R² colors
r2_colors = {
    r'R$^2$ all Data': 'grey',
    r'R$^2$ train Data': 'lightcoral'
}

# nRMSE color
nrmse_color = '#4682B4'  # steel blue

# R² boxplot with patient order
sns.boxplot(
    data=df_r2,
    x='pats',
    y='Value',
    hue='Metric',
    width=0.6,
    fliersize=0,
    palette=r2_colors,
    order=patient_order,
    ax=ax1
)

# R² dots
sns.stripplot(
    data=df_r2,
    x='pats',
    y='Value',
    hue='Metric',
    dodge=True,
    jitter=True,
    alpha=0.7,
    marker='o',
    size=6,
    edgecolor='black',
    linewidth=0.5,
    palette=r2_colors,
    order=patient_order,
    ax=ax1,
    legend=False
)

ax1.set_ylabel('R² Value', fontsize=14)
ax1.set_ylim([0, 1])
ax1.set_xlabel('')
ax1.tick_params(axis='y', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

# Secondary y-axis for nRMSE
ax2 = ax1.twinx()

# nRMSE boxplot
sns.boxplot(
    data=df_nrmse,
    x='pats',
    y='Value',
    width=0.4,
    fliersize=0,
    color=nrmse_color,
    boxprops=dict(alpha=0.3),
    order=patient_order,
    ax=ax2
)

# nRMSE dots
sns.stripplot(
    data=df_nrmse,
    x='pats',
    y='Value',
    dodge=False,
    jitter=True,
    alpha=0.6,
    marker='D',
    size=5,
    edgecolor='black',
    linewidth=0.5,
    color=nrmse_color,
    order=patient_order,
    ax=ax2
)

ax2.set_ylabel('nRMSE', fontsize=14, color=nrmse_color)
ax2.tick_params(axis='y', labelsize=12, colors=nrmse_color)
ax2.spines['right'].set_color(nrmse_color)

# Build combined legend manually with smaller font
r2_handles, r2_labels = ax1.get_legend_handles_labels()
nrmse_handle = plt.Line2D([0], [0], marker='D', color='w',
                          markerfacecolor=nrmse_color, markeredgecolor='black',
                          markersize=5, linestyle='None', label='nRMSE')  # Smaller marker

ax1.legend(handles=r2_handles[:2] + [nrmse_handle],
           title='Metric', fontsize=10, title_fontsize=10,  # Smaller font sizes
           bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.savefig('nRMSE_performance_all.png', dpi=300)
plt.savefig('nRMSE_performance_all.pdf')
plt.show()



# Calculate median nRMSE across all slices and patients
overall_median_nrmse = df_perf['nRMSE'].median()
print(f"Median nRMSE across all slices and patients: {overall_median_nrmse:.4f}")

# Calculate median nRMSE per patient
median_nrmse_per_patient = df_perf.groupby('pats')['nRMSE'].median().sort_index()
print("\nMedian nRMSE per patient:")
for patient, median_nrmse in median_nrmse_per_patient.items():
    print(f"  {patient}: {median_nrmse:.4f}")

# Additional statistics you might find useful
print(f"\nAdditional nRMSE statistics:")
print(f"  Mean: {df_perf['nRMSE'].mean():.4f}")
print(f"  Std: {df_perf['nRMSE'].std():.4f}")
print(f"  Min: {df_perf['nRMSE'].min():.4f}")
print(f"  Max: {df_perf['nRMSE'].max():.4f}")
print(f"  25th percentile: {df_perf['nRMSE'].quantile(0.25):.4f}")
print(f"  75th percentile: {df_perf['nRMSE'].quantile(0.75):.4f}")
