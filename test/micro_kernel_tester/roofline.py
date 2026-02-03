import matplotlib.pyplot as plt
import numpy as np
import csv

# 1. 硬件参数
peak_performance = 83.2  # GFLOPS
peak_bandwidth = 23.05    # GB/s
ridge_point = peak_performance / peak_bandwidth  # 拐点 (FLOPS/Byte)

# 2. 准备绘图数据
# 计算强度轴 (log scale)
intensity = np.logspace(-2, 8, num=500, base=2)
# Roofline 公式实现
theoretical_perf = np.minimum(peak_performance, peak_bandwidth * intensity)


bytes_dtype = {
    "fp32":4
}
results = []
with open('random_res.csv', 'r', encoding='utf-8') as f:
    # DictReader 会自动将第一行作为字典的 Key
    reader = csv.DictReader(f)
    for row in reader:
        results.append(dict(row))

gemm_intensities = []
gemm_perf = []
for line in results:
    m,n,k,dt,pf = int(line['M']), int(line['N']), int(line['K']), bytes_dtype[line['dtype']], float(line['perf'])
    gemm_intensities.append(2*m*n*k/(m*n+n*k+m*k)/dt)
    gemm_perf.append(pf)


# 4. 绘图
plt.figure(figsize=(10, 6))
plt.loglog(intensity, theoretical_perf, 'r--', linewidth=2, label='Roofline')

text_x_bw = 0.5 
text_y_bw = peak_bandwidth * text_x_bw
plt.text(text_x_bw, text_y_bw * 0.55, f'Memory Bandwidth: {peak_bandwidth} GB/s', 
         rotation=42, # 在log-log图中，斜率为1的线旋转角度约为35-40度，具体取决于图表长宽比
         color='red', fontweight='bold', fontsize=11, ha='center')

# 在水平段（Compute Bound）添加文字
plt.text(32, peak_performance * 1.1, f'Compute Peak: {peak_performance} GFLOPS', 
         color='red', fontweight='bold', fontsize=11, ha='center')

# 填充区域
plt.fill_between(intensity, 0.1, theoretical_perf, color='red', alpha=0.1)

# 绘制 GEMM 散点
plt.scatter(gemm_intensities, gemm_perf, color='blue', zorder=5)



# 图表设置
plt.title('autoGEMM microkernel roofline model on 920G', fontsize=14)
plt.xlabel('Operational Intensity (FLOPS/Byte)', fontsize=12)
plt.ylabel('Performance (GFLOPS)', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(loc='upper left')
plt.ylim(bottom=1) # 设置纵坐标起点

plt.tight_layout()
plt.savefig("roofline.png",dpi=200)
