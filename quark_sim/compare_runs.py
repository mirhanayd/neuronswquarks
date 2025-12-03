import json

d1 = json.load(open('outputs/20251203_035245_GMT/session.json'))
d2 = json.load(open('outputs/20251203_042707_GMT/session.json'))

print("=" * 80)
print("KARSILASTIRMA: ESKI vs YENI")
print("=" * 80)

print("\nESKI (256-128-64, lr=0.008, 12000 epoch):")
print(f"  Son kayip: {d1['loss_history'][-1][1]:.6f} GeV^2")
errors1 = [abs(c-n)/abs(c)*100 for c,n in zip(d1['cornell_values'], d1['nn_values'])]
print(f"  Ortalama hata: {sum(errors1)/len(errors1):.2f}%")
print(f"  Max hata: {max(errors1):.2f}%")

print("\nYENI (128-64-32, lr=0.02, 8000 epoch):")
print(f"  Son kayip: {d2['loss_history'][-1][1]:.6f} GeV^2")
errors2 = [abs(c-n)/abs(c)*100 for c,n in zip(d2['cornell_values'], d2['nn_values'])]
print(f"  Ortalama hata: {sum(errors2)/len(errors2):.2f}%")
print(f"  Max hata: {max(errors2):.2f}%")

print("\n" + "=" * 80)
print("SONUC:")
improvement = (1 - d2['loss_history'][-1][1]/d1['loss_history'][-1][1]) * 100
error_reduction = sum(errors1)/len(errors1) - sum(errors2)/len(errors2)
print(f"  Kayip iyilesmesi: {improvement:.1f}% daha iyi")
print(f"  Ortalama hata azalmasi: {error_reduction:.1f} puan")
print("=" * 80)

print("\nDETAYLI KARSILASTIRMA:")
print(f"{'Mesafe':>8} | {'Cornell':>9} | {'Eski NN':>9} | {'Yeni NN':>9} | {'Eski Hata':>10} | {'Yeni Hata':>10}")
print("-" * 80)
for i, d in enumerate(d1['test_distances']):
    c = d1['cornell_values'][i]
    n1 = d1['nn_values'][i]
    n2 = d2['nn_values'][i]
    e1 = errors1[i]
    e2 = errors2[i]
    improvement_mark = "✓" if e2 < e1 else "✗"
    print(f"{d:7.1f} | {c:9.3f} | {n1:9.3f} | {n2:9.3f} | {e1:9.2f}% | {e2:9.2f}% {improvement_mark}")
