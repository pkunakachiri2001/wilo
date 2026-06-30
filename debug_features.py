import json, joblib, numpy as np
import warnings
warnings.filterwarnings('ignore')

fc_raw = joblib.load('models/feature_columns.pkl')
fc  = fc_raw.tolist() if hasattr(fc_raw, 'tolist') else list(fc_raw)
sc  = joblib.load('models/standard_scaler.pkl')
iso = joblib.load('models/isolation_forest.pkl')

sensors = {}
for sensor in ('acceleration', 'audio', 'current'):
    with open(f'Data/{sensor}.jsonl') as f:
        lines = f.readlines()
    max_rows = [json.loads(l) for l in lines if json.loads(l).get('file_type') == 'max']
    if max_rows:
        sensors[sensor] = max_rows[-1]
        r = max_rows[-1]
        print(f"{sensor}: freq1={r.get('frequency1')}, rms={r.get('rms'):.4f}, load={r.get('load_factor')}")

row = []
for col in fc:
    placed = False
    for sensor in ('acceleration', 'audio', 'current'):
        if col.startswith(sensor + '_'):
            suffix = col[len(sensor)+1:]
            val = sensors.get(sensor, {}).get(suffix, 0.0)
            row.append(float(val) if val is not None else 0.0)
            placed = True
            break
    if not placed:
        row.append(0.0)

X = np.array(row).reshape(1, -1)

print()
print("Feature vector vs training mean (z-score):")
for col, val, mean, scale in zip(fc, row, sc.mean_, sc.scale_):
    z = (val - mean) / scale if scale > 0 else 0
    flag = " <-- OUTLIER" if abs(z) > 3 else ""
    print(f"  {col:45s} val={val:10.4f}  mean={mean:10.4f}  z={z:7.2f}{flag}")

X_scaled = sc.transform(X)
score = iso.decision_function(X_scaled)[0]
print()
print("decision_function score:", score)
print("is_anomaly:", score < 0.0)
