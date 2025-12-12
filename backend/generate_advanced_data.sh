#!/bin/bash
# Script to generate realistic ED data using the advanced generator

echo "🚀 Generating realistic ED data with advanced generator..."
echo "Features: SDOH integration, iterative validation, tuned parameters"
echo ""

docker compose exec backend python -c "
import sys
sys.path.insert(0, '/app')
from generate_sample_data_advanced import generate_events_validated, write_csv
import logging
logging.basicConfig(level=logging.INFO)

events, validation = generate_events_validated(num_patients=500, days=2, max_iterations=5)
write_csv(events, '/app/sample_data.csv')

print(f'\n✅ Generated {len(events)} events from {len(set(e[\"patient_id\"] for e in events))} patients')
print(f'📊 Validation: {validation.get(\"pass_rate\", 0)*100:.1f}% pass rate')

lwbs_count = sum(1 for e in events if e[\"event_type\"] == \"lwbs\")
total_patients = len(set(e['patient_id'] for e in events))
print(f'\n📈 Statistics:')
print(f'  Total patients: {total_patients}')
print(f'  LWBS rate: {lwbs_count / total_patients * 100:.1f}% (target: 1.1-1.8%)')
print(f'  Total events: {len(events)}')
print(f'\n💾 Saved to: /app/sample_data.csv')
print(f'📤 Ready to upload via the frontend!')
"

echo ""
echo "📋 To use this data:"
echo "1. The file is saved inside the Docker container at /app/sample_data.csv"
echo "2. Upload it via the frontend Chat page (Upload CSV Data button)"
echo "3. Or copy it to your host: docker compose cp backend:/app/sample_data.csv ./sample_data_advanced.csv"

