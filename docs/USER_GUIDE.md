# User Guide

## Getting Started

### 1. Start the Application

```bash
docker-compose up
```

Wait for all services to start (about 30-60 seconds), then open http://localhost:3000

### 2. Ingest Sample Data

1. Navigate to the **Chat** page
2. Click "📁 Upload CSV Data"
3. Select `sample_data.csv` (generated via `python backend/generate_sample_data.py`)
4. Wait for confirmation message

Alternatively, use the API:
```bash
curl -X POST "http://localhost:8000/api/ingest/csv" \
  -F "file=@sample_data.csv"
```

### 3. View Dashboard

Navigate to the **Dashboard** to see:
- Real-time KPIs (DTD, LOS, LWBS, Bed Utilization)
- Historical trends
- Detected bottlenecks

---

## Using Natural Language Queries

### Basic Queries

Ask questions in plain English on the **Chat** page:

**Examples:**
- "What if we add two nurses during peak hours on weekends?"
- "Simulate adding a triage nurse from 2-6 PM"
- "What happens if we add one doctor on Saturday?"
- "Remove one bed and see the impact"

### Query Structure

The system understands:
- **Actions**: add, remove, shift, modify
- **Resources**: nurse, doctor, bed, tech
- **Quantities**: numbers (e.g., "2 nurses")
- **Time**: "from 2-6 PM", "14:00 to 18:00"
- **Days**: "Saturday", "weekends", "Monday"

### Interpreting Results

After a simulation, you'll see:
- **Baseline metrics**: Current performance
- **Predicted metrics**: Expected after change
- **Deltas**: Percentage changes (negative = improvement)
- **Confidence**: Simulation reliability (0-100%)

**Example Output:**
```
DTD Change: -20.0%  (Good - reduction in wait time)
LOS Change: -8.3%   (Good - shorter stay)
LWBS Drop: -37.5%   (Excellent - fewer patients leaving)
Confidence: 85%
```

---

## Dashboard Features

### Real-Time KPIs

Four key metrics cards:
1. **Door-to-Doctor (DTD)**: Time from arrival to first doctor visit
   - Threshold: 30 minutes (red if exceeded)
2. **Length of Stay (LOS)**: Total time in ED
   - Threshold: 180 minutes
3. **LWBS Rate**: Percentage leaving without being seen
   - Threshold: 5%
4. **Bed Utilization**: Percentage of beds in use
   - Threshold: 90%

### Anomaly Alerts

Yellow alert boxes appear when metrics spike:
- Shows which metric (DTD, LOS, LWBS)
- Displays Z-score (statistical significance)
- Indicates severity (medium/high)

### Charts

- **DTD & LOS Trends**: Line chart showing historical performance
- **LWBS & Bed Utilization**: Bar chart for comparison

### Bottlenecks

Cards showing:
- **Bottleneck Name**: Stage (e.g., "Imaging Queue")
- **Wait Time**: Current delay in minutes
- **Impact Score**: 0-1 scale (higher = more critical)
- **Severity**: low, medium, high, critical
- **Causes**: Root cause analysis
- **Recommendations**: Actionable suggestions

---

## Simulation History

The **History** page shows:
- All past simulations
- Query text
- Results (DTD, LOS, LWBS changes)
- Confidence scores
- Export to CSV

---

## Best Practices

### 1. Start with Data

Always ingest historical data before running simulations. The baseline metrics depend on real data.

### 2. Be Specific

Better queries:
- ✅ "Add 2 nurses from 2-6 PM on Saturday"
- ❌ "Add some nurses"

### 3. Compare Scenarios

Run multiple simulations and compare:
- Baseline vs. Scenario A vs. Scenario B
- Use History page to track changes

### 4. Check Confidence

If confidence < 70%, refine your query or check data quality.

### 5. Monitor Bottlenecks

Regularly check Dashboard for new bottlenecks, especially after:
- Staffing changes
- Peak hours
- High patient volumes

---

## Troubleshooting

### "No data available"

**Solution**: Ingest sample data via Chat page or API.

### "Low confidence parsing"

**Solution**: Be more specific in your query. Include:
- Resource type (nurse, doctor, etc.)
- Quantity (number)
- Time range (if applicable)

### "Simulation failed"

**Solution**: 
1. Check API health: http://localhost:8000/api/health
2. Verify data is ingested
3. Try a simpler query

### Slow Performance

**Solution**:
- Reduce simulation iterations (default: 100)
- Use shorter time windows (24h vs 48h)
- Check ClickHouse and Redis are running

---

## Advanced Usage

### API Integration

Use the API directly for automation:

```python
import requests

# Run simulation
response = requests.post(
    "http://localhost:8000/api/simulate/nlp",
    json={"query": "Add 2 nurses from 2-6 PM"}
)
result = response.json()
```

### Custom Constraints

For optimization, specify constraints:

```json
{
  "bottlenecks": ["Imaging Queue"],
  "constraints": {
    "staff_max": 10,
    "budget": 1000
  },
  "objective": "minimize_dtd"
}
```

---

## Support

For issues:
1. Check logs: `docker-compose logs backend`
2. Review API docs: http://localhost:8000/docs
3. Open a GitHub issue

---

**Happy optimizing! 🚀**


