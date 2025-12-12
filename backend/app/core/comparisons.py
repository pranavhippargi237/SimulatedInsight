import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any, Tuple

from app.data.storage import get_events

logger = logging.getLogger(__name__)


def _day_name(dt: datetime) -> str:
    return dt.strftime("%A")


async def compare_days(
    days: List[str],
    window_hours: int = 720
) -> Dict[str, Any]:
    """
    Compare metrics across specified day names (e.g., ["Saturday","Monday"]).
    Also supports "Weekday" (Mon-Fri) and "Weekend" (Sat-Sun) as special cases.
    Aggregates over the last `window_hours` from now.
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=window_hours)

    events = await get_events(start_time, end_time, raise_if_empty=True)
    if not events:
        return {"error": "no_events", "message": "No events available for comparison"}

    # Handle weekday/weekend special cases
    weekday_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekend_days = ["Saturday", "Sunday"]
    
    # Track which groups we're comparing
    comparing_weekday = any(d.lower() == "weekday" for d in days)
    comparing_weekend = any(d.lower() == "weekend" for d in days)
    
    # Expand weekday/weekend to actual day names for event matching
    expanded_days = []
    for day in days:
        day_lower = day.lower()
        if day_lower == "weekday":
            expanded_days.extend(weekday_days)
        elif day_lower == "weekend":
            expanded_days.extend(weekend_days)
        else:
            expanded_days.append(day)
    
    # Normalize day names - include weekday/weekend as special keys
    days_norm = {d.lower(): d for d in expanded_days}
    if comparing_weekday:
        days_norm["weekday"] = "Weekday"
    if comparing_weekend:
        days_norm["weekend"] = "Weekend"

    aggregates: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(float))
    patients_by_day: Dict[str, set] = defaultdict(set)
    lwbs_patients_by_day: Dict[str, set] = defaultdict(set)

    # Track wait/duration by stage
    duration_sums: Dict[Tuple[str, str], float] = defaultdict(float)
    duration_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for e in events:
        ts = e.get("timestamp")
        if not ts:
            continue
        try:
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
        except Exception:
            continue
        day = _day_name(ts)
        day_lower = day.lower()
        
        # Determine which group this day belongs to
        target_group = None
        if day_lower in days_norm:
            # Direct day match
            target_group = day
        elif comparing_weekday and day in weekday_days:
            target_group = "Weekday"
        elif comparing_weekend and day in weekend_days:
            target_group = "Weekend"
        
        if not target_group:
            continue

        patient_id = e.get("patient_id")
        event_type = e.get("event_type")
        duration = e.get("duration_minutes")

        # Arrivals
        if event_type == "arrival" and patient_id:
            patients_by_day[day].add(patient_id)

        # LWBS
        if event_type == "lwbs" and patient_id:
            lwbs_patients_by_day[day].add(patient_id)

        # Duration by stage
        if duration is not None and event_type in {"doctor_visit", "imaging", "labs", "triage", "bed_assign"}:
            key = (day, event_type)
            duration_sums[key] += float(duration)
            duration_counts[key] += 1

    # Build aggregates - handle weekday/weekend grouping
    unique_days = set()
    for day in days:
        day_lower = day.lower()
        if day_lower == "weekday":
            unique_days.add("Weekday")
        elif day_lower == "weekend":
            unique_days.add("Weekend")
        else:
            # For specific days, use the actual day name
            for d in expanded_days:
                if d.lower() == day_lower:
                    unique_days.add(d)
                    break
    
    # If we have weekday/weekend, aggregate across all matching days
    for target_day in unique_days:
        if target_day == "Weekday":
            # Aggregate all weekday data
            all_weekday_patients = set()
            all_weekday_lwbs = set()
            weekday_duration_sums = defaultdict(float)
            weekday_duration_counts = defaultdict(int)
            
            for wd in weekday_days:
                all_weekday_patients.update(patients_by_day.get(wd, set()))
                all_weekday_lwbs.update(lwbs_patients_by_day.get(wd, set()))
                for stage in ["doctor_visit", "imaging", "labs", "triage", "bed_assign"]:
                    key = (wd, stage)
                    if duration_counts.get(key, 0) > 0:
                        weekday_duration_sums[stage] += duration_sums[key]
                        weekday_duration_counts[stage] += duration_counts[key]
            
            arrivals = len(all_weekday_patients)
            lwbs_patients = len(all_weekday_lwbs)
            lwbs_rate = (lwbs_patients / arrivals) if arrivals else 0.0
            
            metrics = {
                "arrivals": arrivals,
                "lwbs_rate": lwbs_rate,
                "mean_waits": {}
            }
            
            for stage in ["doctor_visit", "imaging", "labs", "triage", "bed_assign"]:
                if weekday_duration_counts.get(stage, 0) > 0:
                    metrics["mean_waits"][stage] = weekday_duration_sums[stage] / weekday_duration_counts[stage]
            
            aggregates["Weekday"] = metrics
            
        elif target_day == "Weekend":
            # Aggregate all weekend data
            all_weekend_patients = set()
            all_weekend_lwbs = set()
            weekend_duration_sums = defaultdict(float)
            weekend_duration_counts = defaultdict(int)
            
            for wd in weekend_days:
                all_weekend_patients.update(patients_by_day.get(wd, set()))
                all_weekend_lwbs.update(lwbs_patients_by_day.get(wd, set()))
                for stage in ["doctor_visit", "imaging", "labs", "triage", "bed_assign"]:
                    key = (wd, stage)
                    if duration_counts.get(key, 0) > 0:
                        weekend_duration_sums[stage] += duration_sums[key]
                        weekend_duration_counts[stage] += duration_counts[key]
            
            arrivals = len(all_weekend_patients)
            lwbs_patients = len(all_weekend_lwbs)
            lwbs_rate = (lwbs_patients / arrivals) if arrivals else 0.0
            
            metrics = {
                "arrivals": arrivals,
                "lwbs_rate": lwbs_rate,
                "mean_waits": {}
            }
            
            for stage in ["doctor_visit", "imaging", "labs", "triage", "bed_assign"]:
                if weekend_duration_counts.get(stage, 0) > 0:
                    metrics["mean_waits"][stage] = weekend_duration_sums[stage] / weekend_duration_counts[stage]
            
            aggregates["Weekend"] = metrics
        else:
            # Regular day aggregation
            arrivals = len(patients_by_day.get(target_day, set()))
            lwbs_patients = len(lwbs_patients_by_day.get(target_day, set()))
            lwbs_rate = (lwbs_patients / arrivals) if arrivals else 0.0

            metrics = {
                "arrivals": arrivals,
                "lwbs_rate": lwbs_rate,
                "mean_waits": {}
            }

            for stage in ["doctor_visit", "imaging", "labs", "triage", "bed_assign"]:
                key = (target_day, stage)
                if duration_counts.get(key, 0) > 0:
                    metrics["mean_waits"][stage] = duration_sums[key] / duration_counts[key]

            aggregates[target_day] = metrics

    # Compute diffs against the first day as baseline
    baseline_day = "Weekday" if "weekday" in [d.lower() for d in days] else ("Weekend" if "weekend" in [d.lower() for d in days] else (days[0] if days else None))
    if baseline_day:
        baseline = aggregates.get(baseline_day, {})
        diffs = {}
        for day in days_norm.values():
            if day == days[0] or day not in aggregates:
                continue
            current = aggregates[day]
            diff_entry = {
                "arrivals_delta": current.get("arrivals", 0) - baseline.get("arrivals", 0),
                "arrivals_pct": pct_change(current.get("arrivals", 0), baseline.get("arrivals", 0)),
                "lwbs_delta": current.get("lwbs_rate", 0) - baseline.get("lwbs_rate", 0),
                "lwbs_pct": pct_change(current.get("lwbs_rate", 0), baseline.get("lwbs_rate", 0)),
                "mean_waits_delta": {}
            }
            for stage in set(list(current.get("mean_waits", {}).keys()) + list(baseline.get("mean_waits", {}).keys())):
                cur = current.get("mean_waits", {}).get(stage, 0)
                base = baseline.get("mean_waits", {}).get(stage, 0)
                diff_entry["mean_waits_delta"][stage] = {
                    "delta": cur - base,
                    "pct": pct_change(cur, base)
                }
            diffs[day] = diff_entry
    else:
        diffs = {}

    # Determine baseline day name for return
    baseline_name = None
    if days:
        if days[0].lower() == "weekday":
            baseline_name = "Weekday"
        elif days[0].lower() == "weekend":
            baseline_name = "Weekend"
        else:
            baseline_name = days[0]
    
    return {
        "aggregates": aggregates,
        "diffs": diffs,
        "baseline": baseline_name
    }


def pct_change(current: float, base: float) -> float:
    if base == 0:
        return 0.0
    return ((current - base) / base) * 100.0


async def detect_spikes(
    window_hours: int = 168,
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Detect volume spikes by hour and day-of-week over the last window_hours.
    Returns top hours by count and basic z-score outliers.
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=window_hours)
    events = await get_events(start_time, end_time, raise_if_empty=True)
    if not events:
        return {"error": "no_events", "message": "No events available for spike analysis"}

    hour_counts: Dict[int, int] = defaultdict(int)
    dow_counts: Dict[str, int] = defaultdict(int)

    for e in events:
        ts = e.get("timestamp")
        if not ts:
            continue
        try:
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
        except Exception:
            continue
        hour_counts[ts.hour] += 1
        dow_counts[_day_name(ts)] += 1

    # Compute z-scores for hours
    hours = list(hour_counts.keys())
    counts = [hour_counts[h] for h in hours]
    mean = sum(counts) / len(counts) if counts else 0
    std = (sum((c - mean) ** 2 for c in counts) / len(counts)) ** 0.5 if counts else 0

    spikes = []
    for h in hours:
        z = (hour_counts[h] - mean) / std if std else 0
        spikes.append({"hour": h, "count": hour_counts[h], "z": z})

    spikes_sorted = sorted(spikes, key=lambda x: x["z"], reverse=True)[:top_n]
    dows_sorted = sorted(dow_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "top_hours": spikes_sorted,
        "top_dows": [{"day": d, "count": c} for d, c in dows_sorted],
        "mean_hourly": mean,
        "std_hourly": std
    }

