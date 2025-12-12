"""
Generate synthetic ED event data for testing.
Realistic data generation based on clinical patterns and distributions.
"""
import csv
import random
from datetime import datetime, timedelta
from faker import Faker
import numpy as np

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)


# Emergency Severity Index (ESI) distribution (real-world)
ESI_DISTRIBUTION = {
    1: 0.02,  # 2% - Resuscitation
    2: 0.08,  # 8% - Emergent
    3: 0.40,  # 40% - Urgent
    4: 0.35,  # 35% - Less urgent
    5: 0.15   # 15% - Non-urgent
}

# Ambulance arrival rate
AMBULANCE_RATE = 0.20

# Admission rate by ESI
ADMISSION_RATE_BY_ESI = {
    1: 0.80,
    2: 0.50,
    3: 0.25,
    4: 0.10,
    5: 0.05
}

# Lab need rate by ESI
LAB_RATE_BY_ESI = {
    1: 0.95,
    2: 0.90,
    3: 0.70,
    4: 0.40,
    5: 0.20
}

# Imaging need rate by ESI
IMAGING_RATE_BY_ESI = {
    1: 0.80,
    2: 0.60,
    3: 0.35,
    4: 0.15,
    5: 0.05
}


def assign_esi():
    """Assign ESI level based on real-world distribution."""
    rand = random.random()
    cumulative = 0.0
    for esi, prob in ESI_DISTRIBUTION.items():
        cumulative += prob
        if rand <= cumulative:
            return esi
    return 3


def get_arrival_rate(hour: int, is_weekend: bool = False) -> float:
    """
    Get arrival rate based on time of day and day of week.
    Real-world patterns: Peak hours 2-6 PM, 8-10 PM.
    """
    base_rate = 12.0  # Average patients/hour
    weekend_mult = 1.15 if is_weekend else 1.0
    
    if (14 <= hour < 18) or (20 <= hour < 22):  # Peak
        return base_rate * 1.8 * weekend_mult
    elif (10 <= hour < 14) or (18 <= hour < 20):  # Moderate
        return base_rate * 1.3 * weekend_mult
    elif 2 <= hour < 6:  # Off-peak
        return base_rate * 0.5 * weekend_mult
    else:
        return base_rate * weekend_mult


def get_service_time(base_time: float, esi: int, use_lognormal: bool = True) -> float:
    """
    Get service time using log-normal distribution (realistic for healthcare).
    Higher acuity (lower ESI) = longer service time.
    """
    esi_multiplier = {
        1: 1.8,
        2: 1.5,
        3: 1.0,
        4: 0.7,
        5: 0.5
    }.get(esi, 1.0)
    
    mean_time = base_time * esi_multiplier
    
    if use_lognormal:
        # Log-normal distribution
        sigma = 0.3
        mu = np.log(mean_time) - (sigma ** 2) / 2
        service_time = np.random.lognormal(mu, sigma)
    else:
        # Normal distribution with bounds
        service_time = np.random.normal(mean_time, mean_time * 0.2)
    
    return max(base_time * 0.3, service_time)


def generate_events(num_patients=250, start_date=None, days=2):
    """
    Generate synthetic ED events with realistic patterns.
    
    Args:
        num_patients: Target number of patients
        start_date: Start date (default: 2 days ago)
        days: Number of days to generate
    """
    if start_date is None:
        start_date = datetime.utcnow() - timedelta(days=days)
    
    events = []
    patient_journeys = {}
    
    # Generate patients with realistic arrival patterns
    current_time = start_date
    patient_id_counter = 0
    
    # Generate arrivals over the time period
    for day_offset in range(days):
        day_start = start_date + timedelta(days=day_offset)
        is_weekend = (day_start.weekday() >= 5)  # Saturday or Sunday
        
        # Generate arrivals for each hour
        for hour in range(24):
            hour_start = day_start + timedelta(hours=hour)
            arrival_rate = get_arrival_rate(hour, is_weekend)
            
            # Generate Poisson arrivals for this hour
            num_arrivals = np.random.poisson(arrival_rate)
            
            for _ in range(num_arrivals):
                # Random arrival time within the hour
                arrival_time = hour_start + timedelta(
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                )
                
                patient_id = f"anon_patient_{patient_id_counter:06d}"
                patient_id_counter += 1
                
                # Assign patient characteristics
                esi = assign_esi()
                is_ambulance = random.random() < AMBULANCE_RATE
                
                # ESI 1-2 and ambulance arrivals bypass triage
                skip_triage = (esi <= 2) or is_ambulance
                
                patient_journeys[patient_id] = {
                    "arrival": arrival_time,
                    "esi": esi,
                    "is_ambulance": is_ambulance,
                    "skip_triage": skip_triage,
                    "triage": None,
                    "doctor": None,
                    "labs": None,
                    "imaging": None,
                    "bed": None,
                    "discharge": None,
                    "lwbs": False
                }
    
    # Process each patient through ED stages
    for patient_id, journey in patient_journeys.items():
        current_time = journey["arrival"]
        esi = journey["esi"]
        
        # Arrival event
        events.append({
            "timestamp": current_time,
            "event_type": "arrival",
            "patient_id": patient_id,
            "stage": "triage" if not journey["skip_triage"] else "doctor",
            "resource_type": None,
            "resource_id": None,
            "duration_minutes": None,
            "esi": esi,
            "is_ambulance": journey["is_ambulance"]
        })
        
        # Triage (only for walk-in, non-critical patients)
        if not journey["skip_triage"]:
            # Triage wait time (0-10 minutes, varies by ESI)
            triage_wait = np.random.exponential(3.0) if esi >= 4 else np.random.exponential(1.0)
            current_time += timedelta(minutes=min(triage_wait, 10))
            
            # Triage service time (3-8 minutes)
            triage_time = get_service_time(5.0, esi)
            triage_time = max(3.0, min(triage_time, 10.0))
            
            events.append({
                "timestamp": current_time,
                "event_type": "triage",
                "patient_id": patient_id,
                "stage": "triage",
                "resource_type": "nurse",
                "resource_id": f"nurse_{random.randint(1, 3)}",
                "duration_minutes": round(triage_time, 1),
                "esi": esi
            })
            current_time += timedelta(minutes=triage_time)
        
        # Doctor visit wait time (varies significantly by ESI and queue)
        # TUNED: Reduced wait times to lower LWBS rate (target: 1.1-1.8% vs 18%)
        # ESI 1-2: 0-5 min, ESI 3: 10-25 min, ESI 4-5: 15-40 min (reduced from 20-60)
        doctor_wait_times = {
            1: (0, 5),
            2: (0, 8),
            3: (10, 25),
            4: (15, 35),
            5: (20, 40)  # Reduced from 30-60
        }
        wait_min, wait_max = doctor_wait_times.get(esi, (10, 25))
        doctor_wait = np.random.lognormal(np.log((wait_min + wait_max) / 2), 0.3)  # Reduced variance
        doctor_wait = max(wait_min, min(doctor_wait, wait_max))
        current_time += timedelta(minutes=doctor_wait)
        
        # Check for LWBS (patients leave if wait too long)
        # TUNED: Doubled thresholds and reduced base probability to target 1.1-1.8% LWBS rate
        total_wait = (current_time - journey["arrival"]).total_seconds() / 60
        lwbs_threshold = {
            1: float('inf'),
            2: 120.0,  # Doubled from 60
            3: 90.0,   # Doubled from 45
            4: 60.0,   # Doubled from 30
            5: 40.0    # Doubled from 20
        }.get(esi, 60.0)
        
        if total_wait > lwbs_threshold:
            # Reduced base from 0.3 to 0.1, and scaling from 0.4 to 0.2
            lwbs_prob = min(0.1 + (total_wait - lwbs_threshold) / 100.0 * 0.2, 0.5)  # Cap at 50% max
            if random.random() < lwbs_prob:
                events.append({
                    "timestamp": current_time,
                    "event_type": "lwbs",
                    "patient_id": patient_id,
                    "stage": None,
                    "resource_type": None,
                    "resource_id": None,
                    "duration_minutes": None,
                    "esi": esi
                })
                journey["lwbs"] = True
                continue  # Patient left, no further events
        
        # Doctor visit
        doctor_base_times = {
            1: 45.0,
            2: 30.0,
            3: 20.0,
            4: 12.0,
            5: 8.0
        }
        doctor_time = get_service_time(doctor_base_times.get(esi, 20.0), esi)
        doctor_time = max(5.0, doctor_time)
        
        events.append({
            "timestamp": current_time,
            "event_type": "doctor_visit",
            "patient_id": patient_id,
            "stage": "doctor",
            "resource_type": "doctor",
            "resource_id": f"doctor_{random.randint(1, 3)}",
            "duration_minutes": round(doctor_time, 1),
            "esi": esi
        })
        current_time += timedelta(minutes=doctor_time)
        
        # Labs (if needed)
        needs_labs = random.random() < LAB_RATE_BY_ESI.get(esi, 0.60)
        if needs_labs:
            # Lab wait time (5-20 minutes)
            lab_wait = np.random.exponential(8.0)
            current_time += timedelta(minutes=min(lab_wait, 20))
            
            # Lab processing time (15-45 minutes)
            lab_time = np.random.lognormal(np.log(25), 0.4)
            lab_time = max(10.0, min(lab_time, 60.0))
            
            events.append({
                "timestamp": current_time,
                "event_type": "labs",
                "patient_id": patient_id,
                "stage": "labs",
                "resource_type": "lab_tech",
                "resource_id": f"lab_tech_1",
                "duration_minutes": round(lab_time, 1),
                "esi": esi
            })
            current_time += timedelta(minutes=lab_time)
        
        # Imaging (if needed)
        needs_imaging = random.random() < IMAGING_RATE_BY_ESI.get(esi, 0.30)
        if needs_imaging:
            # Imaging wait time (10-30 minutes)
            imaging_wait = np.random.exponential(15.0)
            current_time += timedelta(minutes=min(imaging_wait, 30))
            
            # Imaging time (20-40 minutes)
            imaging_time = np.random.lognormal(np.log(25), 0.3)
            imaging_time = max(15.0, min(imaging_time, 60.0))
            
            events.append({
                "timestamp": current_time,
                "event_type": "imaging",
                "patient_id": patient_id,
                "stage": "imaging",
                "resource_type": "tech",
                "resource_id": f"tech_{random.randint(1, 2)}",
                "duration_minutes": round(imaging_time, 1),
                "esi": esi
            })
            current_time += timedelta(minutes=imaging_time)
        
        # Bed assignment (only for admitted patients)
        needs_bed = random.random() < ADMISSION_RATE_BY_ESI.get(esi, 0.20)
        if needs_bed:
            # Bed wait time (0-30 minutes)
            bed_wait = np.random.exponential(10.0)
            current_time += timedelta(minutes=min(bed_wait, 30))
            
            events.append({
                "timestamp": current_time,
                "event_type": "bed_assign",
                "patient_id": patient_id,
                "stage": "bed",
                "resource_type": "bed",
                "resource_id": f"bed_{random.randint(1, 20)}",
                "duration_minutes": None,
                "esi": esi
            })
            
            # Bed stay time (TUNED: Increased for realistic LOS - target 4-5h avg)
            # Base times increased, with behavioral health tail (9-10h for 5% of admits)
            bed_base = {
                1: 420.0,  # 7h (increased from 6h)
                2: 360.0,  # 6h (increased from 5h)
                3: 300.0,  # 5h (increased from 4h)
                4: 240.0,  # 4h (increased from 3h)
                5: 180.0   # 3h (increased from 2h)
            }.get(esi, 300.0)
            
            # 5% of admits have behavioral health tail (9-10h)
            is_behavioral_health = random.random() < 0.05
            if is_behavioral_health:
                bed_base = 540.0  # 9h base for behavioral health
            
            bed_time = np.random.lognormal(np.log(bed_base), 0.4)
            bed_time = max(120.0, min(bed_time, 600.0))  # 2-10 hours (increased from 1-8h)
            current_time += timedelta(minutes=bed_time)
        
        # Discharge
        events.append({
            "timestamp": current_time,
            "event_type": "discharge",
            "patient_id": patient_id,
            "stage": None,
            "resource_type": None,
            "resource_id": None,
            "duration_minutes": None,
            "esi": esi
        })
    
    # Sort by timestamp
    events.sort(key=lambda x: x["timestamp"])
    
    return events


def write_csv(events, filename="sample_data.csv"):
    """Write events to CSV file."""
    with open(filename, "w", newline="") as f:
        fieldnames = [
            "timestamp", "event_type", "patient_id", "stage",
            "resource_type", "resource_id", "duration_minutes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow({
                "timestamp": event["timestamp"].isoformat(),
                "event_type": event["event_type"],
                "patient_id": event["patient_id"],
                "stage": event["stage"] or "",
                "resource_type": event.get("resource_type") or "",
                "resource_id": event.get("resource_id") or "",
                "duration_minutes": event.get("duration_minutes") or "",
            })


if __name__ == "__main__":
    print("Generating realistic synthetic ED events...")
    # Generate 2 days of data (~500-600 patients)
    events = generate_events(num_patients=500, days=2)
    write_csv(events, "sample_data.csv")
    print(f"Generated {len(events)} events from {len(set(e['patient_id'] for e in events))} patients")
    print(f"Saved to sample_data.csv")
    
    # Print statistics
    lwbs_count = sum(1 for e in events if e["event_type"] == "lwbs")
    print(f"\nStatistics:")
    print(f"  Total patients: {len(set(e['patient_id'] for e in events))}")
    print(f"  LWBS rate: {lwbs_count / len(set(e['patient_id'] for e in events)) * 100:.1f}%")
    print(f"  Total events: {len(events)}")
