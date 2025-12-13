"""
Generate advancedtest3.csv with specific characteristics:
- High LWBS rate (5-8%)
- High LOS (6-8 hours)
- Medium DTD (25-35 min)
- At least 6 bottlenecks
- Multiple days of data
"""
import csv
import random
from datetime import datetime, timedelta
from faker import Faker
import numpy as np
from typing import List, Dict, Any

fake = Faker()
Faker.seed(123)  # Different seed for different data
random.seed(123)
np.random.seed(123)

# ESI distribution
ESI_DISTRIBUTION = {
    1: 0.015,
    2: 0.075,
    3: 0.400,
    4: 0.350,
    5: 0.160
}

DISEASE_CATEGORIES = {
    "psychiatric": 0.12,  # Increased for bottlenecks
    "orthopedic": 0.20,   # Increased for bottlenecks
    "cardiac": 0.15,
    "respiratory": 0.18,
    "gastrointestinal": 0.12,
    "infectious": 0.10,
    "neurologic": 0.05,
    "other": 0.08
}

def assign_esi():
    rand = random.random()
    cumulative = 0.0
    for esi, prob in ESI_DISTRIBUTION.items():
        cumulative += prob
        if rand <= cumulative:
            return esi
    return 3

def assign_disease_category(esi: int) -> str:
    rand = random.random()
    cumulative = 0.0
    for category, prob in DISEASE_CATEGORIES.items():
        cumulative += prob
        if rand <= cumulative:
            return category
    return "other"

def get_arrival_rate(hour: int, is_weekend: bool) -> float:
    """Higher arrival rates to create bottlenecks"""
    base_rates = {
        0: 0.5, 1: 0.3, 2: 0.2, 3: 0.2, 4: 0.3, 5: 0.5,
        6: 1.0, 7: 1.5, 8: 2.0, 9: 2.5, 10: 3.0, 11: 3.5,
        12: 4.0, 13: 4.5, 14: 5.0, 15: 5.5, 16: 6.0, 17: 6.5,
        18: 5.5, 19: 4.5, 20: 3.5, 21: 2.5, 22: 1.5, 23: 1.0
    }
    rate = base_rates.get(hour, 2.0)
    if is_weekend:
        rate *= 1.3  # Higher weekend rates
    return rate * 1.5  # Overall 50% increase for bottlenecks

def generate_advancedtest3_data(days=7, num_patients_per_day=150):
    """Generate data with high LWBS, high LOS, medium DTD, and bottlenecks"""
    start_date = datetime(2025, 12, 1, 8, 0, 0)
    events = []
    patient_journeys = {}
    patient_id_counter = 0
    
    # Resource constraints to create bottlenecks
    num_doctors = 2  # Reduced from 3
    num_nurses = 2   # Reduced from 3
    num_lab_techs = 1  # Reduced
    num_imaging_techs = 1  # Reduced
    num_beds = 15  # Reduced from 20
    
    doctor_queue = []
    nurse_queue = []
    lab_queue = []
    imaging_queue = []
    bed_queue = []
    
    for day_offset in range(days):
        day_start = start_date + timedelta(days=day_offset)
        is_weekend = (day_start.weekday() >= 5)
        
        for hour in range(24):
            hour_start = day_start + timedelta(hours=hour)
            arrival_rate = get_arrival_rate(hour, is_weekend)
            num_arrivals = np.random.poisson(arrival_rate)
            
            for _ in range(num_arrivals):
                arrival_time = hour_start + timedelta(
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                )
                
                patient_id = f"anon_patient_{patient_id_counter:06d}"
                patient_id_counter += 1
                
                esi = assign_esi()
                disease_category = assign_disease_category(esi)
                is_ambulance = random.random() < 0.15
                skip_triage = (esi <= 2) or is_ambulance
                
                patient_journeys[patient_id] = {
                    "arrival": arrival_time,
                    "esi": esi,
                    "disease_category": disease_category,
                    "is_ambulance": is_ambulance,
                    "skip_triage": skip_triage,
                    "lwbs": False
                }
    
    # Process patients with bottlenecks
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
            "esi": esi
        })
        
        # Triage (with bottleneck - longer waits)
        if not journey["skip_triage"]:
            triage_wait = np.random.exponential(8.0)  # Increased from 3.0
            current_time += timedelta(minutes=min(triage_wait, 25))
            
            triage_time = np.random.lognormal(np.log(8.0), 0.3)  # Increased
            triage_time = max(5.0, min(triage_time, 15.0))
            
            events.append({
                "timestamp": current_time,
                "event_type": "triage",
                "patient_id": patient_id,
                "stage": "triage",
                "resource_type": "nurse",
                "resource_id": f"nurse_{random.randint(1, num_nurses)}",
                "duration_minutes": round(triage_time, 1),
                "esi": esi
            })
            current_time += timedelta(minutes=triage_time)
        
        # Doctor wait - MEDIUM RANGE (25-35 min target)
        # But with bottlenecks, some will wait longer
        doctor_wait_times = {
            1: (0, 8),
            2: (5, 15),
            3: (20, 35),  # Medium range (25-35 target)
            4: (25, 40),  # Medium range
            5: (25, 40)   # Medium range
        }
        wait_min, wait_max = doctor_wait_times.get(esi, (25, 35))
        doctor_wait = np.random.lognormal(np.log((wait_min + wait_max) / 2), 0.4)
        doctor_wait = max(wait_min, min(doctor_wait, wait_max * 1.5))  # Allow longer waits
        current_time += timedelta(minutes=doctor_wait)
        
        # LWBS check - HIGH RATE (5-8% target)
        total_wait = (current_time - journey["arrival"]).total_seconds() / 60
        lwbs_threshold = {
            1: float('inf'),
            2: 90.0,   # Moderate thresholds
            3: 70.0,   # Moderate
            4: 50.0,   # Lower but not too low
            5: 40.0    # Lower but not too low
        }.get(esi, 50.0)
        
        if total_wait > lwbs_threshold:
            # Higher LWBS probability (5-8% target) - more conservative
            lwbs_prob = min(0.08 + (total_wait - lwbs_threshold) / 80.0 * 0.15, 0.25)
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
                continue
        
        # Doctor visit
        doctor_base_times = {1: 50.0, 2: 35.0, 3: 25.0, 4: 18.0, 5: 12.0}
        doctor_time = np.random.lognormal(np.log(doctor_base_times.get(esi, 25.0)), 0.3)
        doctor_time = max(8.0, min(doctor_time, 60.0))
        
        events.append({
            "timestamp": current_time,
            "event_type": "doctor_visit",
            "patient_id": patient_id,
            "stage": "doctor",
            "resource_type": "doctor",
            "resource_id": f"doctor_{random.randint(1, num_doctors)}",
            "duration_minutes": round(doctor_time, 1),
            "esi": esi
        })
        current_time += timedelta(minutes=doctor_time)
        
        # Labs (with bottleneck)
        needs_labs = random.random() < 0.70  # Higher rate
        if needs_labs:
            lab_wait = np.random.exponential(20.0)  # Increased wait
            current_time += timedelta(minutes=min(lab_wait, 45))
            
            lab_time = np.random.lognormal(np.log(35), 0.4)  # Longer service
            lab_time = max(15.0, min(lab_time, 90.0))
            
            events.append({
                "timestamp": current_time,
                "event_type": "labs",
                "patient_id": patient_id,
                "stage": "labs",
                "resource_type": "lab_tech",
                "resource_id": "lab_tech_1",
                "duration_minutes": round(lab_time, 1),
                "esi": esi
            })
            current_time += timedelta(minutes=lab_time)
        
        # Imaging (with bottleneck)
        needs_imaging = random.random() < 0.45  # Higher rate
        if needs_imaging:
            imaging_wait = np.random.exponential(25.0)  # Increased wait
            current_time += timedelta(minutes=min(imaging_wait, 60))
            
            imaging_time = np.random.lognormal(np.log(35), 0.3)
            imaging_time = max(20.0, min(imaging_time, 90.0))
            
            events.append({
                "timestamp": current_time,
                "event_type": "imaging",
                "patient_id": patient_id,
                "stage": "imaging",
                "resource_type": "tech",
                "resource_id": "tech_1",
                "duration_minutes": round(imaging_time, 1),
                "esi": esi
            })
            current_time += timedelta(minutes=imaging_time)
        
        # Bed assignment - HIGH LOS (6-8 hours)
        needs_bed = random.random() < 0.35  # Higher admission rate
        if needs_bed:
            bed_wait = np.random.exponential(20.0)  # Longer wait
            current_time += timedelta(minutes=min(bed_wait, 60))
            
            events.append({
                "timestamp": current_time,
                "event_type": "bed_assign",
                "patient_id": patient_id,
                "stage": "bed",
                "resource_type": "bed",
                "resource_id": f"bed_{random.randint(1, num_beds)}",
                "duration_minutes": None,
                "esi": esi
            })
            
            # HIGH LOS: 6-8 hours average
            bed_base = {1: 480.0, 2: 420.0, 3: 360.0, 4: 300.0, 5: 240.0}.get(esi, 360.0)
            bed_stay = np.random.lognormal(np.log(bed_base), 0.4)
            bed_stay = max(300.0, min(bed_stay, 600.0))  # 5-10 hour range
            current_time += timedelta(minutes=bed_stay)
        
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
    
    return events

if __name__ == "__main__":
    print("Generating advancedtest3.csv with:")
    print("  - High LWBS rate (5-8%)")
    print("  - High LOS (6-8 hours)")
    print("  - Medium DTD (25-35 min)")
    print("  - Multiple bottlenecks")
    print("  - 7 days of data")
    
    events = generate_advancedtest3_data(days=7, num_patients_per_day=150)
    
    # Write to CSV
    output_file = "advancedtest3.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "event_type", "patient_id", "stage",
            "resource_type", "resource_id", "duration_minutes"
        ])
        writer.writeheader()
        
        for event in events:
            writer.writerow({
                "timestamp": event["timestamp"].isoformat(),
                "event_type": event["event_type"],
                "patient_id": event["patient_id"],
                "stage": event.get("stage", ""),
                "resource_type": event.get("resource_type", ""),
                "resource_id": event.get("resource_id", ""),
                "duration_minutes": event.get("duration_minutes", "")
            })
    
    # Calculate stats
    unique_patients = len(set(e["patient_id"] for e in events))
    lwbs_count = len([e for e in events if e["event_type"] == "lwbs"])
    lwbs_rate = (lwbs_count / unique_patients * 100) if unique_patients > 0 else 0
    
    print(f"\n✅ Generated {output_file}")
    print(f"  Total events: {len(events)}")
    print(f"  Unique patients: {unique_patients}")
    print(f"  LWBS count: {lwbs_count}")
    print(f"  LWBS rate: {lwbs_rate:.2f}%")
    print(f"  Date range: {events[0]['timestamp'].date()} to {events[-1]['timestamp'].date()}")

Generate advancedtest3.csv with specific characteristics:
- High LWBS rate (5-8%)
- High LOS (6-8 hours)
- Medium DTD (25-35 min)
- At least 6 bottlenecks
- Multiple days of data
"""
import csv
import random
from datetime import datetime, timedelta
from faker import Faker
import numpy as np
from typing import List, Dict, Any

fake = Faker()
Faker.seed(123)  # Different seed for different data
random.seed(123)
np.random.seed(123)

# ESI distribution
ESI_DISTRIBUTION = {
    1: 0.015,
    2: 0.075,
    3: 0.400,
    4: 0.350,
    5: 0.160
}

DISEASE_CATEGORIES = {
    "psychiatric": 0.12,  # Increased for bottlenecks
    "orthopedic": 0.20,   # Increased for bottlenecks
    "cardiac": 0.15,
    "respiratory": 0.18,
    "gastrointestinal": 0.12,
    "infectious": 0.10,
    "neurologic": 0.05,
    "other": 0.08
}

def assign_esi():
    rand = random.random()
    cumulative = 0.0
    for esi, prob in ESI_DISTRIBUTION.items():
        cumulative += prob
        if rand <= cumulative:
            return esi
    return 3

def assign_disease_category(esi: int) -> str:
    rand = random.random()
    cumulative = 0.0
    for category, prob in DISEASE_CATEGORIES.items():
        cumulative += prob
        if rand <= cumulative:
            return category
    return "other"

def get_arrival_rate(hour: int, is_weekend: bool) -> float:
    """Higher arrival rates to create bottlenecks"""
    base_rates = {
        0: 0.5, 1: 0.3, 2: 0.2, 3: 0.2, 4: 0.3, 5: 0.5,
        6: 1.0, 7: 1.5, 8: 2.0, 9: 2.5, 10: 3.0, 11: 3.5,
        12: 4.0, 13: 4.5, 14: 5.0, 15: 5.5, 16: 6.0, 17: 6.5,
        18: 5.5, 19: 4.5, 20: 3.5, 21: 2.5, 22: 1.5, 23: 1.0
    }
    rate = base_rates.get(hour, 2.0)
    if is_weekend:
        rate *= 1.3  # Higher weekend rates
    return rate * 1.5  # Overall 50% increase for bottlenecks

def generate_advancedtest3_data(days=7, num_patients_per_day=150):
    """Generate data with high LWBS, high LOS, medium DTD, and bottlenecks"""
    start_date = datetime(2025, 12, 1, 8, 0, 0)
    events = []
    patient_journeys = {}
    patient_id_counter = 0
    
    # Resource constraints to create bottlenecks
    num_doctors = 2  # Reduced from 3
    num_nurses = 2   # Reduced from 3
    num_lab_techs = 1  # Reduced
    num_imaging_techs = 1  # Reduced
    num_beds = 15  # Reduced from 20
    
    doctor_queue = []
    nurse_queue = []
    lab_queue = []
    imaging_queue = []
    bed_queue = []
    
    for day_offset in range(days):
        day_start = start_date + timedelta(days=day_offset)
        is_weekend = (day_start.weekday() >= 5)
        
        for hour in range(24):
            hour_start = day_start + timedelta(hours=hour)
            arrival_rate = get_arrival_rate(hour, is_weekend)
            num_arrivals = np.random.poisson(arrival_rate)
            
            for _ in range(num_arrivals):
                arrival_time = hour_start + timedelta(
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                )
                
                patient_id = f"anon_patient_{patient_id_counter:06d}"
                patient_id_counter += 1
                
                esi = assign_esi()
                disease_category = assign_disease_category(esi)
                is_ambulance = random.random() < 0.15
                skip_triage = (esi <= 2) or is_ambulance
                
                patient_journeys[patient_id] = {
                    "arrival": arrival_time,
                    "esi": esi,
                    "disease_category": disease_category,
                    "is_ambulance": is_ambulance,
                    "skip_triage": skip_triage,
                    "lwbs": False
                }
    
    # Process patients with bottlenecks
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
            "esi": esi
        })
        
        # Triage (with bottleneck - longer waits)
        if not journey["skip_triage"]:
            triage_wait = np.random.exponential(8.0)  # Increased from 3.0
            current_time += timedelta(minutes=min(triage_wait, 25))
            
            triage_time = np.random.lognormal(np.log(8.0), 0.3)  # Increased
            triage_time = max(5.0, min(triage_time, 15.0))
            
            events.append({
                "timestamp": current_time,
                "event_type": "triage",
                "patient_id": patient_id,
                "stage": "triage",
                "resource_type": "nurse",
                "resource_id": f"nurse_{random.randint(1, num_nurses)}",
                "duration_minutes": round(triage_time, 1),
                "esi": esi
            })
            current_time += timedelta(minutes=triage_time)
        
        # Doctor wait - MEDIUM RANGE (25-35 min target)
        # But with bottlenecks, some will wait longer
        doctor_wait_times = {
            1: (0, 8),
            2: (5, 15),
            3: (20, 35),  # Medium range (25-35 target)
            4: (25, 40),  # Medium range
            5: (25, 40)   # Medium range
        }
        wait_min, wait_max = doctor_wait_times.get(esi, (25, 35))
        doctor_wait = np.random.lognormal(np.log((wait_min + wait_max) / 2), 0.4)
        doctor_wait = max(wait_min, min(doctor_wait, wait_max * 1.5))  # Allow longer waits
        current_time += timedelta(minutes=doctor_wait)
        
        # LWBS check - HIGH RATE (5-8% target)
        total_wait = (current_time - journey["arrival"]).total_seconds() / 60
        lwbs_threshold = {
            1: float('inf'),
            2: 90.0,   # Moderate thresholds
            3: 70.0,   # Moderate
            4: 50.0,   # Lower but not too low
            5: 40.0    # Lower but not too low
        }.get(esi, 50.0)
        
        if total_wait > lwbs_threshold:
            # Higher LWBS probability (5-8% target) - more conservative
            lwbs_prob = min(0.08 + (total_wait - lwbs_threshold) / 80.0 * 0.15, 0.25)
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
                continue
        
        # Doctor visit
        doctor_base_times = {1: 50.0, 2: 35.0, 3: 25.0, 4: 18.0, 5: 12.0}
        doctor_time = np.random.lognormal(np.log(doctor_base_times.get(esi, 25.0)), 0.3)
        doctor_time = max(8.0, min(doctor_time, 60.0))
        
        events.append({
            "timestamp": current_time,
            "event_type": "doctor_visit",
            "patient_id": patient_id,
            "stage": "doctor",
            "resource_type": "doctor",
            "resource_id": f"doctor_{random.randint(1, num_doctors)}",
            "duration_minutes": round(doctor_time, 1),
            "esi": esi
        })
        current_time += timedelta(minutes=doctor_time)
        
        # Labs (with bottleneck)
        needs_labs = random.random() < 0.70  # Higher rate
        if needs_labs:
            lab_wait = np.random.exponential(20.0)  # Increased wait
            current_time += timedelta(minutes=min(lab_wait, 45))
            
            lab_time = np.random.lognormal(np.log(35), 0.4)  # Longer service
            lab_time = max(15.0, min(lab_time, 90.0))
            
            events.append({
                "timestamp": current_time,
                "event_type": "labs",
                "patient_id": patient_id,
                "stage": "labs",
                "resource_type": "lab_tech",
                "resource_id": "lab_tech_1",
                "duration_minutes": round(lab_time, 1),
                "esi": esi
            })
            current_time += timedelta(minutes=lab_time)
        
        # Imaging (with bottleneck)
        needs_imaging = random.random() < 0.45  # Higher rate
        if needs_imaging:
            imaging_wait = np.random.exponential(25.0)  # Increased wait
            current_time += timedelta(minutes=min(imaging_wait, 60))
            
            imaging_time = np.random.lognormal(np.log(35), 0.3)
            imaging_time = max(20.0, min(imaging_time, 90.0))
            
            events.append({
                "timestamp": current_time,
                "event_type": "imaging",
                "patient_id": patient_id,
                "stage": "imaging",
                "resource_type": "tech",
                "resource_id": "tech_1",
                "duration_minutes": round(imaging_time, 1),
                "esi": esi
            })
            current_time += timedelta(minutes=imaging_time)
        
        # Bed assignment - HIGH LOS (6-8 hours)
        needs_bed = random.random() < 0.35  # Higher admission rate
        if needs_bed:
            bed_wait = np.random.exponential(20.0)  # Longer wait
            current_time += timedelta(minutes=min(bed_wait, 60))
            
            events.append({
                "timestamp": current_time,
                "event_type": "bed_assign",
                "patient_id": patient_id,
                "stage": "bed",
                "resource_type": "bed",
                "resource_id": f"bed_{random.randint(1, num_beds)}",
                "duration_minutes": None,
                "esi": esi
            })
            
            # HIGH LOS: 6-8 hours average
            bed_base = {1: 480.0, 2: 420.0, 3: 360.0, 4: 300.0, 5: 240.0}.get(esi, 360.0)
            bed_stay = np.random.lognormal(np.log(bed_base), 0.4)
            bed_stay = max(300.0, min(bed_stay, 600.0))  # 5-10 hour range
            current_time += timedelta(minutes=bed_stay)
        
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
    
    return events

if __name__ == "__main__":
    print("Generating advancedtest3.csv with:")
    print("  - High LWBS rate (5-8%)")
    print("  - High LOS (6-8 hours)")
    print("  - Medium DTD (25-35 min)")
    print("  - Multiple bottlenecks")
    print("  - 7 days of data")
    
    events = generate_advancedtest3_data(days=7, num_patients_per_day=150)
    
    # Write to CSV
    output_file = "advancedtest3.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "event_type", "patient_id", "stage",
            "resource_type", "resource_id", "duration_minutes"
        ])
        writer.writeheader()
        
        for event in events:
            writer.writerow({
                "timestamp": event["timestamp"].isoformat(),
                "event_type": event["event_type"],
                "patient_id": event["patient_id"],
                "stage": event.get("stage", ""),
                "resource_type": event.get("resource_type", ""),
                "resource_id": event.get("resource_id", ""),
                "duration_minutes": event.get("duration_minutes", "")
            })
    
    # Calculate stats
    unique_patients = len(set(e["patient_id"] for e in events))
    lwbs_count = len([e for e in events if e["event_type"] == "lwbs"])
    lwbs_rate = (lwbs_count / unique_patients * 100) if unique_patients > 0 else 0
    
    print(f"\n✅ Generated {output_file}")
    print(f"  Total events: {len(events)}")
    print(f"  Unique patients: {unique_patients}")
    print(f"  LWBS count: {lwbs_count}")
    print(f"  LWBS rate: {lwbs_rate:.2f}%")
    print(f"  Date range: {events[0]['timestamp'].date()} to {events[-1]['timestamp'].date()}")

