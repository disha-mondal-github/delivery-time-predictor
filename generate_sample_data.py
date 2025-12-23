import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_delivery_data(num_records=20000, num_customers=500):
    """
    Generate realistic sample delivery data with complex real-world patterns:
    - Pareto distribution (20% of users make 80% of orders)
    - Multiple addresses (Home vs Work)
    - Logic constraints (No bulky items at night)
    - Location correlations (Business districts prefer day slots)
    """
    
    # 1. Expanded Name Lists (2,500 unique combos)
    first_names = [
        'John', 'Sarah', 'Michael', 'Emily', 'David', 'Jessica', 'James', 'Ashley', 
        'Robert', 'Amanda', 'William', 'Jennifer', 'Daniel', 'Lisa', 'Matthew', 'Michelle', 
        'Christopher', 'Laura', 'Andrew', 'Stephanie', 'Joseph', 'Rebecca', 'Joshua', 
        'Sharon', 'Ryan', 'Cynthia', 'Nicholas', 'Kathleen', 'Ben', 'Amy', 'Luke', 
        'Anna', 'Jacob', 'Margaret', 'Patrick', 'Sandra', 'Alexander', 'Betty', 'Isaac', 
        'Helen', 'Thomas', 'Nancy', 'Timothy', 'Donna', 'Samuel', 'Karen', 'Kevin', 
        'Carol', 'Brian', 'Ruth'
    ]
    
    last_names = [
        'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 
        'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 
        'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 
        'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker', 
        'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 
        'Flores', 'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 
        'Mitchell', 'Carter', 'Roberts'
    ]
    
    # 2. Define Neighborhood Zones
    # Residential Zones (Prefer evenings/weekends)
    residential_zones = ['10001', '10002', '10003']
    # Business Zones (Prefer 9-5 weekdays)
    business_zones = ['20001', '20002']

    # 3. Create Customers with Profiles
    customers = []
    customer_weights = [] # For Pareto distribution

    for i in range(num_customers):
        # Pareto Logic: 20% of customers are "Power Users" (High frequency)
        is_power_user = (i < num_customers * 0.2)
        weight = 80 if is_power_user else 5
        customer_weights.append(weight)

        # Address Logic: 30% of people have a Work address too
        has_work_address = random.random() < 0.3
        
        addresses = [{'type': 'home', 'postal': random.choice(residential_zones)}]
        if has_work_address:
            addresses.append({'type': 'work', 'postal': random.choice(business_zones)})

        customer = {
            'customer_id': f'CUST_{i+1:04d}',
            'customer_name': f"{random.choice(first_names)} {random.choice(last_names)}",
            'addresses': addresses,
            # Base preference (overridden by work logic often)
            'preference_type': random.choice(['morning', 'afternoon', 'evening', 'flexible'])
        }
        customers.append(customer)
    
    # Time slots
    time_slots = [
        '09:00-12:00', '12:00-15:00', '15:00-18:00', 
        '18:00-21:00', '21:00-23:00'
    ]
    
    # Preference mappings
    preference_slots = {
        'morning': ['09:00-12:00', '12:00-15:00'],
        'afternoon': ['12:00-15:00', '15:00-18:00'],
        'evening': ['18:00-21:00', '21:00-23:00'],
        'flexible': time_slots
    }
    
    order_types = ['standard', 'express', 'bulky', 'fragile']
    
    records = []
    start_date = datetime.now() - timedelta(days=180)
    
    print("Generating orders (this may take a moment due to complex logic)...")
    
    # Select customers based on weights (Pareto Principle)
    selected_customers = random.choices(customers, weights=customer_weights, k=num_records)

    for customer in selected_customers:
        # Generate random date
        random_days = random.randint(0, 180)
        delivery_date = start_date + timedelta(days=random_days)
        is_weekend = delivery_date.weekday() >= 5
        
        # Select Address for this specific order
        address = random.choice(customer['addresses'])
        postal_code = address['postal']
        addr_type = address['type']
        
        order_type = random.choice(order_types)
        
        # --- COMPLEX LOGIC FOR TIME SLOTS ---
        
        valid_slots = time_slots.copy()
        
        # Rule 1: Bulky items cannot be delivered late at night
        if order_type == 'bulky':
            if '21:00-23:00' in valid_slots: valid_slots.remove('21:00-23:00')
            if '18:00-21:00' in valid_slots and random.random() < 0.5: 
                valid_slots.remove('18:00-21:00') # 50% chance to remove evening too

        # Rule 2: Business addresses prefer working hours
        if addr_type == 'work' and not is_weekend:
            # Heavily favor 9-5
            preferred_slots = ['09:00-12:00', '12:00-15:00', '15:00-18:00']
        elif is_weekend:
            # Weekends favor afternoon/evening
            preferred_slots = ['12:00-15:00', '15:00-18:00', '18:00-21:00']
        else:
            # Standard Home/Weekday logic
            preferred_slots = preference_slots[customer['preference_type']]
        
        # Intersect valid slots with preferred slots
        available_preferred = [s for s in preferred_slots if s in valid_slots]
        
        # Selection Logic
        if available_preferred and random.random() < 0.85:
            time_slot = random.choice(available_preferred)
        else:
            time_slot = random.choice(valid_slots)
            
        # Success Logic (Correlation)
        # Business deliveries on weekends often fail
        if addr_type == 'work' and is_weekend:
            was_successful = 1 if random.random() < 0.2 else 0 # High fail rate
        elif time_slot in preferred_slots:
            was_successful = 1 if random.random() < 0.95 else 0
        else:
            was_successful = 1 if random.random() < 0.6 else 0
            
        record = {
            'delivery_id': f'DEL_{len(records)+1:06d}',
            'customer_id': customer['customer_id'],
            'customer_name': customer['customer_name'],
            'delivery_date': delivery_date.strftime('%Y-%m-%d'),
            'time_slot': time_slot,
            'postal_code': postal_code, # Can vary for same customer now!
            'order_type': order_type,
            'order_value': round(random.uniform(20, 500), 2),
            'was_successful': was_successful
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df = df.sort_values('delivery_date').reset_index(drop=True)
    return df

if __name__ == '__main__':
    # Generate data
    df = generate_sample_delivery_data(num_records=20000, num_customers=500)
    
    output_path = 'data/delivery_history.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nGenerated {len(df)} records")
    print(f"Pareto Check: Top 20% of customers made {df['customer_id'].value_counts().head(100).sum() / len(df):.1%} of orders")
    print(f"Address Check: {df['postal_code'].nunique()} active zones")