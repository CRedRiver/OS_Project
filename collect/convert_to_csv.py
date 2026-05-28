import pandas as pd
import datetime
import os

def convert_trace_to_csv(txt_path, csv_path, region, max_rows=10000):
    """
    Parses a Meta Baleen trace file, extracts relevant IO features, 
    namespaces the User ID, and exports to a CSV file.
    """
    print(f"[*] Converting raw trace {txt_path} to CSV...")

    rows = []
    count = 0

    try:
        with open(txt_path, 'r') as infile:
            for line in infile:
                if count >= max_rows:  # Extract limit to a variable
                    break

                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                
                # Ensure enough columns exist before parsing
                if len(parts) >= 7:
                    try:
                        # Convert to float first to handle traces with decimal timestamps
                        timestamp = float(parts[3])
                        dt_utc = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
                        session = dt_utc.strftime("%Y-%m-%d %H:%M")
                    except ValueError:
                        session = "Unknown_Session"
                    
                    unique_user = f"{region}_{parts[6]}"

                    rows.append({
                        'Session_ID': session,
                        'Block_ID': parts[0],
                        'IO_Offset': parts[1],
                        'IO_Size': parts[2],
                        'Op_Name': parts[4],
                        'User': unique_user
                    })

                    count += 1
                    
    except FileNotFoundError:
        print(f"[!] File not found: {txt_path}")
        return

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=[
        'Session_ID', 'Block_ID', 'IO_Offset', 
        'IO_Size', 'Op_Name', 'User'
    ])

    # Safely create output directories if they don't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Export to CSV
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Processed {count} rows. Saved to {csv_path}")


if __name__ == "__main__":
    regions = ["region3", "region4", "region5", "region6", "region7"]
    TRACES_ROOT = "/home/roofredriver/os_project/traces"
    OUTPUT_CSV_ROOT = "/home/roofredriver/os_project/collect/data"
    
    for region in regions:
        input_file = os.path.join(TRACES_ROOT, f"full_1_1_{region}.trace")
        output_file = os.path.join(OUTPUT_CSV_ROOT, f"1_1_{region}.csv")
        
        try:
            convert_trace_to_csv(input_file, output_file, region)
        except Exception as e:
            print(f"[ERROR] Converting trace from {region} failed: {e}")