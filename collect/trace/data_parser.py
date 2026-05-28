import subprocess
import csv
import datetime
import os
"""
trước khi chạy, kiểm tra ổ đĩa đích: df -h .
chỉnh biến SECOND_TO_TRACE để trace
test bằng cách:
- tải block random: dd if=/dev/urandom of=dummy_data.bin bs=1M count=200
- search web (đổi PREFIX thành local_firefox): 
sudo apt update
sudo apt install firefox
firefox &
"""

def capture_and_parse(device, duration_k, output_csv, prefix="local"):
    temp_trace_base = "temp_trace"
    temp_parsed_txt = "temp_parsed.txt"

    print(f"[*] Step 1: Starting blktrace on {device} for {duration_k} seconds...")
    print("    (Please do some activities like browsing or coding now!)")
    
    try:
        trace_cmd = ["sudo", "blktrace", "-d", device, "-w", str(duration_k), "-o", temp_trace_base]
        subprocess.run(trace_cmd, check=True)

        print("[*] Step 2: Tracing complete. Parsing binary data to text...")
        # convert binary to readable text
        parse_cmd = ["blkparse", "-i", temp_trace_base, "-o", temp_parsed_txt]
        subprocess.run(parse_cmd, check=True)

        print(f"[*] Step 3: Appending text to Baleen-formatted CSV: {output_csv}...")
        
        # Grab the time right now, formatted as "YYYY-MM-DD HH:MM"
        session_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        # Check if the file already exists and has content
        file_exists = os.path.isfile(output_csv) and os.path.getsize(output_csv) > 0

        # Changed 'w' to 'a' for Append mode
        with open(temp_parsed_txt, 'r') as infile, open(output_csv, 'a', newline='') as outfile:
            writer = csv.writer(outfile)
            
            # Only write the header if the file is completely new
            if not file_exists:
                writer.writerow(['Session_ID', 
                                 'Block_ID', 
                                 'IO_Offset', 
                                 'IO_Size', 
                                 'Op_Name', 
                                 'User'])

            for line in infile:
                parts = line.split()
                
                if len(parts) >= 10:
                    action = parts[5]
                    
                    #  Only record 'C' (Completed) requests to avoid duplicates
                    if action == 'C':
                        try:
                            # 1. Extract physical data
                            raw_pid_user = parts[4]
                            rw_type = parts[6]
                            sector = int(parts[7])
                            blocks = int(parts[9])

                            # 2. Convert to Baleen Logic & Apply Namespace Prefix
                            block_id = sector
                            io_offset = sector * 512       # 512 bytes per sector
                            io_size = blocks * 512         # Convert blocks to bytes
                            op_name = 'PUT' if 'W' in rw_type else 'GET' # W=Write, R=Read
                        
                            unique_user = f"{prefix}_{raw_pid_user}"

                            # 3. Write to CSV (Matching the 6-column format exactly)
                            writer.writerow([session_id, block_id, io_offset, io_size, op_name, unique_user])
                        
                        except ValueError:
                            continue

        print("[*] Step 4: Cleaning up temporary files...")
        for file in os.listdir('.'):
            if temp_trace_base in file or file == temp_parsed_txt:
                os.remove(file)
                
        print(f"[SUCCESS] Your training data is ready at: {output_csv}")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] A command failed: {e}")

if __name__ == "__main__":
    TARGET_DEVICE = "/dev/sdd" 
    SECONDS_TO_TRACE = 120
    # output csv path
    OUTPUT_CSV_NAME = "/home/roofredriver/os_project/collect/data/local_trace.csv"

    capture_and_parse(TARGET_DEVICE, SECONDS_TO_TRACE, OUTPUT_CSV_NAME, prefix="local")