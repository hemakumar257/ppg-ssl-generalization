import sys

def read_log(path, num_lines=100):
    encodings = ['utf-16', 'utf-16-le', 'utf-16-be', 'utf-8', 'latin-1']
    content = None
    
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.readlines()
            print(f"--- Read successfully with {enc} ---")
            break
        except Exception:
            continue
            
    if content:
        for line in content[-num_lines:]:
            try:
                # Use 'replace' for the console print
                clean_line = line.encode(sys.stdout.encoding if sys.stdout.encoding else 'ascii', errors='replace').decode(sys.stdout.encoding if sys.stdout.encoding else 'ascii')
                print(clean_line.strip())
            except Exception:
                print("[Un-encodable line]")
    else:
        print("Failed to read file with any tested encoding.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        read_log(path, n)
    else:
        print("Usage: python read_log.py <path> [num_lines]")
