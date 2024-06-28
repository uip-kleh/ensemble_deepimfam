import os

if __name__ == "__main__":
    entry_path = "/home/mizuno/data/aaindex_entry.txt"
    base_path = "/home/mizuno/data/mizuta_NORM_INDEX"
    with open(entry_path, "r") as f:
        for entry in f.read().splitlines():
            hisotry_path = os.path.join(os.path.join(base_path, entry), "result/FOCAL_LOSS/history/csv")
            print(hisotry_path)
