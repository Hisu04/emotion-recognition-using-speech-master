import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import shutil
import sys

# Hỗ trợ in tiếng Việt trên terminal Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def clean_old_models():
    # Thư mục chứa các mô hình pickle cũ
    grid_dir = "grid"
    
    # Danh sách các file cần xóa (file .pickle không tương thích)
    files_to_remove = ["best_classifiers.pickle", "best_regressors.pickle"]
    
    print("--- Dọn dẹp các mô hình cũ không tương thích ---")
    
    deleted_count = 0
    if os.path.exists(grid_dir):
        for filename in files_to_remove:
            file_path = os.path.join(grid_dir, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"[x] Đã xóa: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"[!] Lỗi khi xóa {file_path}: {e}")
    
    if deleted_count == 0:
        print("[!] Không tìm thấy mô hình cũ nào cần xóa.")
    else:
        print(f"[+] Đã dọn dẹp {deleted_count} file.")
        print("\n[GỢI Ý] Bây giờ bạn hãy chạy lệnh sau để huấn luyện lại mô hình tương thích:")
        print("python grid_search.py")

if __name__ == "__main__":
    clean_old_models()
