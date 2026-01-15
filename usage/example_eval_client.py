# --------------------------------------------------------------
# Note:
#
# Please launch the server before running this code.
#
# ```shell
# sh launch_auto_eval_server.sh
# ```
# --------------------------------------------------------------

from adtools.evaluator.auto_server import submit_code


def test_client():
    # Correct code
    code_correct = """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2              
    left = merge_sort(arr[:mid])     
    right = merge_sort(arr[mid:])   
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"""

    print("Sending request with correct code...")
    response = submit_code(host="0.0.0.0", port=8000, code=code_correct, timeout=10)
    print("Response:", response)

    # Incorrect code (or simple return) to check result
    code_simple = """
def merge_sort(arr):
    return sorted(arr)
"""
    print("\nSending request with simple sorted() code...")
    response = submit_code(host="0.0.0.0", port=8000, code=code_simple, timeout=10)
    print("Response:", response)

    # Infinite loop
    code_loop = """
def merge_sort(arr):
    while True: pass
"""
    print("\nSending request with infinite loop (should timeout)...")
    response = submit_code(host="0.0.0.0", port=8000, code=code_loop, timeout=10)
    print("Response:", response)


if __name__ == "__main__":
    test_client()
