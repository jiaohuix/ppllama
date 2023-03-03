import sys
from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    assert len(sys.argv) == 2, f"usage: python {sys.argv[0]} <reprod_file1>  <reprod_file2>"
    reprod_file1 = sys.argv[1]
    reprod_file2 = sys.argv[2]
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info(reprod_file1)
    paddle_info = diff_helper.load_info(reprod_file2)

    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="forward_diff.log")